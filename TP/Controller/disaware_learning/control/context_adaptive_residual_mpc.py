"""Context-adaptive residual MPC for the frozen Stage-2 physics probe.

The controller accepts an absolute command plan from either a joystick
adapter or a planner, and searches only a short, low-dimensional residual
around that plan.  The plan is re-anchored to the current reference on every
tick.  PID is not part of the optimization model; it remains a fallback and a
benchmark elsewhere in the stack.

Stage 2 currently exposes inference-only latent rollouts, so this controller
uses batched finite-difference coordinate search.  This is deliberately
small-horizon MPC rather than a second policy: every accepted command has
already passed the full nonlinear Stage-2 rollout and actuator projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as functional
except ImportError as error:  # pragma: no cover
    raise ImportError("context-adaptive residual MPC requires PyTorch") from error

from disaware_learning.learning.motor_jepa_data import (
    DEPTH_INDEX,
    FEATURE_CONTRACT_ID,
    IMU_ACCEL_SLICE,
    OBSERVATION_LABELS,
    ROTATION_6D_SLICE,
    Standardizer,
    VIO_VELOCITY_SLICE,
    YAW_RATE_INDEX,
)
from disaware_learning.learning.stage2 import (
    IMPLEMENTATION as STAGE2_IMPLEMENTATION,
    load_stage2_checkpoint,
    rotation6d_to_matrix,
)
from disaware_learning.control.jepa_mppi import standard_bluerov2_manual_mixer


IMPLEMENTATION_ID = "context_adaptive_residual_mpc_v1"
COMMAND_ORDER = ("surge", "sway", "heave", "yaw")


def _tuple(values: object, length: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} finite values") from error
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class MpcReference:
    """A source-agnostic execution reference.

    ``command_normalized`` is always required and is the command a joystick
    or planner would execute without assistance.  Optional physical targets
    are supplied by a planner adapter when available; omitting them keeps the
    controller a command-response enhancer instead of inventing a healthy
    reference.
    """

    command_normalized: np.ndarray
    body_velocity_m_s: np.ndarray | None = None
    body_velocity_mask: np.ndarray | None = None
    yaw_rate_rad_s: np.ndarray | None = None
    yaw_rate_mask: np.ndarray | None = None
    position_ned_m: np.ndarray | None = None
    position_mask: np.ndarray | None = None

    @classmethod
    def joystick(
        cls,
        command_normalized: np.ndarray,
        *,
        body_velocity_m_s: np.ndarray | None = None,
        yaw_rate_rad_s: np.ndarray | None = None,
    ) -> "MpcReference":
        return cls(
            command_normalized=np.asarray(command_normalized, dtype=np.float32),
            body_velocity_m_s=body_velocity_m_s,
            yaw_rate_rad_s=yaw_rate_rad_s,
        )

    @classmethod
    def planner(
        cls,
        command_normalized: np.ndarray,
        *,
        body_velocity_m_s: np.ndarray | None = None,
        yaw_rate_rad_s: np.ndarray | None = None,
        position_ned_m: np.ndarray | None = None,
        position_mask: np.ndarray | None = None,
    ) -> "MpcReference":
        return cls(
            command_normalized=np.asarray(command_normalized, dtype=np.float32),
            body_velocity_m_s=body_velocity_m_s,
            yaw_rate_rad_s=yaw_rate_rad_s,
            position_ned_m=position_ned_m,
            position_mask=position_mask,
        )


@dataclass(frozen=True)
class ContextAdaptiveResidualMpcConfig:
    command_hold_steps: int = 5
    control_horizon_seconds: float = 0.8
    # One batch evaluates baseline, the current residual, and both finite
    # differences for every short-horizon coefficient.  It fits the 25 Hz
    # deployment budget on the retained 75-step probe; callers may increase
    # this only after measuring their target hardware.
    iterations: int = 1
    finite_difference_step: tuple[float, ...] = (0.035, 0.035, 0.025, 0.025)
    residual_limit: tuple[float, ...] = (0.16, 0.16, 0.12, 0.12)
    manual_limit: tuple[float, ...] = (0.65, 0.65, 0.45, 0.50)
    command_slew_per_second: tuple[float, ...] = (1.2, 1.2, 1.0, 1.2)
    motor_soft_limit: float = 0.90
    warm_start_residual_weight: float = 0.0
    residual_weight: float = 0.08
    command_rate_weight: float = 0.25
    command_curvature_weight: float = 0.20
    jerk_weight: float = 0.35
    jerk_scale_m_s3: float = 1.0
    tilt_weight: float = 0.75
    tilt_scale_rad: float = math.radians(10.0)
    cross_axis_weight: float = 0.50
    body_velocity_tracking_weight: tuple[float, ...] = (10.0, 10.0, 12.0)
    body_velocity_tracking_scale_m_s: tuple[float, ...] = (0.15, 0.15, 0.12)
    yaw_rate_tracking_weight: float = 10.0
    yaw_rate_tracking_scale_rad_s: float = 0.30
    position_tracking_weight: tuple[float, ...] = (5.0, 5.0, 7.0)
    position_tracking_scale_m: tuple[float, ...] = (0.50, 0.50, 0.30)
    effort_weight: float = 0.02
    motor_saturation_weight: float = 10.0
    workspace_min_ned_m: tuple[float, ...] | None = None
    workspace_max_ned_m: tuple[float, ...] | None = None
    maximum_predicted_tilt_rad: float = math.radians(30.0)
    minimum_predicted_improvement: float = 0.001
    minimum_relative_improvement: float = 0.01
    minimum_execution_change: float = 1e-4
    deadline_seconds: float = 0.040

    def __post_init__(self) -> None:
        for name in (
            "finite_difference_step", "residual_limit", "manual_limit",
            "command_slew_per_second", "body_velocity_tracking_weight",
            "body_velocity_tracking_scale_m_s", "position_tracking_weight",
            "position_tracking_scale_m",
        ):
            values = _tuple(getattr(self, name), 4 if name in {
                "finite_difference_step", "residual_limit", "manual_limit",
                "command_slew_per_second",
            } else 3, name)
            if any(value < 0.0 for value in values):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, values)
        if any(value <= 0.0 for value in self.manual_limit):
            raise ValueError("manual limits must be positive")
        if any(value <= 0.0 for value in self.command_slew_per_second):
            raise ValueError("command slew limits must be positive")
        if self.command_hold_steps < 1 or self.control_horizon_seconds <= 0.0:
            raise ValueError("invalid MPC horizon")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not 0.0 <= self.warm_start_residual_weight < 1.0:
            raise ValueError("warm-start residual weight must be in [0, 1)")
        if not 0.0 < self.motor_soft_limit <= 1.0:
            raise ValueError("motor_soft_limit must be in (0, 1]")
        if self.minimum_execution_change < 0.0 or not math.isfinite(self.minimum_execution_change):
            raise ValueError("minimum_execution_change must be finite and nonnegative")
        positive = (
            "jerk_scale_m_s3", "tilt_scale_rad", "yaw_rate_tracking_scale_rad_s",
            "maximum_predicted_tilt_rad",
        )
        if any(float(getattr(self, name)) <= 0.0 for name in positive):
            raise ValueError("physical scales must be positive")
        if self.workspace_min_ned_m is not None:
            object.__setattr__(self, "workspace_min_ned_m", _tuple(self.workspace_min_ned_m, 3, "workspace_min_ned_m"))
        if self.workspace_max_ned_m is not None:
            object.__setattr__(self, "workspace_max_ned_m", _tuple(self.workspace_max_ned_m, 3, "workspace_max_ned_m"))
        if (self.workspace_min_ned_m is None) != (self.workspace_max_ned_m is None):
            raise ValueError("workspace bounds must be supplied together")
        if self.workspace_min_ned_m is not None and any(
            low >= high for low, high in zip(self.workspace_min_ned_m, self.workspace_max_ned_m)
        ):
            raise ValueError("workspace lower bounds must be below upper bounds")

    def as_dict(self) -> dict[str, Any]:
        return {"implementation_id": IMPLEMENTATION_ID, **asdict(self)}


def load_context_adaptive_residual_mpc_config(
    path: str | Path,
) -> ContextAdaptiveResidualMpcConfig:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("residual MPC config must contain a JSON object")
    implementation = payload.pop("implementation_id", IMPLEMENTATION_ID)
    if implementation != IMPLEMENTATION_ID:
        raise ValueError("residual MPC config implementation_id is incompatible")
    expected = set(ContextAdaptiveResidualMpcConfig.__dataclass_fields__)
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"unknown residual MPC config keys: {sorted(unknown)}")
    return ContextAdaptiveResidualMpcConfig(**payload)


@dataclass(frozen=True)
class ContextAdaptiveResidualMpcResult:
    command: np.ndarray
    planned_command: np.ndarray
    baseline_command: np.ndarray
    residual: np.ndarray
    accepted: bool
    reason: str
    diagnostics: dict[str, Any]


class ContextAdaptiveResidualMpc:
    """Short-horizon absolute-command enhancement using Stage-2 predictions."""

    def __init__(
        self,
        model: nn.Module,
        standardizer: Standardizer,
        *,
        history_steps: int,
        model_horizon_steps: int,
        sample_rate_hz: float,
        config: ContextAdaptiveResidualMpcConfig = ContextAdaptiveResidualMpcConfig(),
        device: str | torch.device = "cpu",
    ) -> None:
        if history_steps < 2 or model_horizon_steps < 1 or sample_rate_hz <= 0.0:
            raise ValueError("invalid Stage-2 time contract")
        self.device = torch.device(device)
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.standardizer = standardizer
        self.history_steps = int(history_steps)
        self.model_horizon_steps = int(model_horizon_steps)
        self.sample_rate_hz = float(sample_rate_hz)
        self.config = config
        self.control_horizon_steps = min(
            self.model_horizon_steps,
            max(1, int(round(config.control_horizon_seconds * self.sample_rate_hz))),
        )
        self.knot_count = max(1, math.ceil(self.control_horizon_steps / config.command_hold_steps))
        self._obs_mean = torch.as_tensor(standardizer.observation_mean, dtype=torch.float32, device=self.device)
        self._obs_std = torch.as_tensor(standardizer.observation_std, dtype=torch.float32, device=self.device)
        self._action_mean = torch.as_tensor(standardizer.action_mean, dtype=torch.float32, device=self.device)
        self._action_std = torch.as_tensor(standardizer.action_std, dtype=torch.float32, device=self.device)
        self._warm_residual = torch.zeros(self.knot_count, 4, dtype=torch.float32, device=self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: ContextAdaptiveResidualMpcConfig = ContextAdaptiveResidualMpcConfig(),
        device: str | torch.device = "cpu",
    ) -> "ContextAdaptiveResidualMpc":
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
        if checkpoint.get("feature_contract_id") != FEATURE_CONTRACT_ID:
            raise ValueError("checkpoint does not use the current feature contract")
        if checkpoint.get("implementation") != STAGE2_IMPLEMENTATION:
            raise ValueError("checkpoint is not the current retained Stage-2 model")
        readiness = checkpoint.get("training_readiness")
        if not isinstance(readiness, Mapping) or readiness.get("passed") is not True:
            raise ValueError("checkpoint failed Stage-2 training readiness")
        model, checkpoint = load_stage2_checkpoint(checkpoint_path, device=device)
        return cls(
            model,
            Standardizer.from_dict(checkpoint["standardizer"]),
            history_steps=int(checkpoint["history_steps"]),
            model_horizon_steps=int(checkpoint["horizon_steps"]),
            sample_rate_hz=float(checkpoint["sample_rate_hz"]),
            config=config,
            device=device,
        )

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    def reset(self) -> None:
        self._warm_residual.zero_()

    def _prepare_history(self, observation_history: np.ndarray, action_history: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        observation = np.asarray(observation_history, dtype=np.float32).copy()
        action = np.asarray(action_history, dtype=np.float32)
        if observation.shape != (self.history_steps, len(OBSERVATION_LABELS)):
            raise ValueError("observation history shape mismatch")
        if action.shape != (self.history_steps, 6):
            raise ValueError("action history shape mismatch")
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError("controller history contains non-finite values")
        current = torch.as_tensor(observation[-1], dtype=torch.float32, device=self.device)
        observation[:, DEPTH_INDEX] -= observation[-1, DEPTH_INDEX]
        obs = (torch.as_tensor(observation, device=self.device) - self._obs_mean) / self._obs_std
        act = (torch.as_tensor(action, device=self.device) - self._action_mean) / self._action_std
        return obs, act, current

    def _series(self, value: object | None, width: int, *, default: Tensor | None = None) -> Tensor | None:
        if value is None:
            return default
        array = np.asarray(value, dtype=np.float32)
        if width == 1 and array.ndim == 0:
            array = np.full((self.model_horizon_steps, 1), float(array), dtype=np.float32)
        if array.shape == (width,):
            array = np.repeat(array[None], self.model_horizon_steps, axis=0)
        elif array.shape == (1, width):
            array = np.repeat(array, self.model_horizon_steps, axis=0)
        elif array.shape == (self.knot_count, width):
            array = np.repeat(array, self.config.command_hold_steps, axis=0)
            if len(array) < self.model_horizon_steps:
                array = np.concatenate((array, np.repeat(array[-1:], self.model_horizon_steps - len(array), axis=0)))
            array = array[: self.model_horizon_steps]
        if array.shape != (self.model_horizon_steps, width) or not np.isfinite(array).all():
            raise ValueError(f"reference field must be [{self.model_horizon_steps}, {width}]")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _prepare_reference(self, reference: MpcReference) -> tuple[Tensor, dict[str, Tensor]]:
        command = np.asarray(reference.command_normalized, dtype=np.float32)
        if command.shape == (4,):
            command = np.repeat(command[None], self.model_horizon_steps, axis=0)
        elif command.shape == (1, 4):
            command = np.repeat(command, self.model_horizon_steps, axis=0)
        elif command.shape == (self.knot_count, 4):
            command = np.repeat(command, self.config.command_hold_steps, axis=0)
            if len(command) < self.model_horizon_steps:
                command = np.concatenate((command, np.repeat(command[-1:], self.model_horizon_steps - len(command), axis=0)))
            command = command[: self.model_horizon_steps]
        if command.shape != (self.model_horizon_steps, 4) or not np.isfinite(command).all():
            raise ValueError(f"reference command must be [4], [{self.knot_count},4], or [{self.model_horizon_steps},4]")
        targets: dict[str, Tensor] = {}
        for name, value, width in (
            ("body_velocity", reference.body_velocity_m_s, 3),
            ("yaw_rate", reference.yaw_rate_rad_s, 1),
            ("position", reference.position_ned_m, 3),
        ):
            target = self._series(value, width)
            if target is not None:
                targets[name] = target
        if reference.body_velocity_mask is not None:
            targets["body_velocity_mask"] = self._series(reference.body_velocity_mask, 3)
        if reference.yaw_rate_mask is not None:
            targets["yaw_rate_mask"] = self._series(reference.yaw_rate_mask, 1)
        if reference.position_mask is not None:
            targets["position_mask"] = self._series(reference.position_mask, 3)
        return torch.as_tensor(command, dtype=torch.float32, device=self.device), targets

    def _project(self, desired: Tensor, previous: Tensor) -> Tensor:
        """Project absolute commands onto bounds, mixer headroom, and slew."""
        limit = torch.as_tensor(self.config.manual_limit, device=self.device)
        slew = torch.as_tensor(self.config.command_slew_per_second, device=self.device)
        desired = desired.clamp(-limit, limit)
        motor, raw = standard_bluerov2_manual_mixer(desired)
        peak = raw[..., :4].abs().amax(dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(peak), self.config.motor_soft_limit / peak.clamp_min(1e-6))
        desired = desired.clone()
        desired[..., (0, 1, 3)] *= scale
        current = previous.view(1, 4).expand(desired.shape[0], -1) if previous.ndim == 1 else previous
        values = []
        for step in range(desired.shape[1]):
            duration = self.dt if step == 0 else self.config.command_hold_steps * self.dt
            maximum = slew * duration
            current = current + (desired[:, step] - current).clamp(-maximum, maximum)
            values.append(current)
        return torch.stack(values, dim=1)

    def _expand_residual(self, residual_knots: Tensor) -> Tensor:
        expanded = residual_knots.repeat_interleave(self.config.command_hold_steps, dim=1)
        if expanded.shape[1] < self.control_horizon_steps:
            expanded = torch.cat((expanded, expanded[:, -1:].expand(-1, self.control_horizon_steps - expanded.shape[1], -1)), dim=1)
        expanded = expanded[:, :self.control_horizon_steps]
        padding = torch.zeros(expanded.shape[0], self.model_horizon_steps - self.control_horizon_steps, 4, device=expanded.device, dtype=expanded.dtype)
        return torch.cat((expanded, padding), dim=1)

    def _plans(self, baseline: Tensor, residual_knots: Tensor, previous: Tensor) -> Tensor:
        delta = self._expand_residual(residual_knots)
        limit = torch.as_tensor(self.config.residual_limit, device=self.device).view(1, 1, 4)
        return self._project(baseline.unsqueeze(0) + delta.clamp(-limit, limit), previous)

    def _predict(self, observation: Tensor, action: Tensor, plans: Tensor, roll_command: Tensor) -> tuple[Tensor, Tensor, Tensor, Mapping[str, Tensor]]:
        if roll_command.ndim == 1:
            roll_command = roll_command.unsqueeze(0).expand(plans.shape[0], -1)
        motor, raw = standard_bluerov2_manual_mixer(plans, roll_command)
        normalized = (motor - self._action_mean) / self._action_std
        batch = len(plans)
        histories = observation.unsqueeze(0).expand(batch, -1, -1)
        actions = action.unsqueeze(0).expand(batch, -1, -1)
        with torch.inference_mode():
            if hasattr(self.model, "predict_residual"):
                output = self.model.predict_residual(histories, actions, normalized)
            else:
                output = self.model(histories, actions, normalized)
        if not isinstance(output, Mapping):
            physical = output
            output = {"predicted_future_physical_observation": physical}
        physical = output.get("predicted_future_physical_observation")
        if physical is None:
            predicted = output["predicted_future_observation"]
            physical = predicted * self._obs_std + self._obs_mean
        if physical.shape != (batch, self.model_horizon_steps, len(OBSERVATION_LABELS)):
            raise ValueError("Stage-2 prediction shape differs from MPC contract")
        return physical, motor, raw, output

    def _cost(self, physical: Tensor, plans: Tensor, raw: Tensor, targets: Mapping[str, Tensor], current: Tensor, previous: Tensor, baseline: Tensor, current_position: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        rotation = rotation6d_to_matrix(physical[..., ROTATION_6D_SLICE])
        body_velocity = physical[..., VIO_VELOCITY_SLICE]
        ned_velocity = (rotation @ body_velocity.unsqueeze(-1)).squeeze(-1)
        position = current_position.view(1, 1, 3) + torch.cumsum(ned_velocity * self.dt, dim=1)
        components: dict[str, Tensor] = {}
        if "body_velocity" in targets:
            mask = targets.get("body_velocity_mask", torch.ones_like(targets["body_velocity"]))
            scale = torch.as_tensor(self.config.body_velocity_tracking_scale_m_s, device=self.device)
            weight = torch.as_tensor(self.config.body_velocity_tracking_weight, device=self.device)
            err = (body_velocity - targets["body_velocity"]) / scale
            components["body_velocity_tracking"] = (err.square() * weight * mask).mean(dim=(1, 2))
        if "yaw_rate" in targets:
            mask = targets.get("yaw_rate_mask", torch.ones_like(targets["yaw_rate"]))
            err = (physical[..., YAW_RATE_INDEX:YAW_RATE_INDEX + 1] - targets["yaw_rate"]) / self.config.yaw_rate_tracking_scale_rad_s
            components["yaw_rate_tracking"] = self.config.yaw_rate_tracking_weight * (err.square() * mask).mean(dim=(1, 2))
        if "position" in targets:
            mask = targets.get("position_mask", torch.ones_like(targets["position"]))
            scale = torch.as_tensor(self.config.position_tracking_scale_m, device=self.device)
            weight = torch.as_tensor(self.config.position_tracking_weight, device=self.device)
            components["position_tracking"] = (((position - targets["position"]) / scale).square() * weight * mask).mean(dim=(1, 2))
        # If physical targets are provided, inactive axes are treated as
        # leakage.  Without targets no artificial axis preference is imposed.
        if "body_velocity" in targets:
            mask = targets.get("body_velocity_mask", torch.ones_like(targets["body_velocity"]))
            current_velocity = current[VIO_VELOCITY_SLICE].view(1, 1, 3)
            leakage = (body_velocity - current_velocity) * (1.0 - mask)
            components["cross_axis"] = self.config.cross_axis_weight * leakage.square().mean(dim=(1, 2))
        acceleration = physical[..., IMU_ACCEL_SLICE]
        prior_accel = current[IMU_ACCEL_SLICE].view(1, 1, 3).expand(len(plans), -1, -1)
        jerk = torch.diff(torch.cat((prior_accel, acceleration), dim=1), dim=1) / self.dt
        components["jerk"] = self.config.jerk_weight * torch.tanh(jerk.abs() / self.config.jerk_scale_m_s3).square().mean(dim=(1, 2))
        roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
        pitch = torch.asin(rotation[..., 2, 0].neg().clamp(-1.0, 1.0))
        tilt = torch.sqrt(roll.square() + pitch.square())
        components["tilt"] = self.config.tilt_weight * torch.tanh(tilt / self.config.tilt_scale_rad).square().mean(dim=1)
        residual = plans - baseline.unsqueeze(0)
        residual_scale = torch.as_tensor(self.config.residual_limit, device=self.device).view(1, 1, 4).clamp_min(1e-4)
        components["residual"] = self.config.residual_weight * (residual / residual_scale).square().mean(dim=(1, 2))
        prior = torch.cat((previous.view(1, 1, 4).expand(len(plans), 1, -1), plans[:, :-1]), dim=1)
        slew = torch.as_tensor(self.config.command_slew_per_second, device=self.device).view(1, 1, 4)
        components["command_rate"] = self.config.command_rate_weight * (((plans - prior) / slew / self.dt).square()).mean(dim=(1, 2))
        if plans.shape[1] >= 2:
            prior2 = torch.cat((previous.view(1, 1, 4).expand(len(plans), 1, -1), previous.view(1, 1, 4).expand(len(plans), 1, -1), plans[:, :-2]), dim=1)
            components["command_curvature"] = self.config.command_curvature_weight * (((plans - 2 * prior + prior2) / slew).square()).mean(dim=(1, 2))
        components["effort"] = self.config.effort_weight * plans.square().mean(dim=(1, 2))
        components["motor_saturation"] = self.config.motor_saturation_weight * functional.relu(raw.abs() - self.config.motor_soft_limit).square().mean(dim=(1, 2))
        if self.config.workspace_min_ned_m is not None:
            low = torch.as_tensor(self.config.workspace_min_ned_m, device=self.device)
            high = torch.as_tensor(self.config.workspace_max_ned_m, device=self.device)
            violation = functional.relu(low - position) + functional.relu(position - high)
            components["workspace"] = 1e5 * violation.square().mean(dim=(1, 2))
        total = sum(components.values())
        finite = torch.isfinite(total) & torch.isfinite(physical).all(dim=(1, 2))
        total = torch.where(finite, total, torch.full_like(total, float("inf")))
        return total, components

    def _candidate_residuals(self, center: Tensor, step: Tensor) -> Tensor:
        candidates = [torch.zeros_like(center), center.clone()]
        for knot in range(self.knot_count):
            for axis in range(4):
                offset = torch.zeros_like(center)
                offset[knot, axis] = step[axis]
                candidates.extend((center + offset, center - offset))
        return torch.stack(candidates, dim=0)

    def optimize(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        reference: MpcReference,
        previous_command: np.ndarray,
        *,
        current_position_ned_m: np.ndarray | None = None,
        roll_command: float | np.ndarray = 0.0,
    ) -> ContextAdaptiveResidualMpcResult:
        started = time.perf_counter()
        previous_values = np.asarray(previous_command, dtype=np.float32)
        if previous_values.shape != (4,) or not np.isfinite(previous_values).all():
            raise ValueError("previous command must contain four finite axes")
        previous = torch.as_tensor(previous_values, dtype=torch.float32, device=self.device)
        observation, action, current = self._prepare_history(observation_history, action_history)
        baseline_reference, targets = self._prepare_reference(reference)
        baseline = self._project(baseline_reference.unsqueeze(0), previous)[0]
        if current_position_ned_m is None:
            current_position = torch.zeros(3, dtype=torch.float32, device=self.device)
        else:
            values = np.asarray(current_position_ned_m, dtype=np.float32)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError("current position must contain three finite values")
            current_position = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        roll_values = np.asarray(roll_command, dtype=np.float32)
        if roll_values.ndim == 0:
            roll = torch.full((self.model_horizon_steps,), float(roll_values), device=self.device)
        elif roll_values.shape == (self.model_horizon_steps,):
            roll = torch.as_tensor(roll_values, device=self.device)
        else:
            raise ValueError("roll command must be scalar or model-horizon vector")
        center = self._warm_residual * self.config.warm_start_residual_weight
        step = torch.as_tensor(self.config.finite_difference_step, device=self.device)
        limit = torch.as_tensor(self.config.residual_limit, device=self.device)
        center = center.clamp(-limit, limit)
        best_plan = baseline.clone()
        best_cost = torch.tensor(float("inf"), device=self.device)
        best_components: dict[str, Tensor] = {}
        best_output: Mapping[str, Tensor] = {}
        baseline_cost = None
        baseline_components: dict[str, Tensor] = {}
        candidate_count = 0
        with torch.inference_mode():
            for iteration in range(self.config.iterations):
                residual_candidates = self._candidate_residuals(center, step).clamp(-limit, limit)
                plans = self._plans(baseline, residual_candidates, previous)
                physical, motor, raw, outputs = self._predict(observation, action, plans, roll)
                costs, components = self._cost(physical, plans, raw, targets, current, previous, baseline, current_position)
                candidate_count += len(plans)
                if iteration == 0:
                    baseline_cost = costs[0].clone()
                    baseline_components = {name: value[0].clone() for name, value in components.items()}
                index = int(torch.argmin(costs).item())
                if costs[index] < best_cost:
                    best_cost = costs[index].clone()
                    best_plan = plans[index].clone()
                    best_components = {name: value[index].clone() for name, value in components.items()}
                    best_output = {name: value[index:index + 1].clone() for name, value in outputs.items()}
                    center = residual_candidates[index].clone()
                step = step * 0.5
        if baseline_cost is None:
            raise RuntimeError("MPC evaluated no candidates")
        elapsed = time.perf_counter() - started
        base_value = float(baseline_cost.cpu())
        best_value = float(best_cost.cpu())
        improvement = base_value - best_value
        required = max(self.config.minimum_predicted_improvement, self.config.minimum_relative_improvement * max(abs(base_value), 1e-9))
        command_changed = bool(
            (best_plan[0] - baseline[0]).abs().amax().item()
            >= self.config.minimum_execution_change
        )
        accepted = (
            math.isfinite(best_value)
            and best_value < base_value
            and improvement >= required
            and command_changed
        )
        reason = "accepted" if accepted else "no predicted improvement"
        if not command_changed and improvement >= required:
            reason = "deferred-only plan improvement"
        if self.config.deadline_seconds > 0.0 and elapsed > self.config.deadline_seconds:
            accepted = False
            reason = "deadline exceeded"
        selected = best_plan if accepted else baseline
        residual = selected[: self.control_horizon_steps] - baseline[: self.control_horizon_steps]
        self._warm_residual.zero_()
        if accepted:
            for knot in range(self.knot_count):
                start = knot * self.config.command_hold_steps
                stop = min(self.control_horizon_steps, start + self.config.command_hold_steps)
                self._warm_residual[knot] = residual[start:stop].mean(dim=0)
        diagnostics: dict[str, Any] = {
            "implementation_id": IMPLEMENTATION_ID,
            "stage2_implementation": STAGE2_IMPLEMENTATION,
            "baseline_cost": base_value,
            "selected_cost": best_value if accepted else base_value,
            "best_candidate_cost": best_value,
            "predicted_improvement": improvement,
            "required_improvement": required,
            "accepted": accepted,
            "immediate_command_change": float((best_plan[0] - baseline[0]).abs().amax().cpu()),
            "candidate_count": candidate_count,
            "control_horizon_steps": self.control_horizon_steps,
            "knot_count": self.knot_count,
            "timing_seconds": elapsed,
            "baseline_components": {name: float(value.cpu()) for name, value in baseline_components.items()},
            "selected_components": {name: float(value.cpu()) for name, value in (best_components if accepted else baseline_components).items()},
        }
        if best_output:
            for name in ("domain_context", "residual_rate_body", "physical_action_delta"):
                if name in best_output:
                    value = best_output[name]
                    if name == "domain_context":
                        diagnostics[name] = value[0].detach().cpu().numpy().tolist()
                    else:
                        diagnostics[name] = value[0, : self.control_horizon_steps].detach().cpu().numpy().tolist()
        return ContextAdaptiveResidualMpcResult(
            command=selected[0].detach().cpu().numpy(),
            planned_command=selected.detach().cpu().numpy(),
            baseline_command=baseline[0].detach().cpu().numpy(),
            residual=residual.detach().cpu().numpy(),
            accepted=accepted,
            reason=reason,
            diagnostics=diagnostics,
        )


__all__ = [
    "COMMAND_ORDER",
    "ContextAdaptiveResidualMpc",
    "ContextAdaptiveResidualMpcConfig",
    "ContextAdaptiveResidualMpcResult",
    "IMPLEMENTATION_ID",
    "MpcReference",
    "load_context_adaptive_residual_mpc_config",
]
