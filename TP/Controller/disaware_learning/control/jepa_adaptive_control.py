"""Factual-dynamics JEPA compensation for deterministic 4-DoF control.

The learned model is queried only at the action currently being executed.  It
estimates the near-term body acceleration under that factual continuation; it
does not rank counterfactual candidates.  A coarse first-order plant provides
the known control direction and the JEPA prediction supplies a bounded,
low-pass residual acceleration estimate.
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
except ImportError as error:  # pragma: no cover - optional learning dependency.
    raise ImportError("JEPA adaptive control requires PyTorch") from error

from disaware_learning.learning.motor_jepa_data import (
    DEPTH_INDEX,
    FEATURE_CONTRACT_ID,
    IMU_GYRO_SLICE,
    OBSERVATION_LABELS,
    Standardizer,
    VIO_VELOCITY_SLICE,
    motor_output_to_modal,
)
from disaware_learning.learning.stage2 import (
    IMPLEMENTATION as STAGE2_IMPLEMENTATION,
    load_stage2_checkpoint,
)


IMPLEMENTATION_ID = "jepa_factual_dynamics_adaptive_control_v1"
CONTROL_AXES = ("surge", "sway", "heave_up", "yaw")


def _vector(values: object, name: str, *, positive: bool = False) -> np.ndarray:
    try:
        result = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain four finite values") from error
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain four finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must contain positive values")
    return result


@dataclass(frozen=True)
class JepaAdaptiveConfig:
    """Conservative nominal model and bounded compensation settings.

    ``input_acceleration_gain_per_command`` maps normalized MANUAL intent to
    ``[vx_dot, vy_dot, vz_down_dot, yaw_rate_dot]``.  Heave is negative because
    positive MANUAL heave commands upward while body FRD velocity is positive
    downward.
    """

    # These values are fitted to the already validated fixed PID.  For a
    # slow reference, the resulting nominal inverse is approximately
    # ``feedforward * reference + kp * (reference - velocity)``.  Keeping the
    # adaptive nominal branch on that scale is important: the JEPA residual
    # must compensate the baseline, not replace it with a saturated command.
    input_acceleration_gain_per_command: tuple[float, ...] = (
        0.0277778,
        0.0299401,
        -0.0225989,
        5.78,
    )
    velocity_damping_rate_per_second: tuple[float, ...] = (
        0.0675,
        0.086527,
        0.0698305,
        2.081,
    )
    tracking_bandwidth_per_second: tuple[float, ...] = (0.08, 0.10, 0.08, 3.526)
    # The collection reference is a velocity/rate command.  Its derivative is
    # only a small anticipatory term; a large derivative feedforward caused
    # the first adaptive rollout to hit the MANUAL limits on every slow wave.
    reference_acceleration_limit: tuple[float, ...] = (0.005, 0.006, 0.004, 0.03)

    model_compensation_gain: tuple[float, ...] = (0.10, 0.10, 0.05, 0.10)
    model_residual_limit: tuple[float, ...] = (0.05, 0.06, 0.03, 0.30)
    # Keep the code default identical to the shipped deployment config.  The
    # rollout launch intentionally permits an empty config path, so a broad
    # dataclass default here would silently bypass the safety envelope.
    model_compensation_command_limit: tuple[float, ...] = (0.12, 0.12, 0.08, 0.06)
    # Factual inference is refreshed at a lower rate than MANUAL.  Keep its
    # correction as a low-bandwidth channel so prediction noise cannot become
    # a second high-bandwidth feedback loop.
    model_compensation_slew_per_second: tuple[float, ...] = (0.20, 0.20, 0.16, 0.12)
    model_filter_time_constant_seconds: tuple[float, ...] = (0.80, 0.80, 1.00, 0.25)
    model_warmup_seconds: float = 1.0

    adaptive_growth: tuple[float, ...] = (0.03, 0.03, 0.02, 0.20)
    adaptive_decay_per_second: tuple[float, ...] = (0.50, 0.50, 0.50, 1.00)
    adaptive_acceleration_limit: tuple[float, ...] = (0.010, 0.010, 0.008, 0.080)
    adaptive_boundary: tuple[float, ...] = (0.03, 0.03, 0.03, 0.05)

    command_limit: tuple[float, ...] = (0.50, 0.50, 0.50, 0.50)
    command_slew_per_second: tuple[float, ...] = (0.80, 0.80, 0.70, 0.60)
    motor_soft_limit: float = 0.90
    deadline_seconds: float = 0.040

    def __post_init__(self) -> None:
        positive_vectors = (
            "velocity_damping_rate_per_second",
            "tracking_bandwidth_per_second",
            "reference_acceleration_limit",
            "model_residual_limit",
            "model_compensation_command_limit",
            "model_compensation_slew_per_second",
            "model_filter_time_constant_seconds",
            "adaptive_decay_per_second",
            "adaptive_boundary",
            "command_limit",
            "command_slew_per_second",
        )
        nonnegative_vectors = (
            "model_compensation_gain",
            "adaptive_growth",
            "adaptive_acceleration_limit",
        )
        for name in positive_vectors:
            values = _vector(getattr(self, name), name, positive=True)
            object.__setattr__(self, name, tuple(float(value) for value in values))
        for name in nonnegative_vectors:
            values = _vector(getattr(self, name), name)
            if np.any(values < 0.0):
                raise ValueError(f"{name} cannot contain negative values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        input_gain = _vector(
            self.input_acceleration_gain_per_command,
            "input_acceleration_gain_per_command",
        )
        if np.any(np.abs(input_gain) < 1e-6):
            raise ValueError("input acceleration gains must be nonzero")
        object.__setattr__(
            self,
            "input_acceleration_gain_per_command",
            tuple(float(value) for value in input_gain),
        )
        if any(value > 1.0 for value in self.command_limit):
            raise ValueError("command limits cannot exceed one")
        for name in ("model_warmup_seconds", "deadline_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 < self.motor_soft_limit <= 1.0:
            raise ValueError("motor_soft_limit must be in (0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {"implementation_id": IMPLEMENTATION_ID, **asdict(self)}


def load_jepa_adaptive_config(path: str | Path) -> JepaAdaptiveConfig:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JEPA adaptive config must contain a JSON object")
    payload = dict(payload)
    implementation = payload.pop("implementation_id", IMPLEMENTATION_ID)
    if implementation != IMPLEMENTATION_ID:
        raise ValueError("unexpected JEPA adaptive implementation_id")
    expected = set(JepaAdaptiveConfig.__dataclass_fields__)
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"unknown JEPA adaptive config keys: {sorted(unknown)}")
    return JepaAdaptiveConfig(**payload)


@dataclass(frozen=True)
class JepaAdaptiveResult:
    command: np.ndarray
    nominal_command: np.ndarray
    model_compensation_command: np.ndarray
    used_model: bool
    reason: str
    diagnostics: dict[str, Any]


class JepaFactualDynamicsAdaptiveController:
    """DM-AC-inspired controller using a JEPA factual acceleration estimate."""

    def __init__(
        self,
        model: nn.Module,
        standardizer: Standardizer,
        *,
        history_steps: int,
        sample_rate_hz: float,
        config: JepaAdaptiveConfig = JepaAdaptiveConfig(),
        device: str | torch.device = "cpu",
    ) -> None:
        if history_steps < 2 or sample_rate_hz <= 0.0:
            raise ValueError("invalid JEPA history/sample-rate contract")
        self.device = torch.device(device)
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.standardizer = standardizer
        self.history_steps = int(history_steps)
        self.sample_rate_hz = float(sample_rate_hz)
        backbone = getattr(model, "backbone", None)
        backbone_config = getattr(backbone, "config", None)
        self.model_horizon_steps = int(getattr(backbone_config, "horizon_steps", 1))
        if self.model_horizon_steps < 1:
            raise ValueError("Stage 2 model horizon must be positive")
        self.config = config
        self.action_dim = int(np.asarray(standardizer.action_mean).shape[0])
        if self.action_dim not in (5, 6):
            raise ValueError("Stage 2 action contract must contain five modal or six motor channels")
        self._observation_mean = torch.as_tensor(
            standardizer.observation_mean, dtype=torch.float32, device=self.device
        )
        self._observation_std = torch.as_tensor(
            standardizer.observation_std, dtype=torch.float32, device=self.device
        )
        self._action_mean = torch.as_tensor(
            standardizer.action_mean, dtype=torch.float32, device=self.device
        )
        self._action_std = torch.as_tensor(
            standardizer.action_std, dtype=torch.float32, device=self.device
        )
        self._filtered_residual: np.ndarray | None = None
        self._last_raw_residual = np.zeros(4, dtype=np.float64)
        self._last_predicted_acceleration = np.zeros(4, dtype=np.float64)
        self._last_model_residual: np.ndarray | None = None
        self._last_compensation_command: np.ndarray | None = None
        self._last_projected_roll_command = 0.0
        self._adaptive_gain = np.zeros(4, dtype=np.float64)
        self._elapsed_seconds = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: JepaAdaptiveConfig = JepaAdaptiveConfig(),
        device: str | torch.device = "cpu",
    ) -> "JepaFactualDynamicsAdaptiveController":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("feature_contract_id") != FEATURE_CONTRACT_ID:
            raise ValueError("checkpoint does not use the retained feature contract")
        if checkpoint.get("implementation") != STAGE2_IMPLEMENTATION:
            raise ValueError("checkpoint is not the retained physics-integrated Stage 2")
        readiness = checkpoint.get("training_readiness")
        if not isinstance(readiness, Mapping) or readiness.get("passed") is not True:
            raise ValueError("checkpoint failed Stage-2 training readiness")
        model, checkpoint = load_stage2_checkpoint(checkpoint_path, device=device)
        return cls(
            model,
            Standardizer.from_dict(checkpoint["standardizer"]),
            history_steps=int(checkpoint["history_steps"]),
            sample_rate_hz=float(checkpoint["sample_rate_hz"]),
            config=config,
            device=device,
        )

    def reset(self) -> None:
        self._filtered_residual = None
        self._last_raw_residual.fill(0.0)
        self._last_predicted_acceleration.fill(0.0)
        self._last_model_residual = None
        self._last_compensation_command = None
        self._last_projected_roll_command = 0.0
        self._adaptive_gain.fill(0.0)
        self._elapsed_seconds = 0.0

    def _prepare_inputs(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
    ) -> tuple[Tensor, Tensor, Tensor, np.ndarray]:
        observation = np.asarray(observation_history, dtype=np.float32).copy()
        action = np.asarray(action_history, dtype=np.float32)
        if observation.shape != (self.history_steps, len(OBSERVATION_LABELS)):
            raise ValueError("observation history shape mismatch")
        if action.shape == (self.history_steps, 6) and self.action_dim == 5:
            action = motor_output_to_modal(action)
        if action.shape != (self.history_steps, self.action_dim):
            raise ValueError("action history shape mismatch")
        if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(action)):
            raise ValueError("controller history contains non-finite values")
        current = observation[-1].astype(np.float64, copy=True)
        observation[:, DEPTH_INDEX] -= observation[-1, DEPTH_INDEX]
        standardized_observation = (
            torch.as_tensor(observation, device=self.device) - self._observation_mean
        ) / self._observation_std
        standardized_action = (
            torch.as_tensor(action, device=self.device) - self._action_mean
        ) / self._action_std
        # DM-AC uses the latest available executed action because the command
        # being computed at this tick has not reached the plant yet.
        held_future_action = standardized_action[-1:].expand(
            self.model_horizon_steps, -1
        ).clone()
        return (
            standardized_observation,
            standardized_action,
            held_future_action,
            current,
        )

    def _predict_factual_acceleration(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        observation, action, future_action, current = self._prepare_inputs(
            observation_history, action_history
        )
        with torch.inference_mode():
            output = self.model(
                observation.unsqueeze(0),
                action.unsqueeze(0),
                future_action.unsqueeze(0),
            )
        linear = output.get("linear_velocity_rate_body_m_s2")
        angular = output.get("angular_velocity_rate_body_rad_s2")
        if (
            linear is None
            or angular is None
            or linear.ndim != 3
            or angular.ndim != 3
            or linear.shape[0] != 1
            or angular.shape[0] != 1
            or linear.shape[1] < 1
            or angular.shape[1] < 1
        ):
            raise ValueError("Stage 2 did not return a future body acceleration rollout")
        acceleration = np.asarray(
            [
                float(linear[0, 0, 0].cpu()),
                float(linear[0, 0, 1].cpu()),
                float(linear[0, 0, 2].cpu()),
                float(angular[0, 0, 2].cpu()),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(acceleration)):
            raise ValueError("JEPA factual acceleration is non-finite")
        # Stage 2 exposes the decomposition used during training.  Prefer it
        # over reconstructing a second nominal plant in the controller: the
        # decomposition already includes actuator lag, allocation and the
        # configured nominal hydrodynamics.
        residual_rate = output.get("residual_rate_body")
        model_residual: np.ndarray | None = None
        if residual_rate is not None:
            if (
                residual_rate.ndim != 3
                or residual_rate.shape[0] != 1
                or residual_rate.shape[1] < 1
                or residual_rate.shape[2] != 6
            ):
                raise ValueError("Stage 2 returned a malformed residual rate rollout")
            model_residual = np.asarray(
                [
                    float(residual_rate[0, 0, 0].cpu()),
                    float(residual_rate[0, 0, 1].cpu()),
                    float(residual_rate[0, 0, 2].cpu()),
                    float(residual_rate[0, 0, 5].cpu()),
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(model_residual)):
                raise ValueError("Stage 2 residual rate is non-finite")
        return acceleration, current, model_residual

    def _project_command(
        self,
        desired: np.ndarray,
        previous: np.ndarray,
        dt: float,
        *,
        roll_command: float = 0.0,
    ) -> np.ndarray:
        limit = _vector(self.config.command_limit, "command_limit", positive=True)
        slew = _vector(
            self.config.command_slew_per_second,
            "command_slew_per_second",
            positive=True,
        )
        desired = np.clip(desired, -limit, limit)
        command = previous + np.clip(desired - previous, -slew * dt, slew * dt)
        if not math.isfinite(float(roll_command)):
            raise ValueError("roll_command must be finite")
        projected_roll = float(
            np.clip(
                roll_command,
                -self.config.motor_soft_limit,
                self.config.motor_soft_limit,
            )
        )
        surge, sway, heave, yaw = command
        # Match the retained six-thruster mixer, including the two vertical
        # motors.  Checking only the horizontal group allowed heave+roll to
        # exceed the motor envelope without any projection or diagnostic.
        motors = np.asarray(
            (
                yaw - surge + sway,
                -yaw - surge - sway,
                -yaw + surge + sway,
                yaw + surge - sway,
                projected_roll - heave,
                -projected_roll - heave,
            ),
            dtype=np.float64,
        )
        peak = float(np.max(np.abs(motors)))
        if peak > self.config.motor_soft_limit:
            scale = self.config.motor_soft_limit / peak
            command *= scale
            projected_roll *= scale
        self._last_projected_roll_command = projected_roll
        return np.clip(command, -limit, limit)

    def step(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        reference_body_twist: np.ndarray,
        next_reference_body_twist: np.ndarray,
        previous_command: np.ndarray,
        dt: float,
        *,
        current_observation: np.ndarray | None = None,
        predicted_factual_acceleration: np.ndarray | None = None,
        auxiliary_roll_command: float = 0.0,
    ) -> JepaAdaptiveResult:
        started = time.perf_counter()
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("control timestep must be positive and finite")
        reference = np.asarray(reference_body_twist, dtype=np.float64)
        next_reference = np.asarray(next_reference_body_twist, dtype=np.float64)
        previous = np.asarray(previous_command, dtype=np.float64)
        for values, name in (
            (reference, "reference body twist"),
            (next_reference, "next reference body twist"),
            (previous, "previous command"),
        ):
            if values.shape != (4,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain four finite values")

        model_updated = predicted_factual_acceleration is None
        model_residual: np.ndarray | None = None
        if model_updated:
            predicted_acceleration, model_observation, model_residual = self._predict_factual_acceleration(
                observation_history, action_history
            )
        else:
            predicted_acceleration = np.asarray(
                predicted_factual_acceleration, dtype=np.float64
            )
            if predicted_acceleration.shape != (4,) or not np.all(
                np.isfinite(predicted_acceleration)
            ):
                raise ValueError("predicted factual acceleration must contain four finite values")
            model_observation = np.asarray(observation_history[-1], dtype=np.float64)
        if current_observation is None:
            current_observation = model_observation
        else:
            current_observation = np.asarray(current_observation, dtype=np.float64)
            if current_observation.shape != (len(OBSERVATION_LABELS),) or not np.all(
                np.isfinite(current_observation)
            ):
                raise ValueError("current observation has the wrong shape or non-finite values")
        current_twist = np.concatenate(
            (
                current_observation[VIO_VELOCITY_SLICE],
                [current_observation[IMU_GYRO_SLICE][2]],
            )
        )
        input_gain = _vector(
            self.config.input_acceleration_gain_per_command,
            "input_acceleration_gain_per_command",
        )
        damping = _vector(
            self.config.velocity_damping_rate_per_second,
            "velocity_damping_rate_per_second",
            positive=True,
        )
        bandwidth = _vector(
            self.config.tracking_bandwidth_per_second,
            "tracking_bandwidth_per_second",
            positive=True,
        )
        reference_rate = np.clip(
            (next_reference - reference) / dt,
            -_vector(
                self.config.reference_acceleration_limit,
                "reference_acceleration_limit",
                positive=True,
            ),
            _vector(
                self.config.reference_acceleration_limit,
                "reference_acceleration_limit",
                positive=True,
            ),
        )
        tracking_error = current_twist - reference

        growth = _vector(self.config.adaptive_growth, "adaptive_growth")
        decay = _vector(
            self.config.adaptive_decay_per_second,
            "adaptive_decay_per_second",
            positive=True,
        )
        adaptive_limit = _vector(
            self.config.adaptive_acceleration_limit,
            "adaptive_acceleration_limit",
        )
        self._adaptive_gain = np.clip(
            self._adaptive_gain + dt * (growth * np.abs(tracking_error) - decay * self._adaptive_gain),
            0.0,
            adaptive_limit,
        )
        adaptive_acceleration = self._adaptive_gain * np.tanh(
            tracking_error
            / _vector(self.config.adaptive_boundary, "adaptive_boundary", positive=True)
        )
        desired_acceleration = (
            reference_rate - bandwidth * tracking_error - adaptive_acceleration
        )

        nominal_current_acceleration = input_gain * previous - damping * current_twist
        if model_updated:
            # Use the residual from the same physical decomposition as the
            # Stage 2 training objective.  The fallback is kept for the small
            # test doubles and for explicitly legacy model stubs only.
            raw_residual = (
                model_residual
                if model_residual is not None
                else predicted_acceleration - nominal_current_acceleration
            )
            self._last_raw_residual = raw_residual.copy()
            self._last_predicted_acceleration = predicted_acceleration.copy()
            self._last_model_residual = (
                None if model_residual is None else model_residual.copy()
            )
        else:
            raw_residual = self._last_raw_residual.copy()
            predicted_acceleration = self._last_predicted_acceleration.copy()
            model_residual = self._last_model_residual
        residual_limit = _vector(
            self.config.model_residual_limit,
            "model_residual_limit",
            positive=True,
        )
        raw_residual = np.clip(raw_residual, -residual_limit, residual_limit)
        if self._filtered_residual is None:
            self._filtered_residual = raw_residual.copy()
        elif model_updated:
            time_constant = _vector(
                self.config.model_filter_time_constant_seconds,
                "model_filter_time_constant_seconds",
                positive=True,
            )
            # The prediction is refreshed at the model rate, not at the
            # command-loop rate.  Using the 25 Hz dt here would make a 0.25 s
            # yaw filter behave like roughly a 1.2 s filter in real time.
            model_dt = 1.0 / self.sample_rate_hz
            alpha = 1.0 - np.exp(-model_dt / time_constant)
            self._filtered_residual += alpha * (raw_residual - self._filtered_residual)

        self._elapsed_seconds += dt
        warmup_fraction = (
            1.0
            if self.config.model_warmup_seconds == 0.0
            else min(1.0, self._elapsed_seconds / self.config.model_warmup_seconds)
        )
        configured_compensation = _vector(
            self.config.model_compensation_gain,
            "model_compensation_gain",
        )
        effective_compensation = warmup_fraction * configured_compensation
        nominal_command = (
            desired_acceleration + damping * current_twist
        ) / input_gain
        target_compensation_command = -effective_compensation * self._filtered_residual / input_gain
        target_compensation_command = np.clip(
            target_compensation_command,
            -_vector(
                self.config.model_compensation_command_limit,
                "model_compensation_command_limit",
                positive=True,
            ),
            _vector(
                self.config.model_compensation_command_limit,
                "model_compensation_command_limit",
                positive=True,
            ),
        )
        compensation_slew = _vector(
            self.config.model_compensation_slew_per_second,
            "model_compensation_slew_per_second",
            positive=True,
        )
        if self._last_compensation_command is None:
            # The normal command projection still limits the first executed
            # step; avoid adding an artificial startup lag to the model term.
            compensation_command = target_compensation_command.copy()
        else:
            compensation_command = self._last_compensation_command + np.clip(
                target_compensation_command - self._last_compensation_command,
                -compensation_slew * dt,
                compensation_slew * dt,
            )
        self._last_compensation_command = compensation_command.copy()
        requested = nominal_command + compensation_command
        selected = self._project_command(
            requested,
            previous,
            dt,
            roll_command=float(auxiliary_roll_command),
        )

        elapsed = time.perf_counter() - started
        used_model = True
        reason = "model compensation applied"
        if self.config.deadline_seconds > 0.0 and elapsed > self.config.deadline_seconds:
            selected = self._project_command(
                nominal_command,
                previous,
                dt,
                roll_command=float(auxiliary_roll_command),
            )
            compensation_command = np.zeros(4, dtype=np.float64)
            used_model = False
            reason = "deadline exceeded; nominal fallback"

        diagnostics: dict[str, Any] = {
            "implementation_id": IMPLEMENTATION_ID,
            "current_body_twist": current_twist.copy(),
            "reference_body_twist": reference.copy(),
            "reference_acceleration": reference_rate.copy(),
            "tracking_error": tracking_error.copy(),
            "desired_acceleration": desired_acceleration.copy(),
            "predicted_factual_acceleration": predicted_acceleration.copy(),
            "nominal_current_acceleration": nominal_current_acceleration.copy(),
            "raw_model_residual_acceleration": raw_residual.copy(),
            "filtered_model_residual_acceleration": self._filtered_residual.copy(),
            "model_residual_source": (
                "stage2_residual_rate" if model_residual is not None else "nominal_fallback"
            ),
            "model_filter_update_dt_seconds": (
                1.0 / self.sample_rate_hz if model_updated else 0.0
            ),
            "effective_model_compensation_gain": effective_compensation.copy(),
            "target_model_compensation_command": target_compensation_command.copy(),
            "model_compensation_slew_per_second": compensation_slew.copy(),
            "auxiliary_roll_command": float(auxiliary_roll_command),
            "projected_auxiliary_roll_command": float(
                self._last_projected_roll_command
            ),
            "adaptive_gain": self._adaptive_gain.copy(),
            "adaptive_acceleration": adaptive_acceleration.copy(),
            "requested_command_before_projection": requested.copy(),
            "timing_seconds": elapsed,
            "model_prediction_updated": model_updated,
        }
        return JepaAdaptiveResult(
            command=selected.astype(np.float32),
            nominal_command=nominal_command.astype(np.float32),
            model_compensation_command=compensation_command.astype(np.float32),
            used_model=used_model,
            reason=reason,
            diagnostics=diagnostics,
        )

    def warmup(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
    ) -> None:
        self._predict_factual_acceleration(observation_history, action_history)
