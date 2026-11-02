"""Reference-conditioned verified-action MPPI using the retained JEPA model.

The controller samples complete four-axis MANUAL command trajectories, maps
them through the standard six-thruster mixer, and asks Stage 2 to predict the
resulting physical trajectory. It executes only a trajectory already evaluated
by Stage 2; softmin population weights update the next receding-horizon nominal.
A reference is a source-agnostic collection of optional pose/twist targets;
joystick and planner adapters construct the same contract before entering this
module.
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
except ImportError as error:  # pragma: no cover - optional controller dependency.
    raise ImportError("JEPA-MPPI requires PyTorch") from error

from disaware_learning.learning.motor_jepa_data import (
    ACTUATOR_MODEL_ID,
    ACTION_CONTRACT_ID,
    ACTION_LABELS,
    DEPTH_INDEX,
    FEATURE_CONTRACT_ID,
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
IMPLEMENTATION_ID = (
    "context_hybrid_absolute_action_jepa_mppi_v54_projected_weighted_realtime"
)
REFERENCE_CONTRACT_ID = "optional_ned_pose_body_or_ned_twist_v1"
COMMAND_ORDER = ("surge", "sway", "heave", "yaw")


def _finite_tuple(values: object, length: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} finite values") from error
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Optional physical targets sampled or time-stamped over one horizon.

    Each vector field accepts one constant vector or ``[time, channel]`` data.
    Scalar yaw fields accept a scalar or ``[time]`` data.  When
    ``time_seconds`` is present, the controller interpolates all nonconstant
    fields to the JEPA sample times.  Masks have matching shapes and allow a
    planner to constrain only selected axes without defining source modes.
    """

    time_seconds: np.ndarray | None = None
    position_ned_m: np.ndarray | None = None
    position_mask: np.ndarray | None = None
    linear_velocity_body_m_s: np.ndarray | None = None
    body_velocity_mask: np.ndarray | None = None
    linear_velocity_ned_m_s: np.ndarray | None = None
    ned_velocity_mask: np.ndarray | None = None
    yaw_rad: np.ndarray | float | None = None
    yaw_mask: np.ndarray | None = None
    yaw_rate_rad_s: np.ndarray | float | None = None
    yaw_rate_mask: np.ndarray | None = None

    @classmethod
    def body_twist(
        cls,
        values: np.ndarray,
        *,
        time_seconds: np.ndarray | None = None,
    ) -> "ReferenceTrajectory":
        """Construct a physical ``[vx, vy, vz, yaw_rate]`` reference."""
        twist = np.asarray(values, dtype=np.float32)
        if twist.shape == (4,):
            return cls(
                time_seconds=time_seconds,
                linear_velocity_body_m_s=twist[:3],
                yaw_rate_rad_s=float(twist[3]),
            )
        if twist.ndim != 2 or twist.shape[1] != 4:
            raise ValueError("body twist must be [4] or [time, 4]")
        return cls(
            time_seconds=time_seconds,
            linear_velocity_body_m_s=twist[:, :3],
            yaw_rate_rad_s=twist[:, 3],
        )

    @classmethod
    def planner(
        cls,
        *,
        time_seconds: np.ndarray,
        position_ned_m: np.ndarray | None = None,
        linear_velocity_ned_m_s: np.ndarray | None = None,
        yaw_rad: np.ndarray | float | None = None,
        yaw_rate_rad_s: np.ndarray | float | None = None,
        position_mask: np.ndarray | None = None,
    ) -> "ReferenceTrajectory":
        """Construct a conventional NED planner trajectory reference."""
        return cls(
            time_seconds=time_seconds,
            position_ned_m=position_ned_m,
            position_mask=position_mask,
            linear_velocity_ned_m_s=linear_velocity_ned_m_s,
            yaw_rad=yaw_rad,
            yaw_rate_rad_s=yaw_rate_rad_s,
        )


@dataclass(frozen=True)
class JepaMppiConfig:
    sample_count: int = 64
    iterations: int = 1
    # Number of model-rate action intervals represented by one optimization
    # knot. The knot plan is expanded back to one action per Stage-2 step
    # before rollout, so this lowers search dimension without changing the
    # learned action cadence.
    command_hold_steps: int = 1
    temperature: float = 0.50
    target_effective_sample_size: float = 16.0
    # Physical modal-action exploration scales derived from the train-set
    # innovation P95 after undoing action standardization. In particular, yaw
    # has a much smaller empirical action scale than translation.
    noise_std: tuple[float, ...] = (0.05, 0.05, 0.07, 0.007)
    noise_correlation: float = 0.85
    # Retain the calibrated local proposal while allowing a fixed share of the
    # same candidate population to query a wider, still support-projected
    # neighborhood. Wide samples are interleaved across all proposal centres.
    wide_noise_fraction: float = 0.0
    wide_noise_scale: float = 1.0
    # Smooth the low-dimensional action knots before Stage 2 evaluates them.
    # A three-point binomial filter is well-defined for the short underwater
    # plan, unlike the long-window Savitzky-Golay filter used by RexROV2 MPPI.
    # The previous executed command anchors the left boundary.
    plan_smoothing_strength: float = 0.0
    # A finite MPPI population cannot discover an entire high-dimensional
    # action trajectory from zero at the first control update. Use the known
    # reference twist to provide a second sampling centre; JEPA still scores
    # every plan and remains the only trajectory model.
    reference_seed_sample_fraction: float = 0.0
    # Inverse response seed for the calibrated Stage-2 modal-action path.
    # These are empirical steady-rate gains from the NMPC training split;
    # yaw is linear here because Stage 2 consumes motor-modal action, not raw
    # propeller RPM or a signed-square command.
    reference_seed_response_gain: tuple[float, ...] = (0.404, 0.427, 0.492, 3.66)
    # First-order inverse response used only to centre part of the MPPI
    # population. For x_dot=(gain*u-x)/tau, u=(x_ref+tau*x_ref_dot)/gain.
    # Stage 2 still evaluates every resulting candidate before execution.
    reference_seed_time_constant_seconds: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    # Add a second, state-feedback proposal around the inverse-response seed.
    # It is never executed directly: Stage 2 predicts and MPPI ranks it like
    # every other candidate. The decaying correction provides the candidate
    # population with early braking directions when the measured response has
    # already overshot a future reference.
    reference_seed_feedback_gain: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    # Outer-loop proposal gain [1/s]. This only centres part of the MPPI
    # population; JEPA still predicts and ranks every resulting plan.
    reference_seed_position_feedback_gain: tuple[float, ...] = (0.0, 0.0, 0.0)
    reference_seed_feedback_decay_seconds: float = 1.0
    # Increase response feedback only when the supplied horizon ends in a
    # genuine full stop. This gives MPPI stronger braking proposals without
    # slowing ordinary nonzero trajectory segments.
    terminal_stop_feedback_gain: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    terminal_stop_body_velocity_threshold_m_s: float = 0.03
    terminal_stop_yaw_rate_threshold_rad_s: float = 0.03
    # Stage 2 was trained on the five-dimensional mixer modal action.  Keep
    # candidate actions inside that empirical support before asking the model
    # to rank them.  This is deliberately a soft trust-region width in
    # standard-deviation units, not a privileged-state constraint.
    modal_action_trust_region_std: tuple[float, ...] = (2.5, 2.5, 2.5, 2.5, 2.5)
    # Empirical train-set P95 envelopes in standardized modal coordinates.
    # Crossing these remains possible, but is continuously penalized before
    # the hard P99-like projection below is reached.
    modal_action_soft_support_std: tuple[float, ...] = (
        1.80,
        1.75,
        1.90,
        1.95,
        1.75,
    )
    modal_action_innovation_soft_support_std: tuple[float, ...] = (
        0.24,
        0.23,
        0.34,
        0.27,
        0.18,
    )
    modal_action_innovation_limit_std: tuple[float, ...] = (
        0.46,
        0.36,
        0.59,
        0.38,
        0.26,
    )
    model_action_support_weight: float = 2.0
    model_action_innovation_support_weight: float = 4.0
    # Only the early part of a plan is perturbed; the remainder carries the
    # shifted prior plan and remains available to the diagnostic rollout.
    optimization_horizon_seconds: float = 1.20
    # Limit the recurrent Stage-2 rollout used online without changing the
    # longer horizon stored in the trained checkpoint. Zero retains the full
    # model horizon. This must align with the piecewise-constant action knots.
    rollout_horizon_seconds: float = 0.0
    # Stage 2 can expose a longer diagnostic rollout than the interval that is
    # accurate enough for action ranking. Zero retains the full model horizon.
    cost_horizon_seconds: float = 0.0
    nominal_update_rate: float = 1.0
    # Keep the fixed nominal rollout available as an exact ablation. When
    # disabled, Stage 2 exposes the same fixed-physics prediction used by
    # nominal_mppi; when enabled, the context path is the only increment.
    include_learned_residual: bool = True
    # Stage 2 is trained with future_action[k] driving x[k] -> x[k + 1].
    # Therefore the first rollout action must be candidate[0], not the action
    # measured at the end of the history window.
    common_action_prefix_steps: int = 0
    # Number of Stage-2 model steps for which the first proposed command is
    # executed. The collector holds between model updates on its faster
    # command loop; this must not be interpreted as another model-rate hold.
    execution_hold_steps: int = 1
    # Approximate asynchronous solve/application latency inside the first
    # Stage-2 action interval. The model receives the interval mean of the
    # measured current action and the candidate that can only start later.
    planner_latency_compensation_seconds: float = 0.0

    manual_limit: tuple[float, ...] = (0.65, 0.65, 0.45, 0.50)
    command_slew_per_second: tuple[float, ...] = (1.2, 1.2, 1.0, 1.2)
    motor_soft_limit: float = 0.90

    position_tracking_weight: tuple[float, ...] = (6.0, 6.0, 9.0)
    position_tracking_scale_m: tuple[float, ...] = (0.50, 0.50, 0.30)
    terminal_position_tracking_weight: tuple[float, ...] = (0.0, 0.0, 0.0)
    terminal_position_stop_only: bool = True
    body_velocity_tracking_weight: tuple[float, ...] = (12.0, 12.0, 14.0)
    ned_velocity_tracking_weight: tuple[float, ...] = (12.0, 12.0, 14.0)
    velocity_tracking_scale_m_s: tuple[float, ...] = (0.15, 0.15, 0.12)
    yaw_tracking_weight: float = 8.0
    yaw_tracking_scale_rad: float = 0.35
    yaw_rate_tracking_weight: float = 10.0
    yaw_rate_tracking_scale_rad_s: float = 0.30
    command_rate_weight: float = 0.35
    # Per-axis multiplier prevents the yaw regularization from being diluted
    # by the three translational axes in the aggregate command cost.
    command_smoothing_axis_weight: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    motor_saturation_weight: float = 12.0
    maximum_predicted_tilt_rad: float = math.radians(30.0)
    tilt_barrier_weight: float = 200.0

    minimum_predicted_improvement: float = 0.0
    minimum_relative_improvement: float = 0.0
    verify_weighted_plan: bool = True
    execution_update_fraction: float = 1.0
    execution_update_fraction_by_axis: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # Reject a completed asynchronous plan if the observation used to build
    # it is older than this when the rollout adapter is ready to apply it.
    maximum_plan_age_seconds: float = 0.30
    # Do not launch successive planner evaluations faster than the measured
    # model-token cadence.  A zero value preserves the legacy unthrottled
    # scheduler (useful for offline callers and timing tests).
    planner_minimum_interval_seconds: float = 0.12
    deadline_seconds: float = 0.040
    seed: int = 20260805

    def __post_init__(self) -> None:
        tuple_fields = {
            "noise_std": 4,
            "reference_seed_response_gain": 4,
            "reference_seed_time_constant_seconds": 4,
            "reference_seed_feedback_gain": 4,
            "reference_seed_position_feedback_gain": 3,
            "terminal_stop_feedback_gain": 4,
            "manual_limit": 4,
            "command_slew_per_second": 4,
            "modal_action_trust_region_std": 5,
            "modal_action_soft_support_std": 5,
            "modal_action_innovation_soft_support_std": 5,
            "modal_action_innovation_limit_std": 5,
            "command_smoothing_axis_weight": 4,
            "execution_update_fraction_by_axis": 4,
            "position_tracking_weight": 3,
            "position_tracking_scale_m": 3,
            "terminal_position_tracking_weight": 3,
            "body_velocity_tracking_weight": 3,
            "ned_velocity_tracking_weight": 3,
            "velocity_tracking_scale_m_s": 3,
        }
        for name, length in tuple_fields.items():
            values = _finite_tuple(getattr(self, name), length, name)
            if any(value < 0.0 for value in values):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, values)
        for name in (
            "manual_limit",
            "command_slew_per_second",
            "position_tracking_scale_m",
            "velocity_tracking_scale_m_s",
        ):
            if any(value <= 0.0 for value in getattr(self, name)):
                raise ValueError(f"{name} must be positive")
        if any(value <= 0.0 for value in self.modal_action_trust_region_std):
            raise ValueError("modal action trust-region widths must be positive")
        if any(
            soft >= hard
            for soft, hard in zip(
                self.modal_action_soft_support_std,
                self.modal_action_trust_region_std,
                strict=True,
            )
        ):
            raise ValueError("soft modal action support must be below the hard limit")
        if any(
            soft >= hard
            for soft, hard in zip(
                self.modal_action_innovation_soft_support_std,
                self.modal_action_innovation_limit_std,
                strict=True,
            )
        ):
            raise ValueError(
                "soft modal innovation support must be below the hard limit"
            )
        if any(value <= 0.0 for value in self.reference_seed_response_gain):
            raise ValueError("reference seed response gains must be positive")
        if any(value < 0.0 for value in self.command_smoothing_axis_weight):
            raise ValueError("command smoothing axis weights must be nonnegative")
        if any(
            value <= 0.0 or value > 1.0
            for value in self.execution_update_fraction_by_axis
        ):
            raise ValueError("per-axis execution update fractions must be in (0, 1]")
        if (
            self.sample_count < 8
            or self.iterations < 1
            or self.command_hold_steps < 1
            or self.common_action_prefix_steps < 0
            or self.execution_hold_steps < 1
        ):
            raise ValueError("MPPI sampling dimensions are too small")
        if self.temperature <= 0.0 or not 0.0 <= self.noise_correlation < 1.0:
            raise ValueError("invalid MPPI temperature or noise correlation")
        if not 0.0 <= self.wide_noise_fraction <= 1.0:
            raise ValueError("wide noise fraction must be in [0, 1]")
        if not math.isfinite(self.wide_noise_scale) or self.wide_noise_scale < 1.0:
            raise ValueError("wide noise scale must be finite and at least one")
        if not 0.0 <= self.reference_seed_sample_fraction <= 1.0:
            raise ValueError("reference seed sample fraction must be in [0, 1]")
        if not 0.0 <= self.plan_smoothing_strength <= 1.0:
            raise ValueError("plan smoothing strength must be in [0, 1]")
        if not isinstance(self.terminal_position_stop_only, bool):
            raise ValueError("terminal position stop-only flag must be boolean")
        if not isinstance(self.include_learned_residual, bool):
            raise ValueError("include_learned_residual must be boolean")
        if not 0.0 < self.optimization_horizon_seconds:
            raise ValueError("optimization horizon must be positive")
        if not 1.0 <= self.target_effective_sample_size <= self.sample_count:
            raise ValueError(
                "target effective sample size must be within the sample batch"
            )
        if not 0.0 < self.nominal_update_rate <= 1.0:
            raise ValueError("nominal update rate must be in (0, 1]")
        if any(value > 1.0 for value in self.manual_limit):
            raise ValueError("MANUAL limits must not exceed one")
        positive = (
            "yaw_tracking_scale_rad",
            "yaw_rate_tracking_scale_rad_s",
            "reference_seed_feedback_decay_seconds",
            "terminal_stop_body_velocity_threshold_m_s",
            "terminal_stop_yaw_rate_threshold_rad_s",
            "maximum_predicted_tilt_rad",
        )
        if any(float(getattr(self, name)) <= 0.0 for name in positive):
            raise ValueError("MPPI physical scales must be positive")
        nonnegative = (
            "yaw_tracking_weight",
            "yaw_rate_tracking_weight",
            "command_rate_weight",
            "motor_saturation_weight",
            "tilt_barrier_weight",
            "minimum_predicted_improvement",
            "minimum_relative_improvement",
            "model_action_support_weight",
            "model_action_innovation_support_weight",
            "cost_horizon_seconds",
            "rollout_horizon_seconds",
            "planner_latency_compensation_seconds",
            "maximum_plan_age_seconds",
            "planner_minimum_interval_seconds",
            "deadline_seconds",
        )
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) < 0.0
            for name in nonnegative
        ):
            raise ValueError("MPPI weights and limits must be finite and nonnegative")
        if not 0.0 < self.motor_soft_limit <= 1.0:
            raise ValueError("motor soft limit must be in (0, 1]")
        if not 0.0 < self.execution_update_fraction <= 1.0:
            raise ValueError("execution update fraction must be in (0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {"implementation_id": IMPLEMENTATION_ID, **asdict(self)}


def load_jepa_mppi_config(path: str | Path) -> JepaMppiConfig:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JEPA-MPPI config must contain a JSON object")
    expected = set(JepaMppiConfig.__dataclass_fields__)
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"unknown JEPA-MPPI config keys: {sorted(unknown)}")
    return JepaMppiConfig(**payload)


def standard_bluerov2_manual_mixer(
    command: Tensor,
    roll_command: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """FRAME_CONFIG=1 MANUAL 4DoF allocation before PWM quantization."""
    if command.shape[-1] != 4:
        raise ValueError("MANUAL command must end in four axes")
    surge, sway, heave, yaw = command.unbind(dim=-1)
    if roll_command is None:
        roll = torch.zeros_like(heave)
    else:
        roll = torch.as_tensor(roll_command, device=command.device, dtype=command.dtype)
        while roll.ndim < heave.ndim:
            roll = roll.unsqueeze(-1)
        roll = roll.expand_as(heave)
    raw = torch.stack(
        (
            yaw - surge + sway,
            -yaw - surge - sway,
            -yaw + surge + sway,
            yaw + surge - sway,
            roll - heave,
            -roll - heave,
        ),
        dim=-1,
    )
    return raw.clamp(-1.0, 1.0), raw


def _motor_to_modal(values: Tensor) -> Tensor:
    """Project six mixer outputs onto the retained five-dimensional manifold."""
    if values.shape[-1] != 6:
        raise ValueError("motor values must end in six channels")
    m1, m2, m3, m4, m5, m6 = values.unbind(dim=-1)
    return torch.stack(
        (
            (-m1 - m2 + m3 + m4) / 4.0,
            (m1 - m2 + m3 - m4) / 4.0,
            -(m5 + m6) / 2.0,
            (m1 - m2 - m3 + m4) / 4.0,
            (m5 - m6) / 2.0,
        ),
        dim=-1,
    )


def _modal_to_motor(values: Tensor) -> Tensor:
    """Reconstruct six mixer outputs for the history prefix."""
    if values.shape[-1] != 5:
        raise ValueError("modal values must end in five channels")
    surge, sway, heave, yaw, roll = values.unbind(dim=-1)
    return torch.stack(
        (
            yaw - surge + sway,
            -yaw - surge - sway,
            -yaw + surge + sway,
            yaw + surge - sway,
            roll - heave,
            -roll - heave,
        ),
        dim=-1,
    )


def _rotation_roll_pitch_yaw(rotation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.asin(rotation[..., 2, 0].neg().clamp(-1.0, 1.0))
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    return roll, pitch, yaw


def _wrap_angle(values: Tensor) -> Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


@dataclass(frozen=True)
class JepaMppiResult:
    command: np.ndarray
    planned_command: np.ndarray
    nominal_command: np.ndarray
    predicted_physical_observation: np.ndarray
    predicted_position_delta_ned_m: np.ndarray | None
    accepted: bool
    reason: str
    diagnostics: dict[str, Any]


class ReferenceConditionedJepaMppi:
    """Absolute-action MPPI with JEPA as its only trajectory model."""

    def __init__(
        self,
        model: nn.Module,
        standardizer: Standardizer,
        *,
        history_steps: int,
        model_horizon_steps: int,
        sample_rate_hz: float,
        config: JepaMppiConfig = JepaMppiConfig(),
        device: str | torch.device = "cpu",
        stage2_implementation: str | None = None,
    ) -> None:
        if history_steps < 2 or model_horizon_steps < 1 or sample_rate_hz <= 0.0:
            raise ValueError("invalid model time contract")
        if model_horizon_steps % config.command_hold_steps:
            raise ValueError("model horizon must be divisible by command hold steps")
        self.device = torch.device(device)
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.stage2_implementation = (
            str(stage2_implementation)
            if stage2_implementation is not None
            else type(model).__name__
        )
        self.standardizer = standardizer
        self.history_steps = int(history_steps)
        self.model_horizon_steps = int(model_horizon_steps)
        self.sample_rate_hz = float(sample_rate_hz)
        self.config = config
        self.knot_count = model_horizon_steps // config.command_hold_steps
        self.optimization_knot_count = min(
            self.knot_count,
            max(
                1,
                int(
                    math.ceil(
                        config.optimization_horizon_seconds
                        * self.sample_rate_hz
                        / config.command_hold_steps
                    )
                ),
            ),
        )
        self.rollout_horizon_steps = (
            min(
                self.model_horizon_steps,
                max(
                    1,
                    int(round(config.rollout_horizon_seconds * self.sample_rate_hz)),
                ),
            )
            if config.rollout_horizon_seconds > 0.0
            else self.model_horizon_steps
        )
        if self.rollout_horizon_steps % config.command_hold_steps:
            raise ValueError("rollout horizon must align with command hold steps")
        self.rollout_knot_count = (
            self.rollout_horizon_steps // config.command_hold_steps
        )
        if self.optimization_knot_count > self.rollout_knot_count:
            raise ValueError("optimization horizon cannot exceed rollout horizon")
        self.cost_horizon_steps = (
            min(
                self.rollout_horizon_steps,
                max(
                    1,
                    int(round(config.cost_horizon_seconds * self.sample_rate_hz)),
                ),
            )
            if config.cost_horizon_seconds > 0.0
            else self.rollout_horizon_steps
        )
        self.nominal_command = torch.zeros(
            self.knot_count, 4, dtype=torch.float32, device=self.device
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)
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
        if self._action_mean.numel() not in (5, 6):
            raise ValueError(
                "JEPA-MPPI supports five modal or six motor action channels"
            )
        self.action_dim = int(self._action_mean.numel())
        if (
            self.config.planner_latency_compensation_seconds
            >= self.rollout_horizon_steps / self.sample_rate_hz
        ):
            raise ValueError(
                "planner latency compensation must be shorter than the rollout horizon"
            )
    @staticmethod
    def _checkpoint_time_contract(checkpoint: Mapping[str, Any]) -> dict[str, float | int]:
        """Return the explicit recorder/model/action timing contract.

        Older checkpoints omitted these fields and are intentionally rejected;
        silently inferring their meaning is what allowed MPPI to query the
        model at a different action cadence than the training loader.
        """
        required = (
            "model_step_stride",
            "sample_rate_hz",
            "recorder_sample_rate_hz",
            "model_step_seconds",
            "action_interval_seconds",
        )
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(
                "checkpoint is missing the unified Stage-2 time contract: "
                f"{missing}"
            )
        stride = float(checkpoint["model_step_stride"])
        sample_rate = float(checkpoint["sample_rate_hz"])
        recorder_rate = float(checkpoint["recorder_sample_rate_hz"])
        model_step_seconds = float(checkpoint["model_step_seconds"])
        action_interval_seconds = float(checkpoint["action_interval_seconds"])
        if stride <= 0.0 or sample_rate <= 0.0 or recorder_rate <= 0.0:
            raise ValueError("checkpoint contains an invalid Stage-2 time contract")
        expected_rate = recorder_rate / stride
        if not math.isclose(sample_rate, expected_rate, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("checkpoint model rate does not match recorder stride")
        expected_seconds = 1.0 / sample_rate
        if not math.isclose(
            model_step_seconds, expected_seconds, rel_tol=1e-6, abs_tol=1e-6
        ) or not math.isclose(
            action_interval_seconds, expected_seconds, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("checkpoint action interval does not match model rate")
        return {
            "model_step_stride": stride,
            "sample_rate_hz": sample_rate,
            "recorder_sample_rate_hz": recorder_rate,
            "model_step_seconds": model_step_seconds,
            "action_interval_seconds": action_interval_seconds,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: JepaMppiConfig = JepaMppiConfig(),
        device: str | torch.device = "cpu",
    ) -> "ReferenceConditionedJepaMppi":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("feature_contract_id") != FEATURE_CONTRACT_ID:
            raise ValueError("checkpoint does not use the retained feature contract")
        base_implementation = checkpoint.get("implementation")
        if base_implementation != STAGE2_IMPLEMENTATION:
            raise ValueError("checkpoint is not the retained Stage-2 implementation")
        readiness = checkpoint.get("training_readiness")
        if not isinstance(readiness, Mapping) or readiness.get("passed") is not True:
            raise ValueError("checkpoint failed Stage-2 training readiness")
        if checkpoint.get("action_contract_id") != ACTION_CONTRACT_ID:
            raise ValueError("checkpoint does not use the measured-modal action contract")
        if checkpoint.get("actuator_model") != ACTUATOR_MODEL_ID:
            raise ValueError("checkpoint does not declare the retained actuator model")
        if tuple(checkpoint.get("action_labels", ())) != ACTION_LABELS:
            raise ValueError("checkpoint action labels do not match the modal contract")
        contract = cls._checkpoint_time_contract(checkpoint)
        model, checkpoint = load_stage2_checkpoint(checkpoint_path, device=device)
        history_steps = int(checkpoint["history_steps"])
        return cls(
            model,
            Standardizer.from_dict(checkpoint["standardizer"]),
            history_steps=history_steps,
            model_horizon_steps=int(checkpoint["horizon_steps"]),
            sample_rate_hz=contract["sample_rate_hz"],
            config=config,
            device=device,
            stage2_implementation=str(base_implementation),
        )

    def _set_nominal_command(self, command: np.ndarray | None = None) -> None:
        """Reset only the action-plan warm start."""
        values = (
            np.zeros(4, dtype=np.float32)
            if command is None
            else np.asarray(command, dtype=np.float32)
        )
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("reset command must contain four finite axes")
        limit = np.asarray(self.config.manual_limit, dtype=np.float32)
        values = np.clip(values, -limit, limit)
        self.nominal_command.copy_(
            torch.as_tensor(values, device=self.device).expand_as(self.nominal_command)
        )

    def reset_plan(self, command: np.ndarray | None = None) -> None:
        """Reset the MPPI warm start after a stale asynchronous result."""
        self._set_nominal_command(command)

    def reset(self, command: np.ndarray | None = None) -> None:
        """Reset controller state at an episode boundary."""
        self._set_nominal_command(command)

    def _prepare_history(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
    ) -> tuple[Tensor, Tensor, Tensor]:
        observation = np.asarray(observation_history, dtype=np.float32).copy()
        action = np.asarray(action_history, dtype=np.float32)
        if observation.shape != (self.history_steps, len(OBSERVATION_LABELS)):
            raise ValueError("observation history shape mismatch")
        if action.shape == (self.history_steps, 6) and self.action_dim == 5:
            action = np.stack(
                (
                    (-action[:, 0] - action[:, 1] + action[:, 2] + action[:, 3]) / 4.0,
                    (action[:, 0] - action[:, 1] + action[:, 2] - action[:, 3]) / 4.0,
                    -(action[:, 4] + action[:, 5]) / 2.0,
                    (action[:, 0] - action[:, 1] - action[:, 2] + action[:, 3]) / 4.0,
                    (action[:, 4] - action[:, 5]) / 2.0,
                ),
                axis=-1,
            ).astype(np.float32)
        if action.shape != (self.history_steps, self.action_dim):
            raise ValueError("action history shape mismatch")
        if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(action)):
            raise ValueError("controller history contains non-finite values")
        current = torch.as_tensor(
            observation[-1].copy(), dtype=torch.float32, device=self.device
        )
        observation[:, DEPTH_INDEX] -= observation[-1, DEPTH_INDEX]
        standardized_observation = (
            torch.as_tensor(observation, device=self.device) - self._observation_mean
        ) / self._observation_std
        standardized_action = (
            torch.as_tensor(action, device=self.device) - self._action_mean
        ) / self._action_std
        return standardized_observation, standardized_action, current

    def _reference_values(
        self,
        values: object,
        *,
        width: int,
        times: np.ndarray | None,
        angular: bool = False,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if width == 1:
            if array.ndim == 0:
                array = array.reshape(1, 1)
            elif array.ndim == 1:
                array = array[:, None]
        elif array.shape == (width,):
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"reference field must have {width} channels")
        if not np.all(np.isfinite(array)):
            raise ValueError("reference fields must be finite")
        if len(array) == 1:
            return np.repeat(array, self.model_horizon_steps, axis=0)
        if times is None:
            if len(array) != self.model_horizon_steps:
                raise ValueError("untimed reference must match the model horizon")
            return array
        if len(array) != len(times):
            raise ValueError("time-stamped reference fields must align with time")
        query = np.arange(1, self.model_horizon_steps + 1, dtype=np.float64)
        query /= self.sample_rate_hz
        output = np.empty((self.model_horizon_steps, width), dtype=np.float64)
        for channel in range(width):
            source = np.unwrap(array[:, channel]) if angular else array[:, channel]
            output[:, channel] = np.interp(query, times, source)
        if angular:
            output = np.arctan2(np.sin(output), np.cos(output))
        return output

    def _reference_mask(
        self,
        values: object | None,
        *,
        width: int,
        times: np.ndarray | None,
    ) -> np.ndarray:
        if values is None:
            return np.ones((self.model_horizon_steps, width), dtype=np.float32)
        interpolated = self._reference_values(
            np.asarray(values, dtype=np.float32), width=width, times=times
        )
        return (interpolated > 0.5).astype(np.float32)

    def _prepare_reference(
        self, reference: ReferenceTrajectory
    ) -> dict[str, tuple[Tensor, Tensor]]:
        times = None
        if reference.time_seconds is not None:
            times = np.asarray(reference.time_seconds, dtype=np.float64)
            if (
                times.ndim != 1
                or len(times) < 1
                or not np.all(np.isfinite(times))
                or np.any(times < 0.0)
                or (len(times) > 1 and np.any(np.diff(times) <= 0.0))
            ):
                raise ValueError(
                    "reference time must be finite, nonnegative, and increasing"
                )
        specifications = (
            ("position", reference.position_ned_m, reference.position_mask, 3, False),
            (
                "body_velocity",
                reference.linear_velocity_body_m_s,
                reference.body_velocity_mask,
                3,
                False,
            ),
            (
                "ned_velocity",
                reference.linear_velocity_ned_m_s,
                reference.ned_velocity_mask,
                3,
                False,
            ),
            ("yaw", reference.yaw_rad, reference.yaw_mask, 1, True),
            ("yaw_rate", reference.yaw_rate_rad_s, reference.yaw_rate_mask, 1, False),
        )
        prepared: dict[str, tuple[Tensor, Tensor]] = {}
        for name, values, mask, width, angular in specifications:
            if values is None:
                continue
            target = self._reference_values(
                values, width=width, times=times, angular=angular
            )
            target_mask = self._reference_mask(mask, width=width, times=times)
            if not np.any(target_mask):
                continue
            prepared[name] = (
                torch.as_tensor(target, dtype=torch.float32, device=self.device),
                torch.as_tensor(target_mask, dtype=torch.float32, device=self.device),
            )
        if not prepared:
            raise ValueError("reference trajectory does not contain an active target")
        return prepared

    def _project_commands(
        self,
        desired: Tensor,
        previous_command: Tensor,
        previous_modal_action: Tensor | None = None,
        roll_command: Tensor | None = None,
    ) -> Tensor:
        """Project plans into command and learned-action support.

        ``previous_command`` is the last MANUAL command and therefore defines
        the actuator-facing slew boundary. ``previous_modal_action`` is the
        modal action corresponding to the command that is currently applied
        at the plant boundary. They are not interchangeable when the mixer or
        actuator dynamics lag the command.
        """
        if desired.ndim != 3 or desired.shape[1:] != (self.knot_count, 4):
            raise ValueError("candidate commands must be [sample, knot, 4]")
        if previous_command.shape != (4,):
            raise ValueError("previous command must contain four axes")
        if roll_command is None:
            roll_command = torch.zeros(
                self.knot_count, device=self.device, dtype=desired.dtype
            )
        roll_command = self._prepare_roll_tensor(roll_command, desired.dtype)
        if previous_modal_action is None:
            previous_modal_action = torch.cat(
                (previous_command, roll_command[:1]), dim=0
            )
        if self.action_dim == 5 and previous_modal_action.shape not in ((4,), (5,)):
            raise ValueError("previous modal action must contain four or five axes")
        limit = torch.as_tensor(self.config.manual_limit, device=self.device)
        slew = torch.as_tensor(self.config.command_slew_per_second, device=self.device)
        desired = desired.clamp(-limit, limit)
        # Keep the combined horizontal allocation inside the configured motor
        # headroom before applying slew. This matches the six-thruster mixer
        # used by the runtime and prevents MPPI from selecting clipped motors.
        _, raw_motor = standard_bluerov2_manual_mixer(
            desired, roll_command.unsqueeze(0)
        )
        horizontal_peak = raw_motor[..., :4].abs().amax(dim=-1, keepdim=True)
        horizontal_scale = torch.minimum(
            torch.ones_like(horizontal_peak),
            torch.as_tensor(self.config.motor_soft_limit, device=self.device)
            / horizontal_peak.clamp_min(1e-6),
        )
        desired = desired.clone()
        desired[..., (0, 1, 3)] *= horizontal_scale
        if self.action_dim == 5:
            # The learned model does not see the four MANUAL channels directly:
            # it sees the mixer-projected modal action.  A command-space bound
            # therefore does not guarantee that the model is being queried in
            # distribution, especially for yaw whose training std is small.
            # Project the candidate in the actual model coordinates and map
            # the four controlled modes back to MANUAL before slew limiting.
            # Convert the *clipped* mixer output, exactly as the Stage-2
            # action path does. Converting ``raw_motor`` here would let a
            # command outside the motor manifold pass a different action to
            # the projector than the model receives during prediction.
            motor, _ = standard_bluerov2_manual_mixer(
                desired, roll_command.unsqueeze(0)
            )
            modal = _motor_to_modal(motor)
            trust = torch.as_tensor(
                self.config.modal_action_trust_region_std,
                device=self.device,
                dtype=desired.dtype,
            )
            lower = self._action_mean[:4] - trust[:4] * self._action_std[:4]
            upper = self._action_mean[:4] + trust[:4] * self._action_std[:4]
            # Roll is a fixed safety auxiliary channel, not an MPPI decision
            # variable. Do not silently clamp it here and then execute the
            # unclamped roll command; the exact roll value is retained in the
            # model action and audited by the nonlinear support terms.
            modal_controlled = torch.maximum(
                torch.minimum(modal[..., :4], upper), lower
            )
            desired = modal_controlled
        current_command = previous_command.expand(len(desired), -1)
        if previous_modal_action.numel() == 4:
            previous_modal_action = torch.cat(
                (previous_modal_action, roll_command[:1]), dim=0
            )
        current_modal = previous_modal_action.expand(len(desired), -1).clone()
        projected = []
        for index, target in enumerate(desired.unbind(dim=1)):
            duration = (
                self.config.execution_hold_steps / self.sample_rate_hz
                if index == 0
                else self.config.command_hold_steps / self.sample_rate_hz
            )
            maximum_step = slew * duration
            command_lower = current_command - maximum_step
            command_upper = current_command + maximum_step
            if self.action_dim == 5:
                innovation_limit = (
                    torch.as_tensor(
                        self.config.modal_action_innovation_limit_std[:4],
                        device=self.device,
                        dtype=desired.dtype,
                    )
                    * self._action_std[:4]
                )
                modal_lower = current_modal[:, :4] - innovation_limit
                modal_upper = current_modal[:, :4] + innovation_limit
                lower = torch.maximum(command_lower, modal_lower)
                upper = torch.minimum(command_upper, modal_upper)
                interval_feasible = lower <= upper
                supported = torch.maximum(torch.minimum(target, upper), lower)
                # No action can obey both boundaries when these intervals are
                # disjoint. Preserve the MANUAL slew contract in that case;
                # nonlinear support validation will reject the plan.
                command_only = torch.maximum(
                    torch.minimum(target, command_upper), command_lower
                )
                current_command = torch.where(
                    interval_feasible, supported, command_only
                )
                # Keep the state in the coordinates consumed by Stage 2.  A
                # MANUAL vector is not a modal action when the mixer has a
                # non-zero roll safety command or actuator allocation.
                current_modal = _motor_to_modal(
                    standard_bluerov2_manual_mixer(
                        current_command, roll_command[index]
                    )[0]
                )
            else:
                current_command = torch.maximum(
                    torch.minimum(target, command_upper), command_lower
                )
                current_modal = current_command
            projected.append(current_command)
        return torch.stack(projected, dim=1)

    def _smooth_command_plans(
        self,
        plans: Tensor,
        previous_command: Tensor,
    ) -> Tensor:
        """Filter complete knot plans before model evaluation.

        The finite-horizon plan is available in full, so a symmetric filter is
        causal with respect to execution.  The first knot is anchored to the
        last command actually sent to the vehicle and the terminal boundary is
        held constant.  Filtering candidates here guarantees Stage 2 scores
        the same smooth plan that can be selected for execution.
        """

        if plans.ndim != 3 or plans.shape[1:] != (self.knot_count, 4):
            raise ValueError("candidate commands must be [sample, knot, 4]")
        if previous_command.shape != (4,):
            raise ValueError("previous command must contain four axes")
        strength = self.config.plan_smoothing_strength
        if strength <= 0.0:
            return plans
        left = torch.cat(
            (
                previous_command.view(1, 1, 4).expand(len(plans), -1, -1),
                plans[:, :-1],
            ),
            dim=1,
        )
        right = torch.cat((plans[:, 1:], plans[:, -1:]), dim=1)
        filtered = 0.25 * left + 0.50 * plans + 0.25 * right
        return plans + strength * (filtered - plans)

    def _initial_plan(
        self,
        previous_command: Tensor,
        previous_modal_action: Tensor | None = None,
        roll_command: Tensor | None = None,
    ) -> Tensor:
        """Return the shifted solution projected from the applied action boundary."""
        return self._project_commands(
            self.nominal_command.unsqueeze(0),
            previous_command,
            previous_modal_action,
            roll_command,
        )[0]

    def _reference_action_seed(
        self,
        reference: Mapping[str, tuple[Tensor, Tensor]],
        nominal_plan: Tensor,
        previous_command: Tensor,
        previous_modal_action: Tensor | None = None,
        roll_command: Tensor | None = None,
    ) -> Tensor:
        """Build a dynamics-aware proposal centre from an available twist target.

        This is not the rollout model or an accepted fallback. It only places
        part of the finite sample population near commands that have the right
        physical direction and scale. JEPA evaluates this seed and every noisy
        plan before any command can be executed.
        """

        seed = nominal_plan.clone()
        gains = torch.as_tensor(
            self.config.reference_seed_response_gain,
            device=self.device,
            dtype=seed.dtype,
        )

        def inverse_response(
            values: Tensor,
            mask: Tensor,
            time_constants: Tensor,
        ) -> Tensor:
            if values.shape[0] <= 1 or not torch.any(time_constants > 0.0):
                return values
            derivative = torch.zeros_like(values)
            derivative[0] = (values[1] - values[0]) * self.sample_rate_hz
            derivative[-1] = (values[-1] - values[-2]) * self.sample_rate_hz
            if values.shape[0] > 2:
                derivative[1:-1] = (
                    0.5 * (values[2:] - values[:-2]) * self.sample_rate_hz
                )
            derivative_mask = mask.bool()
            derivative_mask[0] &= mask[1].bool()
            derivative_mask[-1] &= mask[-2].bool()
            if values.shape[0] > 2:
                derivative_mask[1:-1] &= mask[:-2].bool() & mask[2:].bool()
            derivative = torch.where(
                derivative_mask,
                derivative,
                torch.zeros_like(derivative),
            )
            return values + time_constants * derivative

        def knot_average(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
            width = values.shape[-1]
            grouped_values = values.reshape(
                self.knot_count, self.config.command_hold_steps, width
            )
            grouped_mask = mask.reshape(
                self.knot_count, self.config.command_hold_steps, width
            )
            count = grouped_mask.sum(dim=1)
            averaged = (grouped_values * grouped_mask).sum(dim=1) / count.clamp_min(1.0)
            return averaged, count > 0.0

        if "body_velocity" in reference:
            target, mask = reference["body_velocity"]
            time_constants = torch.as_tensor(
                self.config.reference_seed_time_constant_seconds[:3],
                device=self.device,
                dtype=target.dtype,
            )
            averaged, active = knot_average(
                inverse_response(target, mask, time_constants), mask
            )
            proposed = averaged / gains[:3]
            # Body FRD +z points down while MANUAL heave is positive up.
            proposed[:, 2] = -proposed[:, 2]
            seed[:, :3] = torch.where(active, proposed, seed[:, :3])
        if "yaw_rate" in reference:
            target, mask = reference["yaw_rate"]
            time_constants = torch.as_tensor(
                self.config.reference_seed_time_constant_seconds[3:],
                device=self.device,
                dtype=target.dtype,
            )
            averaged, active = knot_average(
                inverse_response(target, mask, time_constants), mask
            )
            # Stage 2 v5 models the measured modal action through a calibrated
            # linear control-wrench path.  Do not apply the old signed-square
            # inverse used for raw propeller command seeds.
            proposed = averaged / gains[3]
            seed[:, 3:4] = torch.where(active, proposed, seed[:, 3:4])
        seed_batch = self._smooth_command_plans(
            seed.unsqueeze(0), previous_command
        )
        return self._project_commands(
            seed_batch, previous_command, previous_modal_action, roll_command
        )[0]

    def _feedback_action_seed(
        self,
        reference: Mapping[str, tuple[Tensor, Tensor]],
        reference_seed_plan: Tensor,
        current_observation: Tensor,
        current_position_ned: Tensor,
        previous_command: Tensor,
        previous_modal_action: Tensor | None = None,
        roll_command: Tensor | None = None,
    ) -> Tensor:
        """Bias a proposal toward correcting the measured response error.

        The correction decays across the horizon, so it changes the early
        braking direction without replacing the known future reference. This
        remains a proposal only; the learned rollout and MPPI cost decide
        whether any part of it reaches the executable weighted plan.
        """

        feedback = reference_seed_plan.clone()
        feedback_gain = torch.as_tensor(
            self.config.reference_seed_feedback_gain,
            device=self.device,
            dtype=feedback.dtype,
        )
        terminal_feedback_gain = torch.as_tensor(
            self.config.terminal_stop_feedback_gain,
            device=self.device,
            dtype=feedback.dtype,
        )
        terminal_stop_weight = self._terminal_stop_weight(
            reference, dtype=feedback.dtype
        )
        feedback_gain = feedback_gain + terminal_stop_weight * terminal_feedback_gain
        position_gain = torch.as_tensor(
            self.config.reference_seed_position_feedback_gain,
            device=self.device,
            dtype=feedback.dtype,
        )
        if not torch.any(feedback_gain > 0.0) and not torch.any(position_gain > 0.0):
            return feedback
        response_gain = torch.as_tensor(
            self.config.reference_seed_response_gain,
            device=self.device,
            dtype=feedback.dtype,
        )
        knot_time = (
            torch.arange(
                1,
                self.knot_count + 1,
                device=self.device,
                dtype=feedback.dtype,
            )
            * self.config.command_hold_steps
            / self.sample_rate_hz
        )
        decay = torch.exp(
            -knot_time / self.config.reference_seed_feedback_decay_seconds
        ).unsqueeze(-1)

        if "body_velocity" in reference:
            target, mask = reference["body_velocity"]
            correction = (
                feedback_gain[:3]
                * (target[0] - current_observation[VIO_VELOCITY_SLICE])
                / response_gain[:3]
            )
            # Body FRD +z is down while MANUAL heave is positive up.
            correction[2] = -correction[2]
            feedback[:, :3] += decay * correction * mask[0]
        if "position" in reference and torch.any(position_gain > 0.0):
            target_position, position_mask = reference["position"]
            current_rotation = rotation6d_to_matrix(
                current_observation[ROTATION_6D_SLICE]
            )

            # Reference horizons start one model step in the future. Estimate
            # the current reference position first so expected path motion is
            # not mistaken for accumulated tracking error.
            if "ned_velocity" in reference:
                target_velocity, velocity_mask = reference["ned_velocity"]
                reference_velocity_ned = target_velocity[0] * velocity_mask[0]
            elif "body_velocity" in reference:
                target_velocity, velocity_mask = reference["body_velocity"]
                reference_velocity_ned = current_rotation @ (
                    target_velocity[0] * velocity_mask[0]
                )
            else:
                reference_velocity_ned = torch.zeros(
                    3, device=self.device, dtype=feedback.dtype
                )
            reference_position_now = (
                target_position[0]
                - reference_velocity_ned / self.sample_rate_hz
            )
            position_error_ned = (
                reference_position_now - current_position_ned.reshape(3)
            ) * position_mask[0]
            position_error_body = current_rotation.transpose(0, 1) @ position_error_ned
            desired_velocity_correction = position_gain * position_error_body
            command_correction = desired_velocity_correction / response_gain[:3]
            command_correction[2] = -command_correction[2]
            feedback[:, :3] += decay * command_correction
        if "yaw_rate" in reference:
            target, mask = reference["yaw_rate"]
            correction = (
                feedback_gain[3]
                * (target[0, 0] - current_observation[YAW_RATE_INDEX])
                / response_gain[3]
            )
            feedback[:, 3] += decay[:, 0] * correction * mask[0, 0]
        feedback_batch = self._smooth_command_plans(
            feedback.unsqueeze(0), previous_command
        )
        return self._project_commands(
            feedback_batch,
            previous_command,
            previous_modal_action,
            roll_command,
        )[0]

    def _terminal_stop_weight(
        self,
        reference: Mapping[str, tuple[Tensor, Tensor]],
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return a smooth [0, 1] score for a full stop at horizon end."""

        scores: list[Tensor] = []
        if "body_velocity" in reference:
            target, mask = reference["body_velocity"]
            terminal = target[-1] * mask[-1]
            if torch.any(mask[-1] > 0.0):
                scores.append(
                    (
                        1.0
                        - torch.linalg.vector_norm(terminal)
                        / self.config.terminal_stop_body_velocity_threshold_m_s
                    ).clamp(0.0, 1.0)
                )
        if "yaw_rate" in reference:
            target, mask = reference["yaw_rate"]
            if torch.any(mask[-1] > 0.0):
                scores.append(
                    (
                        1.0
                        - torch.abs(target[-1, 0] * mask[-1, 0])
                        / self.config.terminal_stop_yaw_rate_threshold_rad_s
                    ).clamp(0.0, 1.0)
                )
        if not scores:
            return torch.zeros((), device=self.device, dtype=dtype)
        return torch.stack(scores).amin().to(dtype=dtype)

    def _prepare_roll_plan(self, value: object) -> Tensor:
        return self._prepare_roll_tensor(value, torch.float32)

    def _prepare_roll_tensor(self, value: object, dtype: torch.dtype) -> Tensor:
        values = torch.as_tensor(value, dtype=dtype, device=self.device)
        if values.ndim == 0:
            values = values.repeat(self.knot_count)
        elif values.shape == (1,):
            values = values.repeat(self.knot_count)
        if values.shape != (self.knot_count,) or not torch.isfinite(values).all():
            raise ValueError("roll command must be a scalar or [knot] finite array")
        return values

    def _wide_noise_mask(self) -> Tensor:
        """Return an evenly interleaved mask excluding the three exact anchors."""

        mask = torch.zeros(
            self.config.sample_count, dtype=torch.bool, device=self.device
        )
        available = max(0, self.config.sample_count - 3)
        wide_count = min(
            available,
            int(math.floor(available * self.config.wide_noise_fraction + 0.5)),
        )
        if wide_count == 0:
            return mask
        positions = torch.floor(
            (torch.arange(wide_count, device=self.device, dtype=torch.float32) + 0.5)
            * available
            / wide_count
        ).to(torch.int64)
        mask[3 + positions.clamp_max(available - 1)] = True
        return mask

    def _correlated_noise(self) -> Tensor:
        noise = torch.randn(
            self.config.sample_count,
            self.knot_count,
            4,
            generator=self.generator,
            device=self.device,
        )
        rho = self.config.noise_correlation
        innovation = math.sqrt(1.0 - rho * rho)
        for step in range(1, self.knot_count):
            noise[:, step] = rho * noise[:, step - 1] + innovation * noise[:, step]
        noise *= torch.as_tensor(self.config.noise_std, device=self.device)
        wide_mask = self._wide_noise_mask()
        noise[wide_mask] *= self.config.wide_noise_scale
        if self.optimization_knot_count < self.knot_count:
            noise[:, self.optimization_knot_count :] = 0.0
        noise[0].zero_()
        return noise

    def _prepare_prediction_cache(
        self,
        observation_history: Tensor,
        action_history: Tensor,
    ) -> dict[str, Tensor] | None:
        """Cache the measured-history context shared by all candidates."""
        backbone = getattr(self.model, "backbone", None)
        if (
            backbone is None
            or not hasattr(backbone, "encode_context")
            or not hasattr(self.model, "rollout_from_context")
        ):
            return None
        observation_batch = observation_history.unsqueeze(0)
        action_batch = action_history.unsqueeze(0)
        return {
            "backbone_context": backbone.encode_context(
                observation_batch, action_batch
            )
        }

    def _model_action_sequence(
        self,
        motor: Tensor,
        action_history: Tensor,
        current_modal_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build the exact physical and standardized action seen by Stage 2."""
        motor_steps = motor.repeat_interleave(self.config.command_hold_steps, dim=1)
        current_action = (
            action_history[-1] * self._action_std + self._action_mean
            if current_modal_action is None
            else current_modal_action
        )
        if current_action.shape != (self.action_dim,):
            raise ValueError("current modal action has the wrong shape")
        prefix_steps = min(self.config.common_action_prefix_steps, motor_steps.shape[1])
        if prefix_steps:
            if self.action_dim == 5:
                current_motor = _modal_to_motor(current_action).view(1, 1, -1)
            else:
                current_motor = current_action.view(1, 1, -1)
            motor_steps = motor_steps.clone()
            motor_steps[:, :prefix_steps] = current_motor
        model_action = (
            _motor_to_modal(motor_steps) if self.action_dim == 5 else motor_steps
        )
        latency_steps = (
            self.config.planner_latency_compensation_seconds * self.sample_rate_hz
        )
        if latency_steps > 0.0:
            current_action = current_action.view(1, 1, -1)
            candidate_action = model_action
            full_steps = int(math.floor(latency_steps + 1e-9))
            fraction = latency_steps - full_steps
            model_action = current_action.expand_as(candidate_action).clone()
            if full_steps < candidate_action.shape[1]:
                remaining = candidate_action.shape[1] - full_steps
                if fraction <= 1e-8:
                    model_action[:, full_steps:] = candidate_action[:, :remaining]
                else:
                    model_action[:, full_steps : full_steps + 1] = (
                        fraction * current_action
                        + (1.0 - fraction) * candidate_action[:, :1]
                    )
                    if remaining > 1:
                        model_action[:, full_steps + 1 :] = (
                            fraction * candidate_action[:, : remaining - 1]
                            + (1.0 - fraction) * candidate_action[:, 1:remaining]
                        )
        standardized = (model_action - self._action_mean) / self._action_std
        return model_action, standardized

    def _model_support_terms(
        self,
        motor: Tensor,
        action_history: Tensor,
        *,
        standardized_action: Tensor | None = None,
        current_modal_action: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Score only candidate-controlled action steps against train support."""
        motor = motor[:, : self.rollout_knot_count]
        count = len(motor)
        zero = motor.new_zeros(count)
        zero_by_axis = motor.new_zeros(count, 4)
        if self.action_dim != 5:
            return {
                "model_action_support": zero,
                "model_action_innovation_support": zero,
                "model_action_hard_support": zero,
            }, {
                "absolute_max": zero,
                "innovation_max": zero,
                "absolute_max_by_axis": zero_by_axis,
                "innovation_max_by_axis": zero_by_axis,
                "feasible": torch.ones(count, dtype=torch.bool, device=self.device),
            }

        standardized = standardized_action
        if standardized is None:
            _, standardized = self._model_action_sequence(
                motor,
                action_history,
                current_modal_action,
            )
        latency_prefix = int(
            math.ceil(
                self.config.planner_latency_compensation_seconds
                * self.sample_rate_hz
                - 1e-9
            )
        )
        prefix = min(
            max(self.config.common_action_prefix_steps, latency_prefix),
            standardized.shape[1],
        )
        controlled = standardized[:, prefix:, :4]
        previous = (
            standardized[:, prefix - 1 : prefix, :4]
            if prefix > 0
            else (
                (
                    (current_modal_action - self._action_mean)
                    / self._action_std
                )[:4]
                .view(1, 1, 4)
                .expand(count, -1, -1)
                if current_modal_action is not None
                else action_history[-1, :4].view(1, 1, 4).expand(count, -1, -1)
            )
        )
        innovation = torch.diff(
            torch.cat((previous, standardized[..., :4]), dim=1), dim=1
        )[:, prefix:]
        if controlled.shape[1] == 0:
            return {
                "model_action_support": zero,
                "model_action_innovation_support": zero,
                "model_action_hard_support": zero,
            }, {
                "absolute_max": zero,
                "innovation_max": zero,
                "absolute_max_by_axis": zero_by_axis,
                "innovation_max_by_axis": zero_by_axis,
                "feasible": torch.ones(count, dtype=torch.bool, device=self.device),
            }

        soft_absolute = motor.new_tensor(self.config.modal_action_soft_support_std[:4])
        hard_absolute = motor.new_tensor(self.config.modal_action_trust_region_std[:4])
        soft_innovation = motor.new_tensor(
            self.config.modal_action_innovation_soft_support_std[:4]
        )
        hard_innovation = motor.new_tensor(
            self.config.modal_action_innovation_limit_std[:4]
        )
        absolute_excess = functional.relu(controlled.abs() - soft_absolute) / (
            hard_absolute - soft_absolute
        )
        innovation_excess = functional.relu(innovation.abs() - soft_innovation) / (
            hard_innovation - soft_innovation
        )
        absolute_max = controlled.abs().amax(dim=(1, 2))
        innovation_max = innovation.abs().amax(dim=(1, 2))
        absolute_max_by_axis = controlled.abs().amax(dim=1)
        innovation_max_by_axis = innovation.abs().amax(dim=1)
        feasible = (controlled.abs() <= hard_absolute + 1e-5).all(dim=(1, 2)) & (
            innovation.abs() <= hard_innovation + 1e-5
        ).all(dim=(1, 2))
        hard_barrier = torch.where(feasible, zero, torch.full_like(zero, float("inf")))
        # Plans are piecewise constant, so averaging over time would dilute a
        # single knot-boundary excursion by command_hold_steps. Penalize each
        # axis by its worst horizon excursion instead.
        return {
            "model_action_support": (
                self.config.model_action_support_weight
                * absolute_excess.square().amax(dim=1).mean(dim=1)
            ),
            "model_action_innovation_support": (
                self.config.model_action_innovation_support_weight
                * innovation_excess.square().amax(dim=1).mean(dim=1)
            ),
            "model_action_hard_support": hard_barrier,
        }, {
            "absolute_max": absolute_max,
            "innovation_max": innovation_max,
            "absolute_max_by_axis": absolute_max_by_axis,
            "innovation_max_by_axis": innovation_max_by_axis,
            "feasible": feasible,
        }

    def _predict(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        command: Tensor,
        roll_command: Tensor,
        prediction_cache: Mapping[str, Tensor] | None = None,
        current_modal_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        if roll_command.ndim == 1:
            roll_command = roll_command.unsqueeze(0).expand(command.shape[0], -1)
        motor, raw_motor = standard_bluerov2_manual_mixer(command, roll_command)
        if current_modal_action is None:
            _, normalized_motor = self._model_action_sequence(motor, action_history)
        else:
            measured_modal_action = (
                action_history[-1] * self._action_std + self._action_mean
            )
            # Keep the two-argument path for unchanged boundaries.  Besides
            # avoiding needless tensor work, this preserves compatibility with
            # lightweight test/dummy models that wrap the original helper.
            if torch.equal(current_modal_action, measured_modal_action):
                _, normalized_motor = self._model_action_sequence(
                    motor, action_history
                )
            else:
                _, normalized_motor = self._model_action_sequence(
                    motor, action_history, current_modal_action
                )
        normalized_motor = normalized_motor[:, : self.rollout_horizon_steps]
        count = len(command)
        observation_batch = observation_history.unsqueeze(0).expand(count, -1, -1)
        action_batch = action_history.unsqueeze(0).expand(count, -1, -1)
        control_rollout = getattr(self.model, "rollout_control_from_context", None)
        if control_rollout is not None or hasattr(
            self.model, "rollout_from_context"
        ):
            context = (
                None
                if prediction_cache is None
                else prediction_cache["backbone_context"].expand(count, -1)
            )
            rollout_arguments: dict[str, Any] = {"context": context}
            # True is the model's historical default; omit the keyword in that
            # case so lightweight legacy rollout wrappers remain compatible.
            if (
                hasattr(self.config, "include_learned_residual")
                and not self.config.include_learned_residual
            ):
                rollout_arguments["include_residual"] = bool(
                    self.config.include_learned_residual
                )
            rollout = (
                control_rollout
                if control_rollout is not None
                else self.model.rollout_from_context
            )
            prediction = rollout(
                observation_batch,
                action_batch,
                normalized_motor,
                **rollout_arguments,
            )
        else:
            if not bool(getattr(self.config, "include_learned_residual", True)):
                raise ValueError(
                    "nominal-only ablation requires a context-hybrid Stage-2 model"
                )
            prediction = self.model(
                observation_batch,
                action_batch,
                normalized_motor,
            )
        predicted_position_delta: Tensor | None = None
        if isinstance(prediction, Mapping):
            if "predicted_future_physical_observation" in prediction:
                physical = prediction["predicted_future_physical_observation"]
            elif "predicted_future_observation" in prediction:
                physical = (
                    prediction["predicted_future_observation"] * self._observation_std
                    + self._observation_mean
                )
            else:
                raise KeyError("JEPA model output has no future observation")
            candidate_delta = prediction.get("predicted_position_delta_ned_m")
            if candidate_delta is not None:
                predicted_position_delta = candidate_delta
        else:
            physical = prediction
        if physical.shape != (
            count,
            self.rollout_horizon_steps,
            len(OBSERVATION_LABELS),
        ):
            raise ValueError("JEPA rollout shape differs from controller contract")
        if predicted_position_delta is not None:
            expected = (count, self.rollout_horizon_steps, 3)
            if predicted_position_delta.shape != expected:
                raise ValueError(
                    "JEPA position-delta rollout shape differs from controller contract"
                )
            if not torch.isfinite(predicted_position_delta).all():
                predicted_position_delta = None
        return (
            physical,
            motor,
            raw_motor,
            predicted_position_delta,
            normalized_motor,
        )

    def _tracking_error(
        self,
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
        scale: Tensor,
        weight: Tensor,
    ) -> Tensor:
        error = (predicted - target) / scale
        weighted = error.square() * weight * mask
        # ``weight`` is an axis weight, not the normalization denominator.
        # Including it in the denominator silently cancels the configured
        # tracking scale and makes position/body-velocity terms incomparable
        # with yaw-rate terms.
        denominator = mask.sum().clamp_min(1e-9)
        return weighted.sum(dim=(1, 2)) / denominator

    def _trajectory_cost(
        self,
        prediction: Tensor,
        predicted_position_delta: Tensor | None,
        motor: Tensor,
        raw_motor: Tensor,
        command: Tensor,
        reference: Mapping[str, tuple[Tensor, Tensor]],
        current_position_ned: Tensor,
        previous_command: Tensor,
        current_observation: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cost_steps = self.cost_horizon_steps
        prediction = prediction[:, :cost_steps]
        if predicted_position_delta is not None:
            predicted_position_delta = predicted_position_delta[:, :cost_steps]
        motor = motor[:, :cost_steps]
        raw_motor = raw_motor[:, :cost_steps]
        cost_knots = min(
            command.shape[1],
            int(math.ceil(cost_steps / self.config.command_hold_steps)),
        )
        command = command[:, :cost_knots]
        motor = motor[:, :cost_knots]
        raw_motor = raw_motor[:, :cost_knots]
        reference = {
            name: (target[:cost_steps], mask[:cost_steps])
            for name, (target, mask) in reference.items()
        }
        rotation = rotation6d_to_matrix(prediction[..., ROTATION_6D_SLICE])
        body_velocity = prediction[..., VIO_VELOCITY_SLICE]
        ned_velocity = (rotation @ body_velocity.unsqueeze(-1)).squeeze(-1)
        if predicted_position_delta is None:
            # Fallback for lightweight mock models. The retained Stage 2 model
            # supplies a directly supervised cumulative NED displacement.
            position_delta = torch.cumsum(ned_velocity / self.sample_rate_hz, dim=1)
        else:
            position_delta = predicted_position_delta
        position = current_position_ned + position_delta
        roll, pitch, yaw = _rotation_roll_pitch_yaw(rotation)
        components: dict[str, Tensor] = {}

        vector_terms = [
            (
                "position",
                position,
                self.config.position_tracking_scale_m,
                self.config.position_tracking_weight,
            )
        ]
        # Body and NED velocities are the same physical response in different
        # coordinates. Score only the representation supplied by the reference
        # instead of silently counting both when a caller provides both fields.
        if "body_velocity" in reference:
            vector_terms.append(
                (
                    "body_velocity",
                    body_velocity,
                    self.config.velocity_tracking_scale_m_s,
                    self.config.body_velocity_tracking_weight,
                )
            )
        elif "ned_velocity" in reference:
            vector_terms.append(
                (
                    "ned_velocity",
                    ned_velocity,
                    self.config.velocity_tracking_scale_m_s,
                    self.config.ned_velocity_tracking_weight,
                )
            )
        for name, predicted_value, scale, weight in vector_terms:
            if name not in reference:
                continue
            target, mask = reference[name]
            components[f"{name}_tracking"] = self._tracking_error(
                predicted_value,
                target,
                mask,
                torch.as_tensor(scale, device=self.device),
                torch.as_tensor(weight, device=self.device),
            )

        terminal_position_weight = torch.as_tensor(
            self.config.terminal_position_tracking_weight,
            device=self.device,
        )
        if "position" in reference and torch.any(terminal_position_weight > 0.0):
            target, mask = reference["position"]
            terminal_error = (
                position[:, -1] - target[-1]
            ) / torch.as_tensor(
                self.config.position_tracking_scale_m,
                device=self.device,
            )
            terminal_gate = (
                self._terminal_stop_weight(reference, dtype=position.dtype)
                if self.config.terminal_position_stop_only
                else position.new_ones(())
            )
            components["terminal_position_tracking"] = (
                terminal_error.square() * terminal_position_weight * mask[-1]
            ).sum(dim=1) / mask[-1].sum().clamp_min(1.0) * terminal_gate

        if "yaw" in reference:
            target, mask = reference["yaw"]
            error = _wrap_angle(yaw.unsqueeze(-1) - target)
            components["yaw_tracking"] = (
                self.config.yaw_tracking_weight
                * (error / self.config.yaw_tracking_scale_rad).square()
                * mask
            ).sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)
        if "yaw_rate" in reference:
            target, mask = reference["yaw_rate"]
            error = (
                prediction[..., YAW_RATE_INDEX].unsqueeze(-1) - target
            ) / self.config.yaw_rate_tracking_scale_rad_s
            components["yaw_rate_tracking"] = (
                self.config.yaw_rate_tracking_weight * error.square() * mask
            ).sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)

        tilt = torch.sqrt((roll.square() + pitch.square()).clamp_min(0.0))

        prior = torch.cat(
            (previous_command.expand(len(command), 1, -1), command[:, :-1]), dim=1
        )
        durations = command.new_full(
            (command.shape[1],),
            self.config.command_hold_steps / self.sample_rate_hz,
        )
        durations[0] = 1.0 / self.sample_rate_hz
        slew = torch.as_tensor(self.config.command_slew_per_second, device=self.device)
        smoothing_axis = torch.as_tensor(
            self.config.command_smoothing_axis_weight, device=self.device
        ).view(1, 1, -1)
        normalized_rate = (command - prior) / durations[None, :, None] / slew
        components["command_rate"] = self.config.command_rate_weight * (
            normalized_rate.square() * smoothing_axis
        ).mean(dim=(1, 2))
        components["motor_saturation"] = (
            self.config.motor_saturation_weight
            * functional.relu(raw_motor.abs() - self.config.motor_soft_limit)
            .square()
            .mean(dim=(1, 2))
        )
        components["tilt_barrier"] = self.config.tilt_barrier_weight * functional.relu(
            tilt - self.config.maximum_predicted_tilt_rad
        ).square().mean(dim=1)
        return sum(components.values()), components

    def _evaluate(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        command: Tensor,
        reference: Mapping[str, tuple[Tensor, Tensor]],
        current_position_ned: Tensor,
        previous_command: Tensor,
        current_observation: Tensor,
        roll_command: Tensor,
        prediction_cache: Mapping[str, Tensor] | None = None,
    ) -> tuple[
        Tensor,
        dict[str, Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        dict[str, Tensor],
    ]:
        current_modal_action = None
        if self.action_dim == 5:
            current_motor, _ = standard_bluerov2_manual_mixer(
                previous_command.view(1, 4), roll_command[:1]
            )
            current_modal_action = _motor_to_modal(current_motor)[0]
        (
            prediction,
            motor,
            raw_motor,
            predicted_position_delta,
            standardized_action,
        ) = self._predict(
            observation_history,
            action_history,
            command,
            roll_command,
            prediction_cache,
            current_modal_action,
        )
        cost, components = self._trajectory_cost(
            prediction,
            predicted_position_delta,
            motor,
            raw_motor,
            command,
            reference,
            current_position_ned,
            previous_command,
            current_observation,
        )
        support_components, support_diagnostics = self._model_support_terms(
            motor,
            action_history,
            standardized_action=standardized_action,
            current_modal_action=current_modal_action,
        )
        components.update(support_components)
        cost = cost + sum(support_components.values())
        return (
            cost,
            components,
            prediction,
            motor,
            raw_motor,
            predicted_position_delta,
            support_diagnostics,
        )

    def _shift_plan(self, plan: Tensor) -> Tensor:
        expanded = plan.repeat_interleave(self.config.command_hold_steps, dim=0)
        shift = min(self.config.execution_hold_steps, len(expanded))
        if shift < len(expanded):
            expanded[:-shift] = expanded[shift:].clone()
            expanded[-shift:] = expanded[-shift - 1]
        return expanded.reshape(
            self.knot_count, self.config.command_hold_steps, 4
        ).mean(dim=1)

    def _adaptive_weights(self, costs: Tensor) -> tuple[Tensor, float, float]:
        """Compute softmin weights at the configured target ESS.

        MPPI temperature is scale-dependent because the trajectory cost mixes
        position, velocity, and actuator terms. Solving for temperature from a
        target effective sample size keeps the update useful across domains and
        avoids the nearly-uniform weights seen with a fixed ``0.5`` value.
        """
        finite = torch.isfinite(costs)
        weights = torch.zeros_like(costs)
        count = int(finite.sum().item())
        if count == 0:
            return weights, float(self.config.temperature), 0.0
        finite_costs = costs[finite]
        beta = finite_costs.min()
        deltas = finite_costs - beta
        target = min(float(self.config.target_effective_sample_size), float(count))

        def at_temperature(value: Tensor) -> tuple[Tensor, Tensor]:
            scaled = torch.exp(-deltas / value.clamp_min(1e-9))
            scaled = scaled / scaled.sum().clamp_min(1e-12)
            ess = 1.0 / scaled.square().sum()
            return scaled, ess

        spread = deltas.max()
        if float(spread.detach().cpu()) <= 1e-8 or target >= count - 1e-6:
            temperature = costs.new_tensor(max(self.config.temperature, 1e-6))
            scaled, ess = at_temperature(temperature)
        else:
            # All bisection state remains on-device. Calling ``Tensor.item`` in
            # this loop serialized CUDA more than thirty times per control tick.
            low = torch.maximum(spread * 1e-6, costs.new_tensor(1e-9))
            high = torch.maximum(
                spread * 10.0, costs.new_tensor(self.config.temperature)
            )
            target_tensor = costs.new_tensor(target)
            for _ in range(24):
                middle = 0.5 * (low + high)
                _, middle_ess = at_temperature(middle)
                choose_low = middle_ess < target_tensor
                low = torch.where(choose_low, middle, low)
                high = torch.where(choose_low, high, middle)
            temperature = high
            scaled, ess = at_temperature(temperature)
        weights[finite] = scaled
        return (
            weights,
            float(temperature.detach().cpu()),
            float(ess.detach().cpu()),
        )

    def optimize(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        reference: ReferenceTrajectory,
        previous_command: np.ndarray,
        *,
        current_position_ned_m: np.ndarray | None = None,
        roll_command: float | np.ndarray = 0.0,
        capture_candidates: bool = False,
    ) -> JepaMppiResult:
        started = time.perf_counter()
        previous_values = np.asarray(previous_command, dtype=np.float32)
        if previous_values.shape != (4,) or not np.all(np.isfinite(previous_values)):
            raise ValueError("previous command must contain four finite axes")
        previous = torch.as_tensor(previous_values, device=self.device)
        observation, action, current = self._prepare_history(
            observation_history, action_history
        )
        # Keep the last measured Stage-2 action for provenance only.  The
        # model boundary below is derived from the MANUAL command being sent
        # this tick, because recorder history can be one sample old.
        previous_model_action = action[-1] * self._action_std + self._action_mean
        roll_plan = self._prepare_roll_plan(roll_command)
        previous_modal_action = (
            _motor_to_modal(
                standard_bluerov2_manual_mixer(
                    previous.view(1, 4), roll_plan[:1]
                )[0]
            )[0]
            if self.action_dim == 5
            else previous
        )
        prepared_reference = self._prepare_reference(reference)
        if current_position_ned_m is None:
            if "position" in prepared_reference:
                raise ValueError("position tracking requires the current NED position")
            position = torch.zeros(3, dtype=torch.float32, device=self.device)
        else:
            position_values = np.asarray(current_position_ned_m, dtype=np.float32)
            if position_values.shape != (3,) or not np.all(
                np.isfinite(position_values)
            ):
                raise ValueError(
                    "current NED position must contain three finite values"
                )
            position = torch.as_tensor(position_values, device=self.device)
        position = position.view(1, 1, 3)

        nominal_plan = self._initial_plan(previous, previous_modal_action, roll_plan)
        reference_seed_plan = self._reference_action_seed(
            prepared_reference,
            nominal_plan,
            previous,
            previous_modal_action,
            roll_plan,
        )
        feedback_seed_plan = self._feedback_action_seed(
            prepared_reference,
            reference_seed_plan,
            current,
            position,
            previous,
            previous_modal_action,
            roll_plan,
        )
        final_candidates: Tensor | None = None
        final_cost: Tensor | None = None
        final_components: dict[str, Tensor] | None = None
        final_motor: Tensor | None = None
        final_raw_motor: Tensor | None = None
        final_weights: Tensor | None = None
        final_weighted_plan: Tensor | None = None
        final_weighted_cost: Tensor | None = None
        final_weighted_components: dict[str, Tensor] | None = None
        final_weighted_prediction: Tensor | None = None
        final_weighted_position_delta: Tensor | None = None
        final_candidate_prediction: Tensor | None = None
        final_candidate_position_delta: Tensor | None = None
        final_candidate_support: dict[str, Tensor] | None = None
        final_weighted_support: dict[str, Tensor] | None = None
        final_candidate_sources: list[str] | None = None
        final_wide_noise_mask: Tensor | None = None
        weighted_plan_verified = False
        verification_skipped_reason = ""

        with torch.inference_mode():
            prediction_cache = self._prepare_prediction_cache(observation, action)
            nominal_cost: Tensor | None = None
            best_sample_cost = torch.full((), float("inf"), device=self.device)
            nominal_components: dict[str, Tensor] = {}
            nominal = nominal_plan.clone()
            weighted_plan = nominal.clone()
            adaptive_temperature = self.config.temperature
            effective_sample_size = 0.0
            for iteration in range(self.config.iterations):
                noise = self._correlated_noise()
                wide_noise_mask = self._wide_noise_mask()
                desired_candidates = nominal.unsqueeze(0) + noise
                seeded_count = int(
                    round(
                        self.config.sample_count
                        * self.config.reference_seed_sample_fraction
                    )
                )
                seed_start = max(3, self.config.sample_count - seeded_count)
                seed_end = self.config.sample_count
                reference_seed_end = seed_start + (seeded_count + 1) // 2
                if seeded_count > 0 and seed_start < seed_end:
                    desired_candidates[seed_start:reference_seed_end] = (
                        reference_seed_plan.unsqueeze(0)
                        + noise[seed_start:reference_seed_end]
                    )
                    desired_candidates[reference_seed_end:seed_end] = (
                        feedback_seed_plan.unsqueeze(0)
                        + noise[reference_seed_end:seed_end]
                    )
                candidate_sources = [
                    "shifted_previous_model_plan"
                    for _ in range(self.config.sample_count)
                ]
                for index in range(seed_start, reference_seed_end):
                    candidate_sources[index] = "reference_action_seed"
                for index in range(reference_seed_end, seed_end):
                    candidate_sources[index] = "response_feedback_seed"
                desired_candidates = self._smooth_command_plans(
                    desired_candidates, previous
                )
                candidates = self._project_commands(
                    desired_candidates,
                    previous,
                    previous_modal_action,
                    roll_plan,
                )
                # Candidate zero is the shifted prior model-space solution. It
                # is the only fallback and the comparison point for updates.
                candidates[0] = nominal_plan
                if len(candidates) > 1:
                    candidates[1] = reference_seed_plan
                    candidate_sources[1] = "reference_action_seed_anchor"
                if len(candidates) > 2:
                    candidates[2] = feedback_seed_plan
                    candidate_sources[2] = "response_feedback_seed_anchor"
                candidate_sources[0] = "shifted_previous_model_plan_anchor"
                (
                    costs,
                    components,
                    candidate_prediction,
                    motor,
                    raw_motor,
                    candidate_position_delta,
                    candidate_support,
                ) = self._evaluate(
                    observation,
                    action,
                    candidates,
                    prepared_reference,
                    position,
                    previous,
                    current,
                    roll_plan,
                    prediction_cache,
                )
                costs = torch.where(
                    torch.isfinite(costs), costs, torch.full_like(costs, float("inf"))
                )
                if iteration == 0:
                    nominal_cost = costs[0].clone()
                    nominal_components = {
                        name: values[0].clone() for name, values in components.items()
                    }
                finite = torch.isfinite(costs)
                if finite.any():
                    iteration_best = int(torch.argmin(costs).item())
                    if costs[iteration_best] < best_sample_cost:
                        best_sample_cost = costs[iteration_best].clone()
                    weights, adaptive_temperature, effective_sample_size = (
                        self._adaptive_weights(costs)
                    )
                    weighted_target = self._project_commands(
                        (weights[:, None, None] * candidates).sum(
                            dim=0, keepdim=True
                        ),
                        previous,
                        previous_modal_action,
                        roll_plan,
                    )[0]
                    nominal = self._project_commands(
                        (
                            (1.0 - self.config.nominal_update_rate) * nominal
                            + self.config.nominal_update_rate * weighted_target
                        ).unsqueeze(0),
                        previous,
                        previous_modal_action,
                        roll_plan,
                    )[0]
                    weighted_plan = nominal.clone()
                else:
                    weights = torch.zeros_like(costs)
            final_candidates = candidates
            final_cost = costs
            final_components = components
            final_motor = motor
            final_raw_motor = raw_motor
            final_weights = weights
            final_candidate_prediction = candidate_prediction
            final_candidate_position_delta = candidate_position_delta
            final_candidate_support = candidate_support
            final_candidate_sources = candidate_sources
            final_wide_noise_mask = wide_noise_mask
            # The weighted MPPI plan is the executable proposal. Re-evaluate it
            # after projection; a random sample is retained for diagnostics only.
            pre_blend_weighted_plan = weighted_plan.clone()
            executable_weighted_plan = pre_blend_weighted_plan.clone()
            execution_fraction = (
                self.config.execution_update_fraction
                * torch.as_tensor(
                    self.config.execution_update_fraction_by_axis,
                    device=self.device,
                )
            )
            if torch.any(execution_fraction < 1.0):
                executable_weighted_plan[0] = previous + (
                    execution_fraction * (executable_weighted_plan[0] - previous)
                )
                executable_weighted_plan = self._project_commands(
                    executable_weighted_plan.unsqueeze(0),
                    previous,
                    previous_modal_action,
                    roll_plan,
                )[0]
            elapsed_before_verification = time.perf_counter() - started
            can_verify = self.config.verify_weighted_plan
            if (
                can_verify
                and self.config.deadline_seconds > 0.0
                and elapsed_before_verification > 0.55 * self.config.deadline_seconds
            ):
                can_verify = False
                verification_skipped_reason = "candidate batch consumed realtime budget"
            if can_verify:
                (
                    weighted_costs,
                    weighted_components,
                    weighted_prediction,
                    _,
                    _,
                    weighted_position_delta,
                    weighted_support,
                ) = self._evaluate(
                    observation,
                    action,
                    executable_weighted_plan.unsqueeze(0),
                    prepared_reference,
                    position,
                    previous,
                    current,
                    roll_plan,
                    prediction_cache,
                )
                final_weighted_cost = weighted_costs[0]
                final_weighted_components = {
                    name: values[0].clone()
                    for name, values in weighted_components.items()
                }
                final_weighted_prediction = weighted_prediction
                final_weighted_position_delta = weighted_position_delta
                final_weighted_support = weighted_support
                weighted_plan_verified = True
            else:
                # Standard MPPI executes the projected weighted update itself.
                # Snapping it to the nearest sampled candidate makes small
                # estimator or timing changes switch the executed action
                # discontinuously. The expectation below is only a diagnostic
                # estimate; it is not presented as a nonlinear re-evaluation.
                finite_costs = torch.where(
                    torch.isfinite(costs), costs, torch.zeros_like(costs)
                )
                final_weighted_cost = (weights * finite_costs).sum()
                final_weighted_components = {
                    name: (
                        weights
                        * torch.where(
                            torch.isfinite(values), values, torch.zeros_like(values)
                        )
                    ).sum()
                    for name, values in components.items()
                }
                final_weighted_prediction = (
                    weights[:, None, None] * final_candidate_prediction
                ).sum(dim=0, keepdim=True)
                final_weighted_position_delta = None
                if final_candidate_position_delta is not None:
                    final_weighted_position_delta = (
                        weights[:, None, None] * final_candidate_position_delta
                    ).sum(dim=0, keepdim=True)
                weighted_motor, _ = standard_bluerov2_manual_mixer(
                    executable_weighted_plan.unsqueeze(0), roll_plan.unsqueeze(0)
                )
                _, final_weighted_support = self._model_support_terms(
                    weighted_motor,
                    action,
                    current_modal_action=previous_modal_action,
                )
                verification_skipped_reason = (
                    verification_skipped_reason or "verification disabled"
                )
            final_weighted_plan = executable_weighted_plan
        if self.device.type == "cuda":
            # Account for queued CUDA work before reporting the optimization
            # phase. Otherwise later .cpu() calls hide model latency in
            # unmeasured diagnostics postprocessing.
            torch.cuda.synchronize(self.device)
        core_optimization_seconds = time.perf_counter() - started
        postprocess_started = time.perf_counter()
        if nominal_cost is None:
            raise RuntimeError("MPPI produced no nominal candidate")
        if final_weighted_cost is None or final_weighted_plan is None:
            raise RuntimeError("MPPI produced no weighted plan")
        if final_candidate_prediction is None or final_weighted_prediction is None:
            raise RuntimeError("MPPI produced no physical prediction estimate")
        if final_candidate_support is None or final_weighted_support is None:
            raise RuntimeError("MPPI produced no model-support diagnostics")
        if final_candidate_sources is None or final_wide_noise_mask is None:
            raise RuntimeError("MPPI produced no candidate sampling diagnostics")

        elapsed = core_optimization_seconds
        nominal_value = float(nominal_cost.cpu())
        weighted_value = float(final_weighted_cost.cpu())
        nominal_supported = math.isfinite(nominal_value)
        improvement = (
            nominal_value - weighted_value
            if nominal_supported and math.isfinite(weighted_value)
            else -math.inf
        )
        required = (
            max(
                self.config.minimum_predicted_improvement,
                self.config.minimum_relative_improvement
                * max(abs(nominal_value), 1e-9),
            )
            if nominal_supported
            else math.inf
        )
        accepted = (
            nominal_supported
            and math.isfinite(weighted_value)
            and weighted_value < nominal_value
            and improvement >= required
        )
        reason = (
            "accepted"
            if accepted
            else (
                "model action support unavailable"
                if not nominal_supported
                else "no predicted improvement"
            )
        )
        selected = final_weighted_plan if accepted else nominal_plan
        selected_prediction = (
            final_weighted_prediction[0]
            if accepted
            else final_candidate_prediction[0]
        )
        selected_position_delta = (
            final_weighted_position_delta[0]
            if accepted and final_weighted_position_delta is not None
            else (
                final_candidate_position_delta[0]
                if not accepted and final_candidate_position_delta is not None
                else None
            )
        )
        selected_components = (
            final_weighted_components if accepted else nominal_components
        )
        selected_support = (
            final_weighted_support
            if accepted
            else {name: value[:1] for name, value in final_candidate_support.items()}
        )
        candidate_support = final_candidate_support
        supported_candidate_count = int(
            final_candidate_support["feasible"].sum().cpu()
        )
        final_best_candidate_index = int(torch.argmin(final_cost).item())
        non_anchor_mask = torch.arange(
            self.config.sample_count, device=self.device
        ) >= 3
        local_noise_mask = non_anchor_mask & ~final_wide_noise_mask

        def supported_fraction(mask: Tensor) -> float | None:
            count = int(mask.sum().cpu())
            if count == 0:
                return None
            return float(
                final_candidate_support["feasible"][mask].float().mean().cpu()
            )

        wide_weight_mass = float(final_weights[final_wide_noise_mask].sum().cpu())
        seed_distance_scale = torch.as_tensor(
            self.config.manual_limit, device=self.device
        ).clamp_min(1e-6)

        def normalized_plan_distance(plan: Tensor) -> float:
            return float(
                ((selected - plan) / seed_distance_scale)
                .square()
                .mean()
                .sqrt()
                .cpu()
            )

        diagnostics: dict[str, Any] = {
            "implementation_id": IMPLEMENTATION_ID,
            "stage2_implementation": self.stage2_implementation,
            "reference_contract_id": REFERENCE_CONTRACT_ID,
            "reference_terms": sorted(prepared_reference),
            "nominal_cost": nominal_value,
            "nominal_cost_components": {
                name: float(value.cpu()) for name, value in nominal_components.items()
            },
            "selected_cost": weighted_value if accepted else nominal_value,
            "best_candidate_cost": float(best_sample_cost.cpu()),
            "weighted_plan_cost": weighted_value,
            "predicted_improvement": improvement,
            "required_improvement": required,
            "selected_cost_components": {
                name: float(value.cpu()) for name, value in selected_components.items()
            },
            "timing_seconds": elapsed,
            "sample_count": self.config.sample_count,
            "iterations": self.config.iterations,
            "wide_noise_fraction": self.config.wide_noise_fraction,
            "wide_noise_scale": self.config.wide_noise_scale,
            "wide_noise_candidate_count": int(final_wide_noise_mask.sum().cpu()),
            "wide_noise_supported_fraction": supported_fraction(
                final_wide_noise_mask
            ),
            "local_noise_supported_fraction": supported_fraction(local_noise_mask),
            "wide_candidate_weight_mass": wide_weight_mass,
            "best_candidate_index": final_best_candidate_index,
            "best_candidate_sampling_centre": final_candidate_sources[
                final_best_candidate_index
            ],
            "best_candidate_is_wide": bool(
                final_wide_noise_mask[final_best_candidate_index].item()
            ),
            "selected_plan_distance_to_seed_normalized": {
                "shifted_previous_model_plan": normalized_plan_distance(
                    nominal_plan
                ),
                "reference_action_seed": normalized_plan_distance(
                    reference_seed_plan
                ),
                "response_feedback_seed": normalized_plan_distance(
                    feedback_seed_plan
                ),
            },
            "model_supported_manual_limit": self.config.manual_limit,
            "previous_command": previous.cpu().numpy(),
            "previous_model_action": previous_model_action.cpu().numpy(),
            "previous_modal_action": previous_modal_action.cpu().numpy(),
            "previous_command_to_modal_gap": (previous - previous_modal_action[:4])
            .cpu()
            .numpy(),
            "modal_action_soft_support_std": (
                self.config.modal_action_soft_support_std
            ),
            "modal_action_hard_support_std": (
                self.config.modal_action_trust_region_std
            ),
            "modal_action_innovation_soft_support_std": (
                self.config.modal_action_innovation_soft_support_std
            ),
            "modal_action_innovation_hard_support_std": (
                self.config.modal_action_innovation_limit_std
            ),
            "supported_candidate_count": supported_candidate_count,
            "supported_candidate_fraction": (
                supported_candidate_count / self.config.sample_count
            ),
            "selected_modal_action_standardized_abs_max": float(
                selected_support["absolute_max"][0].cpu()
            ),
            "selected_modal_action_innovation_standardized_abs_max": float(
                selected_support["innovation_max"][0].cpu()
            ),
            "selected_modal_action_standardized_abs_max_by_axis": (
                selected_support["absolute_max_by_axis"][0].cpu().numpy()
            ),
            "selected_modal_action_innovation_standardized_abs_max_by_axis": (
                selected_support["innovation_max_by_axis"][0].cpu().numpy()
            ),
            "prediction_cache_used": prediction_cache is not None,
            "pre_blend_command": pre_blend_weighted_plan[0].cpu().numpy(),
            "executed_command": selected[0].cpu().numpy(),
            "execution_update_fraction": self.config.execution_update_fraction,
            "execution_update_fraction_by_axis": (
                self.config.execution_update_fraction_by_axis
            ),
            "adaptive_temperature": adaptive_temperature,
            "effective_sample_size": effective_sample_size,
            "target_effective_sample_size": self.config.target_effective_sample_size,
            "plan_smoothing_strength": self.config.plan_smoothing_strength,
            "terminal_position_tracking_weight": (
                self.config.terminal_position_tracking_weight
            ),
            "terminal_position_stop_only": self.config.terminal_position_stop_only,
            "optimization_horizon_seconds": self.config.optimization_horizon_seconds,
            "rollout_horizon_steps": self.rollout_horizon_steps,
            "rollout_horizon_seconds": (
                self.rollout_horizon_steps / self.sample_rate_hz
            ),
            "cost_horizon_steps": self.cost_horizon_steps,
            "cost_horizon_seconds": self.cost_horizon_steps / self.sample_rate_hz,
            "planner_latency_compensation_seconds": (
                self.config.planner_latency_compensation_seconds
            ),
            "optimization_knot_count": self.optimization_knot_count,
            "weighted_plan_verified": weighted_plan_verified,
            "verification_skipped_reason": verification_skipped_reason,
            "selected_plan_source": (
                "verified_weighted_plan"
                if accepted and weighted_plan_verified
                else (
                    "projected_mppi_weighted_plan"
                    if accepted
                    else "receding_nominal"
                )
            ),
            "nominal_plan_source": "shifted_previous_model_plan",
            "candidate_sampling_centres": (
                [
                    "shifted_previous_model_plan",
                    "reference_action_seed",
                    "response_feedback_seed",
                ]
                if self.config.reference_seed_sample_fraction > 0.0
                else ["shifted_previous_model_plan"]
            ),
            "reference_seed_sample_fraction": (
                self.config.reference_seed_sample_fraction
            ),
            "reference_seed_time_constant_seconds": (
                self.config.reference_seed_time_constant_seconds
            ),
            "reference_seed_first_command": (reference_seed_plan[0].cpu().numpy()),
            "feedback_seed_first_command": (feedback_seed_plan[0].cpu().numpy()),
            "reference_seed_feedback_gain": self.config.reference_seed_feedback_gain,
            "reference_seed_position_feedback_gain": (
                self.config.reference_seed_position_feedback_gain
            ),
            "terminal_stop_feedback_gain": self.config.terminal_stop_feedback_gain,
            "terminal_stop_weight": float(
                self._terminal_stop_weight(
                    prepared_reference, dtype=reference_seed_plan.dtype
                ).cpu()
            ),
            "reference_seed_feedback_decay_seconds": (
                self.config.reference_seed_feedback_decay_seconds
            ),
            "position_prediction_source": (
                "stage2_predicted_position_delta"
                if selected_position_delta is not None
                else "velocity_integral_fallback"
            ),
        }
        if capture_candidates:
            if any(
                value is None
                for value in (
                    final_candidates,
                    final_cost,
                    final_components,
                    final_motor,
                    final_raw_motor,
                    final_weights,
                )
            ):
                raise RuntimeError("MPPI candidate capture is unavailable")
            diagnostics.update(
                {
                    "candidate_command": final_candidates.cpu().numpy(),
                    "candidate_total_cost": final_cost.cpu().numpy(),
                    "candidate_cost_components": {
                        name: values.cpu().numpy()
                        for name, values in final_components.items()
                    },
                    "candidate_motor": final_motor.cpu().numpy(),
                    "candidate_raw_motor": final_raw_motor.cpu().numpy(),
                    "candidate_weights": final_weights.cpu().numpy(),
                    "candidate_sampling_source": final_candidate_sources,
                    "candidate_wide_noise_mask": (
                        final_wide_noise_mask.cpu().numpy()
                    ),
                    "candidate_predicted_position_delta_ned_m": (
                        None
                        if final_candidate_position_delta is None
                        else final_candidate_position_delta.cpu().numpy()
                    ),
                    "candidate_model_support_feasible": (
                        candidate_support["feasible"].cpu().numpy()
                    ),
                    "candidate_modal_action_standardized_abs_max": (
                        candidate_support["absolute_max"].cpu().numpy()
                    ),
                    "candidate_modal_action_innovation_standardized_abs_max": (
                        candidate_support["innovation_max"].cpu().numpy()
                    ),
                    "candidate_modal_action_standardized_abs_max_by_axis": (
                        candidate_support["absolute_max_by_axis"].cpu().numpy()
                    ),
                    "candidate_modal_action_innovation_standardized_abs_max_by_axis": (
                        candidate_support["innovation_max_by_axis"].cpu().numpy()
                    ),
                    "weighted_plan": pre_blend_weighted_plan.cpu().numpy(),
                    "verified_proposal_plan": final_weighted_plan.cpu().numpy(),
                    "weighted_plan_cost_components": {
                        name: value.cpu().numpy()
                        for name, value in final_weighted_components.items()
                    },
                }
            )

        # Materialize both possible outputs before the final wall-time check.
        # This makes the deadline cover model execution, diagnostics transfer,
        # and result conversion rather than only the queued CUDA kernels.
        selected_command_values = selected[0].cpu().numpy().copy()
        selected_plan_values = selected.cpu().numpy().copy()
        selected_prediction_values = selected_prediction.cpu().numpy().copy()
        selected_position_delta_values = (
            None
            if selected_position_delta is None
            else selected_position_delta.cpu().numpy().copy()
        )
        nominal_plan_values = nominal_plan.cpu().numpy().copy()
        nominal_prediction_values = final_candidate_prediction[0].cpu().numpy().copy()
        nominal_position_delta_values = (
            None
            if final_candidate_position_delta is None
            else final_candidate_position_delta[0].cpu().numpy().copy()
        )
        nominal_component_values = {
            name: float(value.cpu()) for name, value in nominal_components.items()
        }
        nominal_support = {
            name: value[:1] for name, value in final_candidate_support.items()
        }

        def select_deadline_fallback() -> None:
            nonlocal accepted, reason, selected
            nonlocal selected_command_values, selected_plan_values
            nonlocal selected_prediction_values, selected_position_delta_values
            accepted = False
            reason = "deadline exceeded"
            selected = nominal_plan
            selected_command_values = nominal_plan_values[0].copy()
            selected_plan_values = nominal_plan_values.copy()
            selected_prediction_values = nominal_prediction_values
            selected_position_delta_values = nominal_position_delta_values
            diagnostics.update(
                {
                    "selected_cost": nominal_value,
                    "selected_cost_components": nominal_component_values,
                    "executed_command": selected_command_values,
                    "selected_plan_source": "receding_nominal",
                    "selected_modal_action_standardized_abs_max": float(
                        nominal_support["absolute_max"][0].cpu()
                    ),
                    "selected_modal_action_innovation_standardized_abs_max": float(
                        nominal_support["innovation_max"][0].cpu()
                    ),
                    "selected_modal_action_standardized_abs_max_by_axis": (
                        nominal_support["absolute_max_by_axis"][0].cpu().numpy()
                    ),
                    "selected_modal_action_innovation_standardized_abs_max_by_axis": (
                        nominal_support["innovation_max_by_axis"][0].cpu().numpy()
                    ),
                    "selected_plan_distance_to_seed_normalized": {
                        "shifted_previous_model_plan": 0.0,
                        "reference_action_seed": normalized_plan_distance(
                            reference_seed_plan
                        ),
                        "response_feedback_seed": normalized_plan_distance(
                            feedback_seed_plan
                        ),
                    },
                }
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_optimize_seconds = time.perf_counter() - started
        deadline_exceeded = (
            self.config.deadline_seconds > 0.0
            and total_optimize_seconds > self.config.deadline_seconds
        )
        if deadline_exceeded:
            select_deadline_fallback()
        self.nominal_command.copy_(self._shift_plan(selected))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_optimize_seconds = time.perf_counter() - started
        # The state update above is tiny, but it is still part of optimize().
        # If it crosses the boundary, return the nominal fallback consistently.
        if (
            not deadline_exceeded
            and self.config.deadline_seconds > 0.0
            and total_optimize_seconds > self.config.deadline_seconds
        ):
            select_deadline_fallback()
            self.nominal_command.copy_(self._shift_plan(selected))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            total_optimize_seconds = time.perf_counter() - started
        diagnostics.update(
            {
                "core_optimization_seconds": core_optimization_seconds,
                "postprocess_seconds": max(
                    0.0, time.perf_counter() - postprocess_started
                ),
                "total_optimize_seconds": total_optimize_seconds,
                # Retain the established key for existing trace consumers.
                "timing_seconds": total_optimize_seconds,
            }
        )
        return JepaMppiResult(
            command=selected_command_values,
            planned_command=selected_plan_values,
            nominal_command=nominal_plan_values,
            predicted_physical_observation=selected_prediction_values,
            predicted_position_delta_ned_m=selected_position_delta_values,
            accepted=accepted,
            reason=reason,
            diagnostics=diagnostics,
        )

    def warmup(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        previous_command: np.ndarray,
        reference: ReferenceTrajectory | None = None,
        current_position_ned_m: np.ndarray | None = None,
        roll_command: float | np.ndarray = 0.0,
    ) -> float:
        observation, action, current = self._prepare_history(
            observation_history, action_history
        )
        previous_model_action = action[-1] * self._action_std + self._action_mean
        previous_modal_action = (
            previous_model_action if self.action_dim == 5 else previous
        )
        previous = torch.as_tensor(
            np.asarray(previous_command, dtype=np.float32), device=self.device
        )
        if previous.shape != (4,):
            raise ValueError("previous command must contain four axes")
        roll_plan = self._prepare_roll_plan(roll_command)
        plan = self._project_commands(
            previous.expand(self.knot_count, -1).unsqueeze(0),
            previous,
            previous_modal_action,
            roll_plan,
        )[0]
        prepared_reference = self._prepare_reference(
            reference
            if reference is not None
            else ReferenceTrajectory.body_twist(np.zeros(4, dtype=np.float32))
        )
        if current_position_ned_m is None:
            if "position" in prepared_reference:
                raise ValueError("position warmup requires the current NED position")
            position = torch.zeros(1, 1, 3, dtype=torch.float32, device=self.device)
        else:
            values = np.asarray(current_position_ned_m, dtype=np.float32)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "current NED position must contain three finite values"
                )
            position = torch.as_tensor(values, device=self.device).view(1, 1, 3)
        candidates = self._project_commands(
            plan.unsqueeze(0) + self._correlated_noise(),
            previous,
            previous_modal_action,
            roll_plan,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            prediction_cache = self._prepare_prediction_cache(observation, action)
            costs, _, _, _, _, _, _ = self._evaluate(
                observation,
                action,
                candidates,
                prepared_reference,
                position,
                previous,
                current,
                roll_plan,
                prediction_cache,
            )
            finite = torch.isfinite(costs)
            if finite.any():
                beta = costs[finite].min()
                weights = torch.zeros_like(costs)
                weights[finite] = torch.exp(
                    -(costs[finite] - beta) / self.config.temperature
                )
                weights /= weights.sum().clamp_min(1e-12)
                self._project_commands(
                    (weights[:, None, None] * candidates).sum(dim=0, keepdim=True),
                    previous,
                    previous_modal_action,
                    roll_plan,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - started
