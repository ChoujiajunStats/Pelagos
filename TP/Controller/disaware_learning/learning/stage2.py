"""Context-conditioned hybrid world model for underwater trajectory control.

Stage 2 keeps the Stage-1 history representation fixed during each imagined
horizon. Candidate actions pass through context-conditioned actuator lag and a
bounded residual of nominal hydrodynamics. The residual has both a state
dependent bias (buoyancy, current, restoring forces, and unmodelled drag) and
a bounded action-dependent wrench matrix (actuator efficiency and coupling).
Candidate ranking uses causal physical state, action, and fixed history-belief
paths. Optionally, each cumulative candidate Stage-1 latent response is decoded
as a bounded physical response potential. Only its temporal increment enters
the physical rollout, and no future observation is consumed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from disaware_learning.learning.motor_jepa_data import (
    ACTUATOR_MODEL_ID,
    ACTION_CONTRACT_ID,
    ACTION_LABELS,
    DEPTH_INDEX,
    FEATURE_CONTRACT_ID,
    IMU_ACCEL_SLICE,
    IMU_GYRO_SLICE,
    MotorJepaWindowDataset,
    OBSERVATION_LABELS,
    ROTATION_6D_SLICE,
    Standardizer,
    TEMPORAL_RESAMPLING_ID,
    VIO_VELOCITY_SLICE,
    common_sample_rate,
    load_partition_sequences,
    resample_episode_sequences,
)
from disaware_learning.learning.stage1 import (
    IMPLEMENTATION as STAGE1_IMPLEMENTATION,
    PredictiveStateJEPA,
    Stage1Config,
    _cross_episode_indices,
    _matched_episode_indices,
)

IMPLEMENTATION = "predictive_belief_jepa_stage2_factual_rollout_selection_v11"
REPRESENTATION_TRAINING_MODES = (
    "predictive_pretrained_frozen",
    "predictive_pretrained_finetuned",
    "supervised_from_scratch",
)
RATE_LABELS = (
    "linear_x_m_s2",
    "linear_y_m_s2",
    "linear_z_m_s2",
    "angular_x_rad_s2",
    "angular_y_rad_s2",
    "angular_z_rad_s2",
)
LATENT_RESPONSE_FEATURES = (
    "response_potential",
    "legacy_cumulative_from_context",
)
CONTROLLED_RATE_INDICES = (0, 1, 2, 5)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _apply_training_config(
    parser: argparse.ArgumentParser,
    config_path: str,
) -> dict[str, object] | None:
    """Apply JSON hyperparameter defaults without changing required data paths."""
    if not config_path:
        return None
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage-2 training config must use schema_version 1")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Stage-2 training config requires a parameters object")
    allowed = {
        action.dest
        for action in parser._actions
        if action.dest
        not in {
            argparse.SUPPRESS,
            "help",
            "manifest",
            "stage1_checkpoint",
            "output_dir",
            "config",
        }
    }
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown Stage-2 training config parameters: {unknown}")
    parser.set_defaults(**parameters)
    return {
        **payload,
        "path": str(path),
    }


def rotation6d_to_matrix(values: Tensor) -> Tensor:
    """Convert the first two body-to-NED rotation columns to SO(3)."""
    if values.shape[-1] != 6:
        raise ValueError("rotation6d must have six channels")
    first = functional.normalize(values[..., :3], dim=-1, eps=1e-6)
    second_raw = values[..., 3:]
    second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
    second = functional.normalize(second, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rotation6d(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3, 3]")
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def _skew(vector: Tensor) -> Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("rotation vector must have three channels")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def so3_exp(rotation_vector: Tensor) -> Tensor:
    """Differentiable Rodrigues exponential with a stable small-angle branch."""
    theta_squared = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = theta_squared.sqrt()
    # sinc(x) is sin(pi*x)/(pi*x), including its analytic value at zero.
    # The half-angle identity avoids both the explicit small-angle branch and
    # cancellation in (1-cos(theta))/theta^2.
    sine_scale = torch.sinc(theta / math.pi)
    half_angle_sinc = torch.sinc(theta / (2.0 * math.pi))
    cosine_scale = 0.5 * half_angle_sinc.square()
    skew = _skew(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    identity = identity.expand(*rotation_vector.shape[:-1], 3, 3)
    return (
        identity
        + sine_scale[..., None] * skew
        + cosine_scale[..., None] * (skew @ skew)
    )


def rotation_squared_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Squared geodesic approximation, equal to theta^2 near the identity."""
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return 2.0 * (1.0 - cosine)


def _wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotation_roll_pitch_yaw(rotation: Tensor) -> Tensor:
    """Return intrinsic roll, pitch, yaw for a body-to-NED rotation."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in [3, 3]")
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.asin(rotation[..., 2, 0].neg().clamp(-1.0, 1.0))
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    return torch.stack((roll, pitch, yaw), dim=-1)


@dataclass(frozen=True)
class Stage2Config:
    probe_hidden_dim: int = 128
    # The fixed branch is a deliberately small, interpretable prior.  Its
    # steady-state gains and time constants are the calibrated nominal
    # BlueROV2 values used by the collection NMPC; the learned branch only
    # explains the remaining domain/context-dependent wrench.
    nominal_effective_mass_kg: tuple[float, ...] = (
        11.189,
        11.189,
        11.189,
        0.45,
        0.45,
        0.80,
    )
    # Calibrated from the 5 Hz NMPC training split.  In particular, the old
    # yaw prior (7.1 gain / 0.54 s) was roughly twice as fast as the plant and
    # forced the learned action residual to cancel, then invert, yaw response.
    nominal_time_constant_seconds: tuple[float, ...] = (
        2.32,
        1.98,
        3.13,
        1.5,
        1.5,
        1.13,
    )
    nominal_steady_rate_gain: tuple[float, ...] = (0.404, 0.427, 0.492, 0.0, 0.0, 3.66)
    nominal_quadratic_damping: tuple[float, ...] = (0.80, 0.80, 0.80, 0.05, 0.05, 0.20)
    nominal_buoyancy_force_n: float = 0.0
    # The heave bias must cover the collection bank's 0--4 N net-buoyancy
    # range.  The remaining limits are deliberately conservative.
    residual_wrench_limit: tuple[float, ...] = (2.0, 2.0, 4.5, 0.30, 0.30, 4.0)
    residual_magnitude_loss_weight: float = 0.01
    residual_smoothness_loss_weight: float = 0.02
    # Stage-2 is a factual physical decoder. Candidate-action replacements are
    # diagnostics only because the dataset contains no same-state
    # counterfactual outcomes.
    # VINS velocity finite differences are noisy. They constrain local
    # acceleration only as an auxiliary; integrated observable rollouts remain
    # the primary objective.
    rate_loss_weight: float = 0.25
    near_horizon_rate_loss_weight: float = 0.50
    near_horizon_rate_seconds: float = 0.50
    actuator_time_constant_seconds: float = 0.10
    # A bounded 6x5 action-dependent correction captures per-thruster gain
    # residuals and cross-axis coupling while preserving the nominal action
    # path.  The matrix is row-normalized so each output wrench correction is
    # bounded by the corresponding value below for unit modal commands.
    residual_action_wrench_limit: tuple[float, ...] = (
        0.50,
        0.50,
        0.75,
        0.10,
        0.10,
        0.80,
    )
    # History context may adapt a small set of physical coefficients, but each
    # multiplicative correction is bounded around the calibrated nominal model.
    # The action residual is separately bounded so it cannot overturn a
    # commanded axis' nominal response sign.
    domain_mass_log_scale_limit: float = math.log(1.25)
    domain_control_log_scale_limit: float = math.log(1.40)
    domain_damping_log_scale_limit: float = math.log(1.50)
    domain_actuator_log_scale_limit: float = math.log(2.0)
    domain_parameter_loss_weight: float = 0.01
    acceleration_error_scale_m_s2: tuple[float, float, float] = (0.50, 0.50, 0.20)
    angular_acceleration_error_scale_rad_s2: tuple[float, float, float] = (
        0.20,
        0.20,
        0.50,
    )
    tilt_error_scale_rad: float = 0.10
    yaw_error_scale_rad: float = 0.20
    velocity_error_scale_m_s: tuple[float, float, float] = (0.10, 0.10, 0.10)
    roll_pitch_rate_error_scale_rad_s: tuple[float, float] = (0.05, 0.05)
    yaw_rate_error_scale_rad_s: float = 0.10
    # Depth and IMU acceleration remain diagnostic scales, not primary losses.
    depth_error_scale_m: float = 0.15
    position_error_scale_m: tuple[float, float, float] = (0.30, 0.30, 0.20)
    tilt_loss_weight: float = 0.1
    yaw_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    roll_pitch_rate_loss_weight: float = 0.1
    yaw_rate_loss_weight: float = 1.0
    position_loss_weight: float = 2.0
    terminal_trajectory_loss_weight: float = 0.10
    # Retained only so old context-only checkpoints deserialize. The delayed
    # latent-wrench and context-replacement experiments were removed.
    future_latent_physics_enabled: bool = False
    future_latent_context_enabled: bool = False
    # Decode each cumulative action-conditioned latent response z_hat[k]-z_t
    # into a bounded physical response potential. Its temporal increment is the
    # only quantity injected into the physical rate, so integrating the latent
    # branch through step k yields exactly that step's response potential.
    future_latent_response_enabled: bool = False
    latent_response_feature: str = "response_potential"
    # This is a physical rate limit. The response-potential limit is dt times
    # this value, preventing the temporal difference from adding a hidden 1/dt
    # gain at the first imagined step.
    latent_response_rate_limit: tuple[float, ...] = (
        0.25,
        0.25,
        0.35,
        0.15,
        0.15,
        0.75,
    )
    latent_response_magnitude_loss_weight: float = 0.002
    latent_response_smoothness_loss_weight: float = 0.005
    latent_loss_weight: float = 0.10
    trajectory_loss_weight: float = 1.0
    near_horizon_weight: float = 2.0
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        triples = (
            self.acceleration_error_scale_m_s2,
            self.angular_acceleration_error_scale_rad_s2,
            self.velocity_error_scale_m_s,
            self.position_error_scale_m,
        )
        if (
            self.probe_hidden_dim < 1
            or any(len(values) != 3 for values in triples)
            or len(self.roll_pitch_rate_error_scale_rad_s) != 2
            or len(self.latent_response_rate_limit) != 6
        ):
            raise ValueError("invalid Stage-2 dimensions")
        if any(
            len(values) != 6
            for values in (
                self.nominal_effective_mass_kg,
                self.nominal_time_constant_seconds,
                self.nominal_steady_rate_gain,
                self.nominal_quadratic_damping,
                self.residual_wrench_limit,
                self.residual_action_wrench_limit,
            )
        ):
            raise ValueError(
                "nominal and residual wrench vectors must contain six values"
            )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in (
                *self.nominal_effective_mass_kg,
                *self.nominal_time_constant_seconds,
            )
        ):
            raise ValueError(
                "nominal masses and time constants must be positive and finite"
            )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in (
                *self.nominal_steady_rate_gain,
                *self.nominal_quadratic_damping,
                *self.residual_wrench_limit,
            )
        ):
            raise ValueError(
                "nominal gains, damping, and residual limits must be finite and non-negative"
            )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.residual_action_wrench_limit
        ):
            raise ValueError(
                "action residual wrench limits must be finite and non-negative"
            )
        if not math.isfinite(self.nominal_buoyancy_force_n):
            raise ValueError("nominal buoyancy force must be finite")
        scalars = (
            *self.acceleration_error_scale_m_s2,
            *self.angular_acceleration_error_scale_rad_s2,
            self.tilt_error_scale_rad,
            self.yaw_error_scale_rad,
            *self.velocity_error_scale_m_s,
            *self.roll_pitch_rate_error_scale_rad_s,
            self.yaw_rate_error_scale_rad_s,
            self.depth_error_scale_m,
            *self.position_error_scale_m,
            self.tilt_loss_weight,
            self.yaw_loss_weight,
            self.velocity_loss_weight,
            self.roll_pitch_rate_loss_weight,
            self.yaw_rate_loss_weight,
            self.position_loss_weight,
            self.trajectory_loss_weight,
            self.near_horizon_weight,
            self.learning_rate,
            self.grad_clip_norm,
            self.residual_magnitude_loss_weight,
            self.residual_smoothness_loss_weight,
            self.rate_loss_weight,
            self.near_horizon_rate_loss_weight,
            self.near_horizon_rate_seconds,
            self.actuator_time_constant_seconds,
            self.domain_mass_log_scale_limit,
            self.domain_control_log_scale_limit,
            self.domain_damping_log_scale_limit,
            self.domain_actuator_log_scale_limit,
            self.domain_parameter_loss_weight,
            self.terminal_trajectory_loss_weight,
            *self.latent_response_rate_limit,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in scalars
        ):
            raise ValueError(
                "Stage-2 physical scales and primary weights must be positive"
            )
        if not math.isfinite(self.latent_loss_weight) or self.latent_loss_weight < 0.0:
            raise ValueError(
                "Stage-2 latent diagnostic weight must be finite and non-negative"
            )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.latent_response_magnitude_loss_weight,
                self.latent_response_smoothness_loss_weight,
            )
        ):
            raise ValueError(
                "latent response regularization weights must be finite and non-negative"
            )
        if self.weight_decay < 0.0:
            raise ValueError("Stage-2 weight decay must be non-negative")
        if self.near_horizon_weight < 1.0:
            raise ValueError("near-horizon weight must be at least one")
        if not all(
            isinstance(value, bool)
            for value in (
                self.future_latent_physics_enabled,
                self.future_latent_context_enabled,
                self.future_latent_response_enabled,
            )
        ):
            raise ValueError("future latent mode flags must be boolean")
        if self.future_latent_physics_enabled or self.future_latent_context_enabled:
            raise ValueError(
                "legacy delayed future-latent modes were removed; "
                "use future_latent_response_enabled"
            )
        if self.latent_response_feature not in LATENT_RESPONSE_FEATURES:
            raise ValueError(
                "latent_response_feature must be one of " f"{LATENT_RESPONSE_FEATURES}"
            )


class NominalUnderwaterDynamics(nn.Module):
    """Fixed six-DOF underwater prior used by the Stage-2 residual probe.

    In body coordinates the model is

    ``M nu_dot = tau_control - C(nu) nu - D1 nu - D2 |nu| nu + tau_b``.

    The default control wrench is chosen from the calibrated nominal
    steady-state rate gains and time constants, so the prior remains tied to
    the collection plant rather than an arbitrary network scale.  All
    parameters are buffers: only the residual network is learned.
    """

    def __init__(self, config: Stage2Config) -> None:
        super().__init__()
        mass = torch.as_tensor(config.nominal_effective_mass_kg, dtype=torch.float32)
        tau = torch.as_tensor(config.nominal_time_constant_seconds, dtype=torch.float32)
        gain = torch.as_tensor(config.nominal_steady_rate_gain, dtype=torch.float32)
        quadratic = torch.as_tensor(
            config.nominal_quadratic_damping, dtype=torch.float32
        )
        # The body +z force direction is down, while the MANUAL heave/modal
        # convention is positive up.  The allocation matrix carries that sign.
        allocation = torch.as_tensor(
            (
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            ),
            dtype=torch.float32,
        )
        self.register_buffer("effective_mass", mass)
        self.register_buffer("linear_damping", mass / tau)
        self.register_buffer("quadratic_damping", quadratic)
        self.register_buffer("control_wrench_per_action", mass * gain / tau)
        self.register_buffer("control_allocation", allocation)
        self.register_buffer(
            "buoyancy_wrench",
            torch.as_tensor(
                (0.0, 0.0, -config.nominal_buoyancy_force_n, 0.0, 0.0, 0.0),
                dtype=torch.float32,
            ),
        )

    def prepare_coefficients(
        self,
        reference: Tensor,
        domain_parameters: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Prepare domain-scaled coefficients shared by every rollout step."""
        batch_size = reference.shape[0]
        if domain_parameters is None:
            return {
                "mass": self.effective_mass.expand(batch_size, -1),
                "control_wrench_per_action": (
                    self.control_wrench_per_action.expand(batch_size, -1)
                ),
                "linear_damping": self.linear_damping.expand(batch_size, -1),
                "quadratic_damping": self.quadratic_damping.expand(batch_size, -1),
            }
        required = ("mass_scale", "control_scale", "damping_scale")
        missing = [name for name in required if name not in domain_parameters]
        if missing:
            raise ValueError(f"domain parameters are missing fields: {missing}")
        mass_scale = domain_parameters["mass_scale"]
        control_scale = domain_parameters["control_scale"]
        damping_scale = domain_parameters["damping_scale"]
        expected = (batch_size, 6)
        if any(
            value.shape != expected
            for value in (mass_scale, control_scale, damping_scale)
        ):
            raise ValueError("domain parameter scales must be [batch, 6]")
        return {
            "mass": self.effective_mass * mass_scale,
            "control_wrench_per_action": (
                self.control_wrench_per_action * control_scale
            ),
            "linear_damping": self.linear_damping * damping_scale,
            "quadratic_damping": self.quadratic_damping * damping_scale,
        }

    def forward(
        self,
        velocity: Tensor,
        omega: Tensor,
        physical_action: Tensor,
        rotation: Tensor,
        domain_parameters: Mapping[str, Tensor] | None = None,
        *,
        prepared_coefficients: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if velocity.shape[-1] != 3 or omega.shape[-1] != 3:
            raise ValueError("nominal dynamics expects 3D velocity and angular rate")
        if physical_action.shape[-1] != len(ACTION_LABELS):
            raise ValueError("physical action channel count differs from the mixer")
        if rotation.shape[-2:] != (3, 3):
            raise ValueError("nominal dynamics expects an SO(3) rotation")
        motion = torch.cat((velocity, omega), dim=-1)
        coefficients = prepared_coefficients
        if coefficients is None:
            coefficients = self.prepare_coefficients(velocity, domain_parameters)
        required_coefficients = (
            "mass",
            "control_wrench_per_action",
            "linear_damping",
            "quadratic_damping",
        )
        missing_coefficients = [
            name for name in required_coefficients if name not in coefficients
        ]
        if missing_coefficients:
            raise ValueError(
                f"prepared nominal coefficients are missing: {missing_coefficients}"
            )
        expected = (velocity.shape[0], 6)
        if any(coefficients[name].shape != expected for name in required_coefficients):
            raise ValueError("prepared nominal coefficients must be [batch, 6]")
        mass = coefficients["mass"]
        action_modal = physical_action @ self.control_allocation
        control_wrench = action_modal * coefficients["control_wrench_per_action"]
        linear_momentum = velocity * mass[:, :3]
        angular_momentum = omega * mass[:, 3:]
        coriolis = torch.cat(
            (
                torch.cross(omega, linear_momentum, dim=-1),
                torch.cross(omega, angular_momentum, dim=-1),
            ),
            dim=-1,
        )
        damping = (
            coefficients["linear_damping"] * motion
            + coefficients["quadratic_damping"] * motion.abs() * motion
        )
        # A constant NED buoyancy force is rotated into the body frame.  The
        # nominal benchmark defaults to neutral buoyancy; a nonzero value is a
        # soft prior and its mismatch is learned by the residual branch.
        buoyancy_body = (
            rotation.transpose(-1, -2)
            @ self.buoyancy_wrench[:3].expand(rotation.shape[0], -1).unsqueeze(-1)
        ).squeeze(-1)
        restoring = torch.cat(
            (buoyancy_body, self.buoyancy_wrench[3:].expand(rotation.shape[0], -1)),
            dim=-1,
        )
        wrench = control_wrench - coriolis - damping + restoring
        rate = wrench / mass
        return rate, wrench


class PhysicsRateProbe(nn.Module):
    """Nominal dynamics plus bounded history-conditioned residual wrenches.

    The Stage-1 context summarizes only observed history and is fixed over an
    imagined horizon. Recurrent physical state and lagged candidate action are
    the causal transition inputs. The optional latent branch decodes a bounded
    cumulative response potential. Context and state can interpret a nonzero
    latent response, but cannot create a latent correction by themselves.
    """

    def __init__(
        self,
        context_dim: int,
        belief_dim: int,
        config: Stage2Config,
    ) -> None:
        super().__init__()
        if context_dim < 1:
            raise ValueError("context dimension must be positive")
        hidden = config.probe_hidden_dim
        self.context_dim = context_dim
        self.action_dim = len(ACTION_LABELS)
        # Bias is action-independent at a physical transition.  The allocation
        # correction additionally sees effective action and command innovation.
        state_input_dim = context_dim + 6
        self.nominal = NominalUnderwaterDynamics(config)
        self.register_buffer(
            "residual_wrench_limit",
            torch.as_tensor(config.residual_wrench_limit, dtype=torch.float32),
        )
        self.register_buffer(
            "residual_action_wrench_limit",
            torch.as_tensor(config.residual_action_wrench_limit, dtype=torch.float32),
        )
        self.register_buffer(
            "domain_log_scale_limits",
            torch.as_tensor(
                (
                    config.domain_mass_log_scale_limit,
                    config.domain_control_log_scale_limit,
                    config.domain_damping_log_scale_limit,
                    config.domain_actuator_log_scale_limit,
                ),
                dtype=torch.float32,
            ),
        )
        if belief_dim < 1 or belief_dim >= context_dim:
            raise ValueError("Stage-2 belief width must be inside the full context")
        self.belief_dim = int(belief_dim)
        self.state_dim = context_dim - self.belief_dim
        self.future_latent_response_enabled = config.future_latent_response_enabled
        self.latent_response_feature = config.latent_response_feature
        # Six effective-mass, six control-gain, six damping, and five modal
        # actuator-lag corrections. These are dynamics-context quantities, so
        # they read the history belief only, not the current state embedding.
        self.domain_parameter_network = nn.Sequential(
            nn.Linear(self.belief_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6 + 6 + 6 + self.action_dim),
        )
        self.action_residual_network = nn.Sequential(
            nn.Linear(self.belief_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6 * self.action_dim),
        )
        self.drift_network = nn.Sequential(
            nn.Linear(state_input_dim + 6, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        self.latent_response_encoder: nn.Sequential | None
        self.latent_response_output: nn.Linear | None
        # These modules only exist when an older response checkpoint is loaded.
        self.latent_response_conditioning: nn.Sequential | None
        self.latent_response_projection: nn.Linear | None
        self.latent_response_direct: nn.Linear | None
        if self.future_latent_response_enabled:
            response_rank = min(32, hidden)
            if self.latent_response_feature == "response_potential":
                response_conditioning_dim = context_dim + 6 + 6 + 2 * self.action_dim
                self.latent_response_encoder = nn.Sequential(
                    nn.Linear(self.state_dim, hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden, response_rank, bias=False),
                )
                self.latent_response_conditioning = nn.Sequential(
                    nn.Linear(response_conditioning_dim, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, response_rank),
                )
                self.latent_response_projection = None
                self.latent_response_direct = nn.Linear(
                    response_rank, len(CONTROLLED_RATE_INDICES), bias=False
                )
            else:
                response_conditioning_dim = context_dim + 6 + 6 + 2 * self.action_dim
                self.latent_response_encoder = None
                self.latent_response_conditioning = nn.Sequential(
                    nn.Linear(response_conditioning_dim, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, response_rank),
                )
                self.latent_response_projection = nn.Linear(
                    self.state_dim, response_rank, bias=False
                )
                self.latent_response_direct = nn.Linear(self.state_dim, 6, bias=False)
                nn.init.zeros_(self.latent_response_direct.weight)
            output_dim = (
                len(CONTROLLED_RATE_INDICES)
                if self.latent_response_feature == "response_potential"
                else 6
            )
            self.latent_response_output = nn.Linear(
                response_rank, output_dim, bias=False
            )
            if self.latent_response_direct is not None:
                nn.init.zeros_(self.latent_response_direct.weight)
            nn.init.zeros_(self.latent_response_output.weight)
        else:
            self.latent_response_encoder = None
            self.latent_response_conditioning = None
            self.latent_response_projection = None
            self.latent_response_direct = None
            self.latent_response_output = None
        self.register_buffer(
            "latent_response_rate_limit",
            torch.as_tensor(config.latent_response_rate_limit, dtype=torch.float32),
            persistent=False,
        )
        finals = (
            self.domain_parameter_network[-1],
            self.action_residual_network[-1],
            self.drift_network[-1],
        )
        if not all(isinstance(final, nn.Linear) for final in finals):
            raise TypeError("probe final layer must be linear")
        for final in finals:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def neutral_domain_parameters(self, context: Tensor) -> dict[str, Tensor]:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("dynamics context has the wrong shape")
        batch_size = context.shape[0]
        return {
            "mass_scale": context.new_ones(batch_size, 6),
            "control_scale": context.new_ones(batch_size, 6),
            "damping_scale": context.new_ones(batch_size, 6),
            "actuator_time_constant_scale": context.new_ones(
                batch_size, self.action_dim
            ),
            "normalized_deviation": context.new_zeros(
                batch_size, 6 + 6 + 6 + self.action_dim
            ),
        }

    def domain_parameters(self, context: Tensor) -> dict[str, Tensor]:
        """Decode bounded context corrections without claiming identification."""
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("dynamics context has the wrong shape")
        belief = context[:, self.state_dim :]
        normalized = torch.tanh(self.domain_parameter_network(belief))
        mass, control, damping, actuator = torch.split(
            normalized, (6, 6, 6, self.action_dim), dim=-1
        )
        limits = self.domain_log_scale_limits
        return {
            "mass_scale": torch.exp(limits[0] * mass),
            "control_scale": torch.exp(limits[1] * control),
            "damping_scale": torch.exp(limits[2] * damping),
            "actuator_time_constant_scale": torch.exp(limits[3] * actuator),
            "normalized_deviation": normalized,
        }

    def _features(
        self,
        context: Tensor,
        normalized_motion_state: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        physical_effective_action: Tensor,
        rotation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("dynamics context has the wrong shape")
        tensors = (
            normalized_motion_state,
            normalized_effective_action,
            normalized_action_innovation,
        )
        if any(
            value.ndim != 2 or value.shape[0] != context.shape[0] for value in tensors
        ):
            raise ValueError(
                "dynamics state and action inputs must be aligned matrices"
            )
        if normalized_motion_state.shape[-1] != 6:
            raise ValueError("normalized motion state must contain velocity and omega")
        if (
            any(value.shape[-1] != self.action_dim for value in tensors[1:])
            or physical_effective_action.ndim != 2
            or physical_effective_action.shape[0] != context.shape[0]
            or physical_effective_action.shape[-1] != self.action_dim
        ):
            raise ValueError(
                "normalized modal action input has the wrong channel count"
            )
        if rotation is None:
            rotation_features = normalized_motion_state.new_zeros(
                normalized_motion_state.shape[0], 6
            )
        else:
            if rotation.shape != (context.shape[0], 3, 3):
                raise ValueError("residual rotation must be [batch, 3, 3]")
            rotation_features = matrix_to_rotation6d(rotation)
        state_features = torch.cat(
            (context, normalized_motion_state, rotation_features), dim=-1
        )
        return state_features, context[:, self.state_dim :]

    def residual_wrench(
        self,
        context: Tensor,
        normalized_motion_state: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        physical_effective_action: Tensor,
        rotation: Tensor | None = None,
        action_residual_matrix: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state_features, belief = self._features(
            context,
            normalized_motion_state,
            normalized_effective_action,
            normalized_action_innovation,
            physical_effective_action,
            rotation,
        )
        drift_logits = self.drift_network(state_features)
        bias_wrench = torch.tanh(drift_logits) * self.residual_wrench_limit
        matrix = action_residual_matrix
        if matrix is None:
            matrix = self.action_residual_matrix(context)
        elif matrix.shape != (context.shape[0], 6, self.action_dim):
            raise ValueError("action residual matrix must be [batch, 6, action]")
        bounded_action = physical_effective_action.clamp(-1.0, 1.0)
        action_wrench = torch.bmm(matrix, bounded_action.unsqueeze(-1)).squeeze(-1)
        return bias_wrench + action_wrench, bias_wrench, action_wrench

    def _latent_response_condition(
        self,
        context: Tensor,
        normalized_motion_state: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        rotation: Tensor,
    ) -> Tensor:
        return torch.cat(
            (
                context,
                normalized_motion_state,
                matrix_to_rotation6d(rotation),
                normalized_effective_action,
                normalized_action_innovation,
            ),
            dim=-1,
        )

    def latent_response_potential(
        self,
        context: Tensor,
        normalized_motion_state: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        rotation: Tensor,
        current_latent: Tensor | None,
        future_latent: Tensor | None,
    ) -> Tensor:
        """Decode cumulative candidate latent response into a physical potential."""
        if not self.future_latent_response_enabled:
            return normalized_motion_state.new_zeros(
                normalized_motion_state.shape[0], 6
            )
        if self.latent_response_feature != "response_potential":
            raise RuntimeError("response potential requested from a legacy decoder")
        expected_shape = (context.shape[0], self.state_dim)
        if future_latent is None or future_latent.shape != expected_shape:
            raise ValueError("latent response decoder requires same-step z_hat")
        if current_latent is None or current_latent.shape != expected_shape:
            raise ValueError("latent response decoder requires current z_t")
        modules = (
            self.latent_response_encoder,
            self.latent_response_conditioning,
            self.latent_response_direct,
            self.latent_response_output,
        )
        if any(module is None for module in modules):
            raise RuntimeError("latent response-potential decoder is incomplete")
        latent_feature = self.latent_response_encoder(future_latent - current_latent)
        modulation = torch.tanh(
            self.latent_response_conditioning(
                self._latent_response_condition(
                    context,
                    normalized_motion_state,
                    normalized_effective_action,
                    normalized_action_innovation,
                    rotation,
                )
            )
        )
        modal_logits = self.latent_response_direct(
            latent_feature
        ) + self.latent_response_output(latent_feature * modulation)
        modal_response = torch.tanh(modal_logits)
        return torch.cat(
            (
                modal_response[:, :3],
                modal_response.new_zeros(modal_response.shape[0], 2),
                modal_response[:, 3:],
            ),
            dim=-1,
        )

    def legacy_latent_response_rate(
        self,
        context: Tensor,
        normalized_motion_state: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        rotation: Tensor,
        future_latent: Tensor | None,
    ) -> Tensor:
        """Preserve the exact decoder used by historical response checkpoints."""
        if self.latent_response_feature == "response_potential":
            raise RuntimeError(
                "legacy rate requested from a response-potential decoder"
            )
        expected_shape = (context.shape[0], self.state_dim)
        if future_latent is None or future_latent.shape != expected_shape:
            raise ValueError("legacy latent response requires same-step z_hat")

        modules = (
            self.latent_response_conditioning,
            self.latent_response_projection,
            self.latent_response_direct,
            self.latent_response_output,
        )
        if any(module is None for module in modules):
            raise RuntimeError("legacy latent response decoder is incomplete")
        latent_delta = future_latent - context[:, : self.state_dim]
        conditioning = self._latent_response_condition(
            context,
            normalized_motion_state,
            normalized_effective_action,
            normalized_action_innovation,
            rotation,
        )
        gate = torch.tanh(self.latent_response_conditioning(conditioning))
        projected = self.latent_response_projection(latent_delta)
        logits = self.latent_response_direct(latent_delta) + (
            self.latent_response_output(projected * gate)
        )
        return torch.tanh(logits) * self.latent_response_rate_limit

    def action_residual_matrix(self, context: Tensor) -> Tensor:
        """Decode the history-conditioned action allocation once per rollout."""
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError("dynamics context has the wrong shape")
        belief = context[:, self.state_dim :]
        matrix = torch.tanh(self.action_residual_network(belief)).reshape(
            context.shape[0], 6, self.action_dim
        )
        row_limit = self.residual_action_wrench_limit[:, None] / float(self.action_dim)
        return matrix * row_limit

    def forward(
        self,
        context: Tensor,
        normalized_motion_state: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        physical_effective_action: Tensor,
        rotation: Tensor | None = None,
    ) -> Tensor:
        """Return the residual-rate branch for the normalized probe API.

        Full nominal-plus-residual rates are produced by
        :meth:`forward_with_components`, which receives the unnormalized
        velocity and angular rate required by the hydrodynamic equation.
        Keeping this method residual-only avoids silently treating standardized
        network inputs as physical SI units for callers that inspect the probe.
        """
        residual_wrench, _, _ = self.residual_wrench(
            context,
            normalized_motion_state,
            normalized_effective_action,
            normalized_action_innovation,
            physical_effective_action,
            rotation,
        )
        residual_rate = residual_wrench / self.nominal.effective_mass
        return residual_rate

    def predict_rate(
        self,
        context: Tensor,
        velocity: Tensor,
        omega: Tensor,
        normalized_motion_state: Tensor,
        physical_action: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        physical_effective_action: Tensor,
        rotation: Tensor,
        domain_parameters: Mapping[str, Tensor],
        *,
        include_residual: bool = True,
        action_residual_matrix: Tensor | None = None,
        nominal_coefficients: Mapping[str, Tensor] | None = None,
        latent_response_rate: Tensor | None = None,
    ) -> Tensor:
        """Return only the rate needed by online control rollout."""
        fixed_nominal_rate, _ = self.nominal(
            velocity,
            omega,
            physical_action,
            rotation,
        )
        context_adapted_rate, _ = self.nominal(
            velocity,
            omega,
            physical_action,
            rotation,
            domain_parameters=domain_parameters,
            prepared_coefficients=nominal_coefficients,
        )
        if not include_residual:
            return fixed_nominal_rate
        residual_wrench, _, _ = self.residual_wrench(
            context,
            normalized_motion_state,
            normalized_effective_action,
            normalized_action_innovation,
            physical_effective_action,
            rotation,
            action_residual_matrix,
        )
        effective_mass = (
            nominal_coefficients["mass"]
            if nominal_coefficients is not None
            else self.nominal.effective_mass * domain_parameters["mass_scale"]
        )
        context_increment_rate = (
            context_adapted_rate - fixed_nominal_rate
        ) + residual_wrench / effective_mass
        if latent_response_rate is None:
            latent_response_rate = torch.zeros_like(fixed_nominal_rate)
        elif latent_response_rate.shape != fixed_nominal_rate.shape:
            raise ValueError("latent response rate must be [batch, 6]")
        return fixed_nominal_rate + context_increment_rate + latent_response_rate

    def forward_with_components(
        self,
        context: Tensor,
        velocity: Tensor,
        omega: Tensor,
        normalized_motion_state: Tensor,
        physical_action: Tensor,
        normalized_effective_action: Tensor,
        normalized_action_innovation: Tensor,
        physical_effective_action: Tensor,
        rotation: Tensor,
        domain_parameters: Mapping[str, Tensor] | None = None,
        *,
        include_residual: bool = True,
        action_residual_matrix: Tensor | None = None,
        nominal_coefficients: Mapping[str, Tensor] | None = None,
        latent_response_rate: Tensor | None = None,
    ) -> dict[str, Tensor]:
        fixed_nominal_rate, fixed_nominal_wrench = self.nominal(
            velocity,
            omega,
            physical_action,
            rotation,
        )
        context_adapted_rate, context_adapted_wrench = self.nominal(
            velocity,
            omega,
            physical_action,
            rotation,
            domain_parameters=domain_parameters,
            prepared_coefficients=nominal_coefficients,
        )
        if include_residual:
            (
                residual_wrench,
                residual_bias_wrench,
                residual_action_wrench,
            ) = self.residual_wrench(
                context,
                normalized_motion_state,
                normalized_effective_action,
                normalized_action_innovation,
                physical_effective_action,
                rotation,
                action_residual_matrix,
            )
        else:
            residual_wrench = torch.zeros_like(context_adapted_wrench)
            residual_bias_wrench = torch.zeros_like(context_adapted_wrench)
            residual_action_wrench = torch.zeros_like(context_adapted_wrench)
        effective_mass = (
            nominal_coefficients["mass"]
            if nominal_coefficients is not None
            else self.nominal.effective_mass
        )
        if nominal_coefficients is None and domain_parameters is not None:
            effective_mass = effective_mass * domain_parameters["mass_scale"]
        residual_rate = residual_wrench / effective_mass
        context_increment_rate = (
            context_adapted_rate - fixed_nominal_rate
        ) + residual_rate
        if latent_response_rate is None:
            latent_response_rate = torch.zeros_like(fixed_nominal_rate)
        elif latent_response_rate.shape != fixed_nominal_rate.shape:
            raise ValueError("latent response rate must be [batch, 6]")
        return {
            "predicted_rate_body": (
                fixed_nominal_rate + context_increment_rate + latent_response_rate
            ),
            # Keep the fixed calibrated prior distinct from the learned,
            # context-conditioned coefficient correction. This makes the
            # nominal ablation and learned action derivative unambiguous.
            "fixed_nominal_rate_body": fixed_nominal_rate,
            "context_adapted_rate_body": context_adapted_rate,
            "domain_rate_correction_body": (context_adapted_rate - fixed_nominal_rate),
            "context_increment_rate_body": context_increment_rate,
            "latent_response_rate_body": latent_response_rate,
            "nominal_rate_body": fixed_nominal_rate,
            "residual_rate_body": residual_rate,
            "fixed_nominal_wrench": fixed_nominal_wrench,
            "context_adapted_wrench": context_adapted_wrench,
            "nominal_wrench": fixed_nominal_wrench,
            "residual_wrench": residual_wrench,
            "residual_bias_wrench": residual_bias_wrench,
            "residual_action_wrench": residual_action_wrench,
        }


class PhysicsIntegratedStage2(nn.Module):
    """Decode physical dynamics while preserving the Stage-1 representation.

    The physical decoder is intentionally not a second unconstrained world
    model: all Stage-1 parameters are fixed, the nominal hydrodynamic branch
    is fixed, and the probe learns bounded context/action residuals on top.
    """

    def __init__(
        self,
        backbone: PredictiveStateJEPA,
        standardizer: Standardizer,
        *,
        sample_rate_hz: float,
        config: Stage2Config = Stage2Config(),
        representation_training_mode: str = "predictive_pretrained_frozen",
    ) -> None:
        super().__init__()
        if sample_rate_hz <= 0.0 or not math.isfinite(sample_rate_hz):
            raise ValueError("sample rate must be positive and finite")
        if representation_training_mode not in REPRESENTATION_TRAINING_MODES:
            raise ValueError(
                "representation training mode must be one of "
                f"{REPRESENTATION_TRAINING_MODES}"
            )
        self.backbone = backbone
        self.config = config
        self.sample_rate_hz = float(sample_rate_hz)
        self.representation_training_mode = representation_training_mode
        self.probe = PhysicsRateProbe(
            backbone.context_dim,
            backbone.config.latent_dim,
            config,
        )
        self.register_buffer(
            "observation_mean",
            torch.as_tensor(standardizer.observation_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "observation_std",
            torch.as_tensor(standardizer.observation_std, dtype=torch.float32),
        )
        self.register_buffer(
            "action_mean",
            torch.as_tensor(standardizer.action_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "action_std",
            torch.as_tensor(standardizer.action_std, dtype=torch.float32),
        )
        self.configure_trainable_backbone()

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    def configure_trainable_backbone(self) -> None:
        # The response mode consumes the pretrained Stage-1 prediction as a
        # fixed representation. This prevents downstream physical supervision
        # from redefining z_hat to become an arbitrary Stage-2 hidden feature.
        # Retain the established transition fine-tuning behavior for legacy
        # modes and their matched context-only baselines.
        self.backbone.requires_grad_(False)
        if not (
            self.representation_training_mode == "predictive_pretrained_frozen"
            and self.config.future_latent_response_enabled
        ):
            for module in self._trainable_transition_modules():
                module.requires_grad_(True)
        if self.representation_training_mode != "predictive_pretrained_frozen":
            for module in self._supervised_representation_modules():
                module.requires_grad_(True)
        self.backbone.dynamics_context_encoder.eval()
        self.backbone.dynamics_context_normalization.eval()
        self.backbone.observation_encoder.eval()
        self.backbone.target_observation_encoder.eval()

    def _trainable_transition_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.backbone.action_embedding,
            self.backbone.transition_conditioning,
            self.backbone.transition,
            self.backbone.transition_normalization,
        )

    def _supervised_representation_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.backbone.observation_encoder,
            self.backbone.dynamics_context_encoder,
            self.backbone.dynamics_context_normalization,
        )

    def train(self, mode: bool = True) -> "PhysicsIntegratedStage2":
        super().train(mode)
        self.backbone.eval()
        if self.representation_training_mode != "predictive_pretrained_frozen":
            for module in self._supervised_representation_modules():
                module.train(mode)
        else:
            for module in self._supervised_representation_modules():
                module.eval()
        self.backbone.target_observation_encoder.eval()
        return self

    @property
    def actuator_lag_alpha(self) -> float:
        """Exact first-order update coefficient at the Stage-2 model rate."""
        return 1.0 - math.exp(-self.dt / self.config.actuator_time_constant_seconds)

    def denormalize_observation(self, values: Tensor) -> Tensor:
        return values * self.observation_std + self.observation_mean

    def normalize_observation(self, values: Tensor) -> Tensor:
        return (values - self.observation_mean) / self.observation_std

    def normalize_action(self, values: Tensor) -> Tensor:
        return (values - self.action_mean) / self.action_std

    def denormalize_action(self, values: Tensor) -> Tensor:
        return values * self.action_std + self.action_mean

    def _reconstruct_effective_action(
        self,
        action_history: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        """Reconstruct the lagged actuator state from the available history.

        The first history sample is used as the unknown pre-window state; after
        a few actuator time constants its influence is negligible. This is
        more causal than assuming the last command was already applied.
        """
        physical_history = self.denormalize_action(action_history)
        history_steps = action_history.shape[1]
        if history_steps == 1:
            return physical_history[:, 0]
        decay = 1.0 - alpha
        exponents = torch.arange(
            history_steps - 2,
            -1,
            -1,
            dtype=alpha.dtype,
            device=alpha.device,
        ).view(1, -1, 1)
        tail_weights = alpha.unsqueeze(1) * decay.unsqueeze(1).pow(exponents)
        return decay.pow(history_steps - 1) * physical_history[:, 0] + (
            tail_weights * physical_history[:, 1:]
        ).sum(dim=1)

    def _integrate_dynamics(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        context: Tensor,
        physical_action: Tensor,
        predicted_latent: Tensor | None = None,
        *,
        include_residual: bool = True,
        minimal_output: bool = False,
    ) -> dict[str, Tensor]:
        if (
            observation_history.ndim != 3
            or action_history.ndim != 3
            or physical_action.ndim != 3
        ):
            raise ValueError("Stage-2 inputs must be [batch, time, channel]")
        if (
            action_history.shape[0] != observation_history.shape[0]
            or action_history.shape[1] < 1
            or physical_action.shape[0] != observation_history.shape[0]
            or physical_action.shape[1] < 1
        ):
            raise ValueError(
                "action history and observation history batches must align"
            )
        if action_history.shape[-1] != len(ACTION_LABELS) or physical_action.shape[
            -1
        ] != len(ACTION_LABELS):
            raise ValueError("action tensors have the wrong modal-action channel count")
        if context.shape != (
            observation_history.shape[0],
            self.backbone.context_dim,
        ):
            raise ValueError("Stage-2 dynamics context has the wrong shape")
        if predicted_latent is not None:
            expected_latent_shape = (
                observation_history.shape[0],
                physical_action.shape[1],
                self.backbone.config.latent_dim,
            )
            if predicted_latent.shape != expected_latent_shape:
                raise ValueError(
                    "action-conditioned latent rollout has the wrong shape"
                )
        if self.config.future_latent_response_enabled and predicted_latent is None:
            raise ValueError("latent-conditioned rollout requires predicted latent")
        current = self.denormalize_observation(observation_history[:, -1])

        velocity = current[:, VIO_VELOCITY_SLICE]
        omega = current[:, IMU_GYRO_SLICE]
        depth = current[:, DEPTH_INDEX : DEPTH_INDEX + 1]
        rotation = rotation6d_to_matrix(current[:, ROTATION_6D_SLICE])

        future_rotation = []
        future_velocity = []
        future_omega = []
        future_acceleration = []
        future_depth = []
        future_position_delta = []
        future_rate = []
        future_fixed_nominal_rate = []
        future_context_adapted_rate = []
        future_domain_rate_correction = []
        future_context_increment_rate = []
        future_residual_rate = []
        future_residual_wrench = []
        future_residual_bias_wrench = []
        future_residual_action_wrench = []
        future_latent_response_rate = []
        future_latent_response_potential = []
        # ``effective_modal_action`` is the actuator state at the end of the
        # interval.  The plant, however, experiences the exact interval mean
        # of the first-order actuator state.  Keep both values explicit so
        # diagnostics and controller code cannot confuse a terminal state
        # with the action that generated the interval response.
        future_effective_action = []
        future_interval_effective_action = []
        future_action_innovation = []
        domain_parameters = (
            self.probe.domain_parameters(context)
            if include_residual
            else self.probe.neutral_domain_parameters(context)
        )
        action_residual_matrix = (
            self.probe.action_residual_matrix(context) if include_residual else None
        )
        nominal_coefficients = self.probe.nominal.prepare_coefficients(
            current, domain_parameters
        )
        actuator_time_constant = (
            self.config.actuator_time_constant_seconds
            * domain_parameters["actuator_time_constant_scale"]
        )
        alpha = 1.0 - torch.exp(-self.dt / actuator_time_constant)
        # For e_dot=(u-e)/tau and a piecewise-constant command, the interval
        # average is e0 + beta*(u-e0), beta=1-(tau/dt)*(1-exp(-dt/tau)).
        # Evaluating the dynamics at the terminal state would apply the new
        # command too early and is exactly the one-step local-action bias that
        # showed up in the factual-vs-delayed audit.
        interval_alpha = (1.0 - (actuator_time_constant / self.dt) * alpha).clamp(
            0.0, 1.0
        )
        # Action history is standardized at the data-loader boundary. The
        # actuator state remains in physical modal-action units so that the
        # nominal allocation and learned correction share one convention.
        effective_action = self._reconstruct_effective_action(action_history, alpha)
        position_delta = torch.zeros(
            (len(current), 3), dtype=current.dtype, device=current.device
        )
        current_latent = context[:, : self.backbone.config.latent_dim]
        previous_latent_response_potential = current.new_zeros((len(current), 6))
        motion_mean = torch.cat(
            (
                self.observation_mean[VIO_VELOCITY_SLICE],
                self.observation_mean[IMU_GYRO_SLICE],
            )
        )
        motion_std = torch.cat(
            (
                self.observation_std[VIO_VELOCITY_SLICE],
                self.observation_std[IMU_GYRO_SLICE],
            )
        )
        for step in range(physical_action.shape[1]):
            commanded_action = physical_action[:, step]
            action_innovation = commanded_action - effective_action
            interval_effective_action = (
                effective_action + interval_alpha * action_innovation
            )
            effective_action = effective_action + alpha * action_innovation
            motion_state = torch.cat((velocity, omega), dim=-1)
            normalized_motion_state = (motion_state - motion_mean) / motion_std
            normalized_effective_action = self.normalize_action(
                interval_effective_action
            )
            normalized_action_innovation = action_innovation / self.action_std
            # History context and domain belief stay fixed for all candidates.
            # z_hat[k]-z_t is cumulative over the candidate prefix, so decode a
            # cumulative response potential and inject only its temporal change.
            step_latent = (
                predicted_latent[:, step]
                if self.config.future_latent_response_enabled
                else None
            )
            latent_response_rate = torch.zeros_like(motion_state)
            latent_response_potential = previous_latent_response_potential
            if (
                include_residual
                and self.config.future_latent_response_enabled
                and step_latent is not None
            ):
                if self.config.latent_response_feature == "response_potential":
                    latent_response_potential = self.probe.latent_response_potential(
                        context,
                        normalized_motion_state,
                        normalized_effective_action,
                        normalized_action_innovation,
                        rotation,
                        current_latent,
                        step_latent,
                    ) * (self.dt * self.probe.latent_response_rate_limit)
                    latent_response_rate = (
                        latent_response_potential - previous_latent_response_potential
                    ) / self.dt
                else:
                    latent_response_rate = self.probe.legacy_latent_response_rate(
                        context,
                        normalized_motion_state,
                        normalized_effective_action,
                        normalized_action_innovation,
                        rotation,
                        step_latent,
                    )
                    latent_response_potential = (
                        previous_latent_response_potential
                        + self.dt * latent_response_rate
                    )
            if minimal_output:
                predicted_rate = self.probe.predict_rate(
                    context,
                    velocity,
                    omega,
                    normalized_motion_state,
                    interval_effective_action,
                    normalized_effective_action,
                    normalized_action_innovation,
                    interval_effective_action,
                    rotation,
                    domain_parameters,
                    include_residual=include_residual,
                    action_residual_matrix=action_residual_matrix,
                    nominal_coefficients=nominal_coefficients,
                    latent_response_rate=latent_response_rate,
                )
            else:
                components = self.probe.forward_with_components(
                    context,
                    velocity,
                    omega,
                    normalized_motion_state,
                    interval_effective_action,
                    normalized_effective_action,
                    normalized_action_innovation,
                    interval_effective_action,
                    rotation,
                    domain_parameters,
                    include_residual=include_residual,
                    action_residual_matrix=action_residual_matrix,
                    nominal_coefficients=nominal_coefficients,
                    latent_response_rate=latent_response_rate,
                )
                predicted_rate = components["predicted_rate_body"]
            previous_latent_response_potential = latent_response_potential
            linear_rate = predicted_rate[:, :3]
            angular_rate = predicted_rate[:, 3:]
            velocity = velocity + self.dt * linear_rate
            omega = omega + self.dt * angular_rate
            rotation = rotation @ so3_exp(self.dt * omega)
            ned_velocity = (rotation @ velocity.unsqueeze(-1)).squeeze(-1)
            position_delta = position_delta + self.dt * ned_velocity
            depth = depth + self.dt * ned_velocity[:, 2:3]

            # The retained IMU contract records body translational
            # acceleration with gravity already absent in the GT/VIO proxy.
            # Keep the target causal and avoid a hidden gravity shortcut.
            body_acceleration = linear_rate + torch.cross(omega, velocity, dim=-1)
            if not minimal_output:
                future_rate.append(predicted_rate)
                future_fixed_nominal_rate.append(components["fixed_nominal_rate_body"])
                future_context_adapted_rate.append(
                    components["context_adapted_rate_body"]
                )
                future_domain_rate_correction.append(
                    components["domain_rate_correction_body"]
                )
                future_context_increment_rate.append(
                    components.get(
                        "context_increment_rate_body",
                        components["domain_rate_correction_body"]
                        + components["residual_rate_body"],
                    )
                )
                future_residual_rate.append(components["residual_rate_body"])
                future_residual_wrench.append(components["residual_wrench"])
                future_residual_bias_wrench.append(components["residual_bias_wrench"])
                future_residual_action_wrench.append(
                    components["residual_action_wrench"]
                )
                future_latent_response_rate.append(
                    components.get(
                        "latent_response_rate_body",
                        torch.zeros_like(components["predicted_rate_body"]),
                    )
                )
                future_latent_response_potential.append(latent_response_potential)
                future_effective_action.append(effective_action)
                future_interval_effective_action.append(interval_effective_action)
                future_action_innovation.append(action_innovation)
            future_rotation.append(matrix_to_rotation6d(rotation))
            future_velocity.append(velocity)
            future_omega.append(omega)
            future_acceleration.append(body_acceleration)
            future_depth.append(depth)
            future_position_delta.append(position_delta)
        physical_future = torch.cat(
            (
                torch.stack(future_rotation, dim=1),
                torch.stack(future_velocity, dim=1),
                torch.stack(future_omega, dim=1),
                torch.stack(future_acceleration, dim=1),
                torch.stack(future_depth, dim=1),
            ),
            dim=-1,
        )
        minimal = {
            "predicted_position_delta_ned_m": torch.stack(future_position_delta, dim=1),
            "predicted_future_physical_observation": physical_future,
        }
        if minimal_output:
            return minimal
        rate_rollout = torch.stack(future_rate, dim=1)
        return {
            **minimal,
            "predicted_rate_body": rate_rollout,
            "linear_velocity_rate_body_m_s2": rate_rollout[..., :3],
            "angular_velocity_rate_body_rad_s2": rate_rollout[..., 3:],
            "fixed_nominal_rate_body": torch.stack(future_fixed_nominal_rate, dim=1),
            "context_adapted_rate_body": torch.stack(
                future_context_adapted_rate, dim=1
            ),
            "domain_rate_correction_body": torch.stack(
                future_domain_rate_correction, dim=1
            ),
            "context_increment_rate_body": torch.stack(
                future_context_increment_rate, dim=1
            ),
            "nominal_rate_body": torch.stack(future_fixed_nominal_rate, dim=1),
            "residual_rate_body": torch.stack(future_residual_rate, dim=1),
            "residual_wrench_body": torch.stack(future_residual_wrench, dim=1),
            "residual_bias_wrench_body": torch.stack(
                future_residual_bias_wrench, dim=1
            ),
            "residual_action_wrench_body": torch.stack(
                future_residual_action_wrench, dim=1
            ),
            "latent_response_rate_body": torch.stack(
                future_latent_response_rate, dim=1
            ),
            "latent_response_potential_body": torch.stack(
                future_latent_response_potential, dim=1
            ),
            "commanded_modal_action": physical_action,
            "effective_modal_action": torch.stack(future_effective_action, dim=1),
            "interval_effective_modal_action": torch.stack(
                future_interval_effective_action, dim=1
            ),
            "modal_action_innovation": torch.stack(future_action_innovation, dim=1),
            "domain_mass_scale": domain_parameters["mass_scale"],
            "domain_control_scale": domain_parameters["control_scale"],
            "domain_damping_scale": domain_parameters["damping_scale"],
            "domain_actuator_time_constant_scale": domain_parameters[
                "actuator_time_constant_scale"
            ],
            "domain_actuator_time_constant_seconds": actuator_time_constant,
            "domain_parameter_normalized_deviation": domain_parameters[
                "normalized_deviation"
            ],
            "predicted_future_observation": self.normalize_observation(physical_future),
            "current_physical_observation": current,
        }

    def rollout_from_context(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
        *,
        context: Tensor | None = None,
        include_residual: bool = True,
    ) -> dict[str, Tensor]:
        """Roll out physical dynamics from fixed observed-history context.

        ``future_action`` is standardized at the model boundary. The optional
        context cache must come from the same observation/action history and is
        shared across candidate plans within one controller tick.
        """
        if action_history.shape[0] != future_action.shape[0]:
            raise ValueError("action history and future action batches must align")
        if context is None:
            context = self.backbone.encode_context(observation_history, action_history)
        physical_action = self.denormalize_action(future_action)
        action_embeddings = self.backbone.encode_action_windows(
            action_history, future_action
        )
        predicted_latent = self.backbone.rollout_action_embeddings(
            context, action_embeddings
        )
        output = self._integrate_dynamics(
            observation_history,
            action_history,
            context,
            physical_action,
            predicted_latent,
            include_residual=include_residual,
        )
        output["predicted_latent"] = predicted_latent
        return output

    def rollout_control_from_context(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
        *,
        context: Tensor | None = None,
        include_residual: bool = True,
    ) -> dict[str, Tensor]:
        """Run the numerically identical rollout with control-only outputs."""
        if action_history.shape[0] != future_action.shape[0]:
            raise ValueError("action history and future action batches must align")
        if context is None:
            context = self.backbone.encode_context(observation_history, action_history)
        physical_action = self.denormalize_action(future_action)
        action_embeddings = self.backbone.encode_action_windows(
            action_history, future_action
        )
        predicted_latent = self.backbone.rollout_action_embeddings(
            context, action_embeddings
        )
        return self._integrate_dynamics(
            observation_history,
            action_history,
            context,
            physical_action,
            predicted_latent,
            include_residual=include_residual,
            minimal_output=True,
        )

    def rollout_held_action(
        self,
        observation_history: Tensor,
        action_history: Tensor,
    ) -> dict[str, Tensor]:
        """Forecast the no-candidate fallback using the last applied action.

        This is deliberately separate from :meth:`forward`: a held-action
        forecast is a baseline prediction, not a candidate-control evaluation.
        It keeps the actuator lag state causal without inventing an action plan.
        """
        if action_history.ndim != 3 or action_history.shape[1] < 1:
            raise ValueError("action history must contain at least one timestep")
        future_action = action_history[:, -1:].expand(
            -1, self.backbone.config.horizon_steps, -1
        )
        return self.forward(observation_history, action_history, future_action)

    def forward(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
        future_observation: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if observation_history.shape[:2] != action_history.shape[:2]:
            raise ValueError("observation and action histories must align")
        if future_action.shape[1] != self.backbone.config.horizon_steps:
            raise ValueError("candidate motor plan has the wrong horizon")
        context = self.backbone.encode_context(observation_history, action_history)
        output = self.rollout_from_context(
            observation_history,
            action_history,
            future_action,
            context=context,
        )
        if future_observation is not None:
            output["target_latent"] = self.backbone.encode_target_windows(
                future_observation
            )
        return output


def _scaled_group_errors(
    model: PhysicsIntegratedStage2,
    output: Mapping[str, Tensor],
    target_standardized: Tensor,
) -> dict[str, Tensor]:
    predicted = output["predicted_future_physical_observation"]
    target = model.denormalize_observation(target_standardized)
    predicted_rotation = rotation6d_to_matrix(predicted[..., ROTATION_6D_SLICE])
    target_rotation = rotation6d_to_matrix(target[..., ROTATION_6D_SLICE])
    orientation_error = _wrap_angle(
        rotation_roll_pitch_yaw(predicted_rotation)
        - rotation_roll_pitch_yaw(target_rotation)
    )
    config = model.config
    velocity_scale = predicted.new_tensor(config.velocity_error_scale_m_s)
    roll_pitch_rate_scale = predicted.new_tensor(
        config.roll_pitch_rate_error_scale_rad_s
    )
    acceleration_scale = predicted.new_tensor(config.acceleration_error_scale_m_s2)
    return {
        "tilt": (orientation_error[..., :2] / config.tilt_error_scale_rad)
        .square()
        .mean(dim=-1),
        "yaw": (orientation_error[..., 2] / config.yaw_error_scale_rad).square(),
        "velocity": (
            (predicted[..., VIO_VELOCITY_SLICE] - target[..., VIO_VELOCITY_SLICE])
            / velocity_scale
        )
        .square()
        .mean(dim=-1),
        "roll_pitch_rate": (
            (
                predicted[..., IMU_GYRO_SLICE.start : IMU_GYRO_SLICE.start + 2]
                - target[..., IMU_GYRO_SLICE.start : IMU_GYRO_SLICE.start + 2]
            )
            / roll_pitch_rate_scale
        )
        .square()
        .mean(dim=-1),
        "yaw_rate": (
            (
                predicted[..., IMU_GYRO_SLICE.stop - 1]
                - target[..., IMU_GYRO_SLICE.stop - 1]
            )
            / config.yaw_rate_error_scale_rad_s
        ).square(),
        "acceleration": (
            (predicted[..., IMU_ACCEL_SLICE] - target[..., IMU_ACCEL_SLICE])
            / acceleration_scale
        )
        .square()
        .mean(dim=-1),
        "depth": (
            (predicted[..., DEPTH_INDEX] - target[..., DEPTH_INDEX])
            / config.depth_error_scale_m
        ).square(),
    }


def _rollout_error(
    model: PhysicsIntegratedStage2,
    output: Mapping[str, Tensor],
    target_standardized: Tensor,
    target_position_delta_ned_m: Tensor | None = None,
) -> Tensor:
    errors = _scaled_group_errors(model, output, target_standardized)
    config = model.config
    weighted = (
        config.tilt_loss_weight * errors["tilt"]
        + config.yaw_loss_weight * errors["yaw"]
        + config.velocity_loss_weight * errors["velocity"]
        + config.roll_pitch_rate_loss_weight * errors["roll_pitch_rate"]
        + config.yaw_rate_loss_weight * errors["yaw_rate"]
    )
    denominator = (
        config.tilt_loss_weight
        + config.yaw_loss_weight
        + config.velocity_loss_weight
        + config.roll_pitch_rate_loss_weight
        + config.yaw_rate_loss_weight
    )
    if target_position_delta_ned_m is not None:
        if (
            target_position_delta_ned_m.shape
            != output["predicted_position_delta_ned_m"].shape
        ):
            raise ValueError("future position delta differs from predicted rollout")
        position_scale = target_position_delta_ned_m.new_tensor(
            config.position_error_scale_m
        )
        position = (
            (
                (output["predicted_position_delta_ned_m"] - target_position_delta_ned_m)
                / position_scale
            )
            .square()
            .mean(dim=-1)
        )
        weighted = weighted + config.position_loss_weight * position
        denominator += config.position_loss_weight
    return weighted / denominator


def _factual_body_rate_target(
    model: PhysicsIntegratedStage2,
    target_standardized: Tensor,
    current_physical_observation: Tensor,
) -> Tensor:
    """Finite-difference factual acceleration target for every rollout step."""
    target = model.denormalize_observation(target_standardized)
    if current_physical_observation.ndim != 2:
        raise ValueError("current physical observation must be [batch, channel]")
    if target.shape[0] != current_physical_observation.shape[0]:
        raise ValueError("target and current observation batches must align")
    previous_velocity = torch.cat(
        (
            current_physical_observation[:, None, VIO_VELOCITY_SLICE],
            target[:, :-1, VIO_VELOCITY_SLICE],
        ),
        dim=1,
    )
    previous_omega = torch.cat(
        (
            current_physical_observation[:, None, IMU_GYRO_SLICE],
            target[:, :-1, IMU_GYRO_SLICE],
        ),
        dim=1,
    )
    return torch.cat(
        (
            (target[..., VIO_VELOCITY_SLICE] - previous_velocity) / model.dt,
            (target[..., IMU_GYRO_SLICE] - previous_omega) / model.dt,
        ),
        dim=-1,
    )


def stage2_loss(
    model: PhysicsIntegratedStage2,
    output: Mapping[str, Tensor],
    target_standardized: Tensor,
    future_position_delta_ned_m: Tensor,
) -> dict[str, Tensor]:
    if (
        future_position_delta_ned_m.ndim != 3
        or future_position_delta_ned_m.shape[-1] != 3
    ):
        raise ValueError("future position delta must be [batch, horizon, 3]")
    if future_position_delta_ned_m.shape[:2] != target_standardized.shape[:2]:
        raise ValueError(
            "future position delta and target observation horizons must align"
        )
    errors = _scaled_group_errors(model, output, target_standardized)
    config = model.config
    time_seconds = (
        torch.arange(
            1,
            target_standardized.shape[1] + 1,
            dtype=target_standardized.dtype,
            device=target_standardized.device,
        )
        * model.dt
    )
    horizon_weight = torch.where(
        time_seconds <= 0.2,
        1.5 * config.near_horizon_weight,
        torch.where(
            time_seconds <= 0.5,
            config.near_horizon_weight,
            torch.where(
                time_seconds <= 1.0,
                0.75 * config.near_horizon_weight,
                torch.where(time_seconds <= 2.0, 0.75, 0.35),
            ),
        ),
    )
    horizon_weight = horizon_weight / horizon_weight.mean()
    position_scale = future_position_delta_ned_m.new_tensor(
        config.position_error_scale_m
    )
    position_per_step = (
        (
            (output["predicted_position_delta_ned_m"] - future_position_delta_ned_m)
            / position_scale
        )
        .square()
        .mean(dim=-1)
    )
    position = position_per_step.mean()
    per_horizon = _rollout_error(
        model,
        output,
        target_standardized,
        future_position_delta_ned_m,
    )
    rollout = (per_horizon * horizon_weight).mean()
    target_rate = _factual_body_rate_target(
        model,
        target_standardized,
        output["current_physical_observation"],
    )
    rate_scale = target_rate.new_tensor(
        (
            *config.acceleration_error_scale_m_s2,
            *config.angular_acceleration_error_scale_rad_s2,
        )
    )
    scaled_rate_error = (
        (output["predicted_rate_body"] - target_rate) / rate_scale
    ).square()
    rate_per_step = scaled_rate_error.mean(dim=-1)
    rate = (rate_per_step * horizon_weight).mean()
    near_stop = min(
        max(1, int(round(config.near_horizon_rate_seconds * model.sample_rate_hz))),
        rate_per_step.shape[1],
    )
    controlled_rate_indices = (0, 1, 2, 5)
    near_horizon_rate = scaled_rate_error[:, :near_stop, controlled_rate_indices].mean()
    target_latent = output.get("target_latent")
    if target_latent is None:
        raise ValueError("Stage-2 training loss requires frozen target latents")
    latent_per_step = (output["predicted_latent"] - target_latent).square().mean(dim=-1)
    latent = (latent_per_step * horizon_weight).mean()
    residual_wrench = output.get("residual_wrench_body")
    residual_bias_wrench = output.get("residual_bias_wrench_body")
    residual_action_wrench = output.get("residual_action_wrench_body")
    if (
        residual_wrench is None
        or residual_bias_wrench is None
        or residual_action_wrench is None
    ):
        raise ValueError("Stage-2 output is missing residual wrench diagnostics")
    bias_limit = residual_wrench.new_tensor(
        model.config.residual_wrench_limit
    ).clamp_min(1e-6)
    action_limit = residual_wrench.new_tensor(
        model.config.residual_action_wrench_limit
    ).clamp_min(1e-6)
    residual_limit = bias_limit + action_limit
    normalized_bias = residual_bias_wrench / bias_limit
    normalized_action = residual_action_wrench / action_limit
    residual_bias_magnitude = normalized_bias.square().mean()
    residual_action_magnitude = normalized_action.square().mean()
    residual_magnitude = 0.5 * (residual_bias_magnitude + residual_action_magnitude)
    residual_near_limit_fraction = (
        torch.cat((normalized_bias.abs(), normalized_action.abs()), dim=-1)
        .ge(0.95)
        .float()
        .mean()
    )
    if residual_wrench.shape[1] > 1:
        residual_smoothness = (
            ((residual_wrench[:, 1:] - residual_wrench[:, :-1]) / residual_limit)
            .square()
            .mean()
        )
    else:
        residual_smoothness = residual_magnitude.new_zeros(())
    latent_response_rate = output.get("latent_response_rate_body")
    latent_response_potential = output.get("latent_response_potential_body")
    if latent_response_rate is None or latent_response_potential is None:
        raise ValueError("Stage-2 output is missing latent response diagnostics")
    latent_response_rate_limit = latent_response_rate.new_tensor(
        config.latent_response_rate_limit
    ).clamp_min(1e-6)
    latent_response_potential_limit = model.dt * latent_response_rate_limit
    controlled_axes = list(CONTROLLED_RATE_INDICES)
    normalized_latent_potential = (
        latent_response_potential[..., controlled_axes]
        / latent_response_potential_limit[controlled_axes]
    )
    latent_response_magnitude = normalized_latent_potential.square().mean()
    normalized_latent_rate = (
        latent_response_rate[..., controlled_axes]
        / latent_response_rate_limit[controlled_axes]
    )
    if latent_response_rate.shape[1] > 1:
        latent_response_smoothness = (
            (normalized_latent_rate[:, 1:] - normalized_latent_rate[:, :-1])
            .square()
            .mean()
        )
    else:
        latent_response_smoothness = latent_response_magnitude.new_zeros(())
    terminal_trajectory = torch.stack(
        (
            position_per_step[:, -1],
            errors["yaw"][:, -1],
            errors["velocity"][:, -1],
            errors["yaw_rate"][:, -1],
        ),
        dim=-1,
    ).mean()
    domain_parameter_deviation = (
        output["domain_parameter_normalized_deviation"].square().mean()
    )
    total = (
        config.trajectory_loss_weight * rollout
        + config.rate_loss_weight * rate
        + config.near_horizon_rate_loss_weight * near_horizon_rate
        + config.terminal_trajectory_loss_weight * terminal_trajectory
        + config.latent_loss_weight * latent
        + config.residual_magnitude_loss_weight * residual_magnitude
        + config.residual_smoothness_loss_weight * residual_smoothness
        + config.domain_parameter_loss_weight * domain_parameter_deviation
        + config.latent_response_magnitude_loss_weight * latent_response_magnitude
        + config.latent_response_smoothness_loss_weight * latent_response_smoothness
    )
    return {
        "total": total,
        "rollout": rollout,
        "rate": rate,
        "near_horizon_rate": near_horizon_rate,
        "terminal_trajectory": terminal_trajectory,
        "latent": latent,
        "residual_magnitude": residual_magnitude,
        "residual_bias_magnitude": residual_bias_magnitude,
        "residual_action_magnitude": residual_action_magnitude,
        "residual_smoothness": residual_smoothness,
        "residual_near_limit_fraction": residual_near_limit_fraction,
        "latent_response_magnitude": latent_response_magnitude,
        "latent_response_smoothness": latent_response_smoothness,
        "domain_parameter_deviation": domain_parameter_deviation,
        "tilt": errors["tilt"].mean(),
        "yaw": errors["yaw"].mean(),
        "velocity": errors["velocity"].mean(),
        "roll_pitch_rate": errors["roll_pitch_rate"].mean(),
        "yaw_rate": errors["yaw_rate"].mean(),
        "acceleration": errors["acceleration"].mean(),
        "depth": errors["depth"].mean(),
        "position": position,
        "per_horizon": per_horizon.mean(dim=0),
    }


def _delayed_action_plan(
    action_history: Tensor,
    future_action: Tensor,
) -> tuple[Tensor, Tensor]:
    """Delay a factual plan by one step for a conditioning diagnostic.

    The resulting rollout is not a labeled counterfactual and must never be
    used as a ranking target.
    """
    if action_history.ndim != 3 or future_action.ndim != 3:
        raise ValueError("actions must be [batch, time, channel]")
    if action_history.shape[0] != future_action.shape[0]:
        raise ValueError("action history and future action batches must align")
    delayed = torch.cat((action_history[:, -1:], future_action[:, :-1]), dim=1)
    rms = (delayed - future_action).square().mean(dim=(1, 2)).sqrt()
    return delayed, rms


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    required = (
        "observation_history",
        "action_history",
        "future_action",
        "future_observation",
        "future_position_delta_ned_m",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"missing Stage-2 batch fields: {missing}")
    moved = {
        name: batch[name].to(device=device, dtype=torch.float32, non_blocking=True)
        for name in required
    }
    return moved


def train_one_epoch(
    model: PhysicsIntegratedStage2,
    loader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    names = (
        "total",
        "rollout",
        "rate",
        "near_horizon_rate",
        "terminal_trajectory",
        "latent",
        "residual_magnitude",
        "residual_bias_magnitude",
        "residual_action_magnitude",
        "residual_smoothness",
        "residual_near_limit_fraction",
        "latent_response_magnitude",
        "latent_response_smoothness",
        "domain_parameter_deviation",
        "tilt",
        "yaw",
        "velocity",
        "roll_pitch_rate",
        "yaw_rate",
        "acceleration",
        "depth",
        "position",
    )
    sums = {name: 0.0 for name in names}
    sample_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
            batch["future_observation"],
        )
        losses = stage2_loss(
            model,
            output,
            batch["future_observation"],
            batch["future_position_delta_ned_m"],
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("non-finite Stage-2 loss")
        losses["total"].backward()
        clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            model.config.grad_clip_norm,
        )
        optimizer.step()
        for name in names:
            sums[name] += float(losses[name].detach().cpu()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("empty Stage-2 training loader")
    return {name: value / sample_count for name, value in sums.items()}


def _horizon_boundaries(horizon_steps: int) -> tuple[tuple[str, int, int], ...]:
    if horizon_steps < 1:
        raise ValueError("horizon must contain at least one step")
    stops = (
        min(5, horizon_steps),
        min(12, horizon_steps),
        min(25, horizon_steps),
        min(50, horizon_steps),
        horizon_steps,
    )
    starts = (0, 5, 12, 25, 50)
    return tuple(
        (f"steps_{start + 1}_{stop}", start, stop)
        for start, stop in zip(starts, stops, strict=True)
        if stop > start
    )


@dataclass
class _MetricAccumulator:
    horizon_steps: int
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if self.horizon_steps < 1 or self.sample_rate_hz <= 0.0:
            raise ValueError("metric horizon and sample rate must be positive")
        self.model_horizon = torch.zeros(self.horizon_steps, dtype=torch.float64)
        self.nominal_horizon = torch.zeros_like(self.model_horizon)
        self.persistence_horizon = torch.zeros_like(self.model_horizon)
        self.position_horizon = torch.zeros_like(self.model_horizon)
        self.nominal_position_horizon = torch.zeros_like(self.model_horizon)
        self.persistence_position_horizon = torch.zeros_like(self.model_horizon)
        self.rotation_angle_squared = 0.0
        self.orientation_component_squared = torch.zeros(3, dtype=torch.float64)
        self.velocity_squared = torch.zeros(3, dtype=torch.float64)
        self.angular_squared = torch.zeros(3, dtype=torch.float64)
        self.acceleration_squared = torch.zeros(3, dtype=torch.float64)
        self.depth_squared = 0.0
        self.position_squared = torch.zeros(3, dtype=torch.float64)
        self.orientation_horizon_squared = torch.zeros(
            self.horizon_steps, 3, dtype=torch.float64
        )
        self.velocity_horizon_squared = torch.zeros_like(
            self.orientation_horizon_squared
        )
        self.angular_horizon_squared = torch.zeros_like(
            self.orientation_horizon_squared
        )
        self.position_axis_horizon_squared = torch.zeros_like(
            self.orientation_horizon_squared
        )
        self.rate_error_squared = torch.zeros(6, dtype=torch.float64)
        self.nominal_rate_error_squared = torch.zeros_like(self.rate_error_squared)
        self.rate_predicted_sum = torch.zeros(6, dtype=torch.float64)
        self.rate_target_sum = torch.zeros(6, dtype=torch.float64)
        self.rate_predicted_squared = torch.zeros(6, dtype=torch.float64)
        self.rate_target_squared = torch.zeros(6, dtype=torch.float64)
        self.rate_cross_sum = torch.zeros(6, dtype=torch.float64)
        self.model_better = 0
        self.model_better_than_nominal = 0
        domain_parameter_count = 3 * len(RATE_LABELS) + len(ACTION_LABELS)
        self.domain_parameter_sum = torch.zeros(
            domain_parameter_count, dtype=torch.float64
        )
        self.domain_parameter_min = torch.full(
            (domain_parameter_count,), math.inf, dtype=torch.float64
        )
        self.domain_parameter_max = torch.full(
            (domain_parameter_count,), -math.inf, dtype=torch.float64
        )
        self.domain_normalized_abs_sum = torch.zeros(
            domain_parameter_count, dtype=torch.float64
        )
        self.domain_near_limit_count = torch.zeros(
            domain_parameter_count, dtype=torch.float64
        )
        self.sample_count = 0

    def update(
        self,
        model: PhysicsIntegratedStage2,
        output: Mapping[str, Tensor],
        nominal_output: Mapping[str, Tensor],
        target_standardized: Tensor,
        target_position_delta_ned_m: Tensor,
        current_standardized: Tensor,
    ) -> None:
        predicted_physical = output["predicted_future_physical_observation"]
        target_physical = model.denormalize_observation(target_standardized)
        batch_size, horizon, _ = target_physical.shape
        persistence_physical = model.denormalize_observation(current_standardized)
        persistence_physical = persistence_physical.expand(-1, horizon, -1).clone()
        persistence_output = {
            "predicted_future_physical_observation": persistence_physical,
            "predicted_position_delta_ned_m": torch.zeros_like(
                target_position_delta_ned_m
            ),
        }
        model_error = (
            _rollout_error(
                model,
                output,
                target_standardized,
                target_position_delta_ned_m,
            )
            .double()
            .cpu()
        )
        nominal_error = (
            _rollout_error(
                model,
                nominal_output,
                target_standardized,
                target_position_delta_ned_m,
            )
            .double()
            .cpu()
        )
        persistence_error = (
            _rollout_error(
                model,
                persistence_output,
                target_standardized,
                target_position_delta_ned_m,
            )
            .double()
            .cpu()
        )
        self.model_horizon += model_error.sum(dim=0)
        self.nominal_horizon += nominal_error.sum(dim=0)
        self.persistence_horizon += persistence_error.sum(dim=0)
        self.model_better += int(
            (model_error.mean(dim=1) < persistence_error.mean(dim=1)).sum()
        )
        self.model_better_than_nominal += int(
            (model_error.mean(dim=1) < nominal_error.mean(dim=1)).sum()
        )

        predicted_position_delta = output["predicted_position_delta_ned_m"]
        position_error = (
            (predicted_position_delta - target_position_delta_ned_m)
            .square()
            .mean(dim=-1)
            .double()
            .cpu()
        )
        nominal_position_error = (
            (
                nominal_output["predicted_position_delta_ned_m"]
                - target_position_delta_ned_m
            )
            .square()
            .mean(dim=-1)
            .double()
            .cpu()
        )
        persistence_position_error = target_position_delta_ned_m.square().mean(dim=-1)
        self.position_horizon += position_error.sum(dim=0)
        self.nominal_position_horizon += nominal_position_error.sum(dim=0)
        self.persistence_position_horizon += (
            persistence_position_error.double().cpu().sum(dim=0)
        )

        predicted_rotation = rotation6d_to_matrix(
            predicted_physical[..., ROTATION_6D_SLICE]
        )
        target_rotation = rotation6d_to_matrix(target_physical[..., ROTATION_6D_SLICE])
        self.rotation_angle_squared += float(
            rotation_squared_error(predicted_rotation, target_rotation)
            .double()
            .sum()
            .cpu()
        )
        orientation_error = _wrap_angle(
            rotation_roll_pitch_yaw(predicted_rotation)
            - rotation_roll_pitch_yaw(target_rotation)
        )
        self.orientation_component_squared += (
            orientation_error.square().double().sum(dim=(0, 1)).cpu()
        )
        self.orientation_horizon_squared += (
            orientation_error.square().double().sum(dim=0).cpu()
        )
        self.velocity_squared += (
            (
                predicted_physical[..., VIO_VELOCITY_SLICE]
                - target_physical[..., VIO_VELOCITY_SLICE]
            )
            .square()
            .double()
            .sum(dim=(0, 1))
            .cpu()
        )
        self.velocity_horizon_squared += (
            (
                predicted_physical[..., VIO_VELOCITY_SLICE]
                - target_physical[..., VIO_VELOCITY_SLICE]
            )
            .square()
            .double()
            .sum(dim=0)
            .cpu()
        )
        self.angular_squared += (
            (
                predicted_physical[..., IMU_GYRO_SLICE]
                - target_physical[..., IMU_GYRO_SLICE]
            )
            .square()
            .double()
            .sum(dim=(0, 1))
            .cpu()
        )
        self.angular_horizon_squared += (
            (
                predicted_physical[..., IMU_GYRO_SLICE]
                - target_physical[..., IMU_GYRO_SLICE]
            )
            .square()
            .double()
            .sum(dim=0)
            .cpu()
        )
        self.acceleration_squared += (
            (
                predicted_physical[..., IMU_ACCEL_SLICE]
                - target_physical[..., IMU_ACCEL_SLICE]
            )
            .square()
            .double()
            .sum(dim=(0, 1))
            .cpu()
        )
        self.depth_squared += float(
            (predicted_physical[..., DEPTH_INDEX] - target_physical[..., DEPTH_INDEX])
            .square()
            .double()
            .sum()
            .cpu()
        )
        self.position_squared += (
            (predicted_position_delta - target_position_delta_ned_m)
            .square()
            .double()
            .sum(dim=(0, 1))
            .cpu()
        )
        self.position_axis_horizon_squared += (
            (predicted_position_delta - target_position_delta_ned_m)
            .square()
            .double()
            .sum(dim=0)
            .cpu()
        )

        current_physical = model.denormalize_observation(current_standardized)
        previous_velocity = torch.cat(
            (
                current_physical[..., VIO_VELOCITY_SLICE],
                target_physical[:, :-1, VIO_VELOCITY_SLICE],
            ),
            dim=1,
        )
        previous_angular = torch.cat(
            (
                current_physical[..., IMU_GYRO_SLICE],
                target_physical[:, :-1, IMU_GYRO_SLICE],
            ),
            dim=1,
        )
        target_rate = torch.cat(
            (
                (target_physical[..., VIO_VELOCITY_SLICE] - previous_velocity)
                / model.dt,
                (target_physical[..., IMU_GYRO_SLICE] - previous_angular) / model.dt,
            ),
            dim=-1,
        )
        predicted_rate = output["predicted_rate_body"]
        nominal_rate = nominal_output["predicted_rate_body"]
        predicted_double = predicted_rate.double().cpu()
        nominal_double = nominal_rate.double().cpu()
        target_double = target_rate.double().cpu()
        self.rate_error_squared += (
            (predicted_double - target_double).square().sum(dim=(0, 1))
        )
        self.nominal_rate_error_squared += (
            (nominal_double - target_double).square().sum(dim=(0, 1))
        )
        self.rate_predicted_sum += predicted_double.sum(dim=(0, 1))
        self.rate_target_sum += target_double.sum(dim=(0, 1))
        self.rate_predicted_squared += predicted_double.square().sum(dim=(0, 1))
        self.rate_target_squared += target_double.square().sum(dim=(0, 1))
        self.rate_cross_sum += (predicted_double * target_double).sum(dim=(0, 1))
        domain_scales = (
            torch.cat(
                (
                    output["domain_mass_scale"],
                    output["domain_control_scale"],
                    output["domain_damping_scale"],
                    output["domain_actuator_time_constant_scale"],
                ),
                dim=-1,
            )
            .double()
            .cpu()
        )
        normalized_domain = (
            output["domain_parameter_normalized_deviation"].double().cpu()
        )
        self.domain_parameter_sum += domain_scales.sum(dim=0)
        self.domain_parameter_min = torch.minimum(
            self.domain_parameter_min, domain_scales.amin(dim=0)
        )
        self.domain_parameter_max = torch.maximum(
            self.domain_parameter_max, domain_scales.amax(dim=0)
        )
        self.domain_normalized_abs_sum += normalized_domain.abs().sum(dim=0)
        self.domain_near_limit_count += normalized_domain.abs().ge(0.95).sum(dim=0)
        self.sample_count += batch_size

    @staticmethod
    def _improvement(model_value: float, baseline_value: float) -> float:
        return (
            0.0
            if baseline_value <= 0.0
            else (baseline_value - model_value) / baseline_value
        )

    def result(self) -> dict[str, object]:
        if self.sample_count < 1:
            raise ValueError("cannot finalize empty Stage-2 metrics")
        model_horizon = self.model_horizon / self.sample_count
        nominal_horizon = self.nominal_horizon / self.sample_count
        persistence_horizon = self.persistence_horizon / self.sample_count
        position_horizon = self.position_horizon / self.sample_count
        nominal_position_horizon = self.nominal_position_horizon / self.sample_count
        persistence_position_horizon = (
            self.persistence_position_horizon / self.sample_count
        )
        model_mse = float(model_horizon.mean())
        nominal_mse = float(nominal_horizon.mean())
        persistence_mse = float(persistence_horizon.mean())

        def group(start: int, stop: int) -> dict[str, float]:
            model_value = float(model_horizon[start:stop].mean())
            baseline_value = float(persistence_horizon[start:stop].mean())
            return {
                "model_mse": model_value,
                "persistence_mse": baseline_value,
                "relative_improvement": self._improvement(model_value, baseline_value),
            }

        boundaries = _horizon_boundaries(self.horizon_steps)
        groups = {
            name: group(start, stop) for name, start, stop in boundaries if stop > start
        }
        position_groups = {
            name: {
                "model_mse": float(position_horizon[start:stop].mean()),
                "persistence_mse": float(
                    persistence_position_horizon[start:stop].mean()
                ),
                "relative_improvement": self._improvement(
                    float(position_horizon[start:stop].mean()),
                    float(persistence_position_horizon[start:stop].mean()),
                ),
            }
            for name, start, stop in boundaries
            if stop > start
        }
        nominal_groups = {
            name: {
                "hybrid_mse": float(model_horizon[start:stop].mean()),
                "nominal_mse": float(nominal_horizon[start:stop].mean()),
                "hybrid_improvement": self._improvement(
                    float(model_horizon[start:stop].mean()),
                    float(nominal_horizon[start:stop].mean()),
                ),
            }
            for name, start, stop in boundaries
            if stop > start
        }
        point_count = self.sample_count * self.horizon_steps
        rate_target_mean = self.rate_target_sum / point_count
        rate_predicted_mean = self.rate_predicted_sum / point_count
        rate_target_ss = (
            self.rate_target_squared - point_count * rate_target_mean.square()
        )
        rate_predicted_ss = (
            self.rate_predicted_squared - point_count * rate_predicted_mean.square()
        )
        rate_covariance = (
            self.rate_cross_sum - point_count * rate_target_mean * rate_predicted_mean
        )
        rate_r2 = 1.0 - self.rate_error_squared / rate_target_ss.clamp_min(1e-12)
        rate_correlation = rate_covariance / torch.sqrt(
            rate_target_ss.clamp_min(1e-12) * rate_predicted_ss.clamp_min(1e-12)
        )

        def domain_group(
            start: int,
            stop: int,
            labels: tuple[str, ...],
        ) -> dict[str, object]:
            return {
                "labels": list(labels),
                "mean_scale": (
                    self.domain_parameter_sum[start:stop] / self.sample_count
                ).tolist(),
                "minimum_scale": self.domain_parameter_min[start:stop].tolist(),
                "maximum_scale": self.domain_parameter_max[start:stop].tolist(),
                "mean_absolute_normalized_deviation": (
                    self.domain_normalized_abs_sum[start:stop] / self.sample_count
                ).tolist(),
                "near_limit_fraction": (
                    self.domain_near_limit_count[start:stop] / self.sample_count
                ).tolist(),
            }

        rate_count = len(RATE_LABELS)
        action_start = 3 * rate_count
        domain_diagnostics = {
            "mass": domain_group(0, rate_count, RATE_LABELS),
            "control": domain_group(rate_count, 2 * rate_count, RATE_LABELS),
            "damping": domain_group(2 * rate_count, action_start, RATE_LABELS),
            "actuator_time_constant": domain_group(
                action_start,
                action_start + len(ACTION_LABELS),
                ACTION_LABELS,
            ),
            "maximum_near_limit_fraction": float(
                (self.domain_near_limit_count / self.sample_count).max()
            ),
        }
        state_rmse_horizon = torch.sqrt(model_horizon.clamp_min(0.0))
        state_error_growth = torch.diff(
            state_rmse_horizon,
            prepend=state_rmse_horizon.new_zeros(1),
        )

        requested_horizons_seconds = (0.5, 1.0, 2.0, 3.0)
        fidelity_at_seconds: dict[str, object] = {}
        used_steps: set[int] = set()
        for requested_seconds in requested_horizons_seconds:
            step = min(
                self.horizon_steps,
                max(1, int(round(requested_seconds * self.sample_rate_hz))),
            )
            if step in used_steps:
                continue
            used_steps.add(step)
            index = step - 1
            position_axis_rmse = torch.sqrt(
                self.position_axis_horizon_squared[index] / self.sample_count
            )
            velocity_axis_rmse = torch.sqrt(
                self.velocity_horizon_squared[index] / self.sample_count
            )
            angular_axis_rmse = torch.sqrt(
                self.angular_horizon_squared[index] / self.sample_count
            )
            orientation_axis_rmse = torch.sqrt(
                self.orientation_horizon_squared[index] / self.sample_count
            )
            actual_seconds = step / self.sample_rate_hz
            fidelity_at_seconds[f"{actual_seconds:g}s"] = {
                "step": step,
                "normalized_state_rmse": float(state_rmse_horizon[index]),
                "position_axis_rmse_m": position_axis_rmse.tolist(),
                "position_vector_rmse_m": float(
                    torch.linalg.vector_norm(position_axis_rmse)
                ),
                "body_velocity_axis_rmse_m_s": velocity_axis_rmse.tolist(),
                "body_velocity_vector_rmse_m_s": float(
                    torch.linalg.vector_norm(velocity_axis_rmse)
                ),
                "yaw_rmse_rad": float(orientation_axis_rmse[2]),
                "yaw_rmse_deg": math.degrees(float(orientation_axis_rmse[2])),
                "yaw_rate_rmse_rad_s": float(angular_axis_rmse[2]),
            }
        return {
            "sample_count": self.sample_count,
            "normalized_state_mse": model_mse,
            "normalized_state_rmse": math.sqrt(model_mse),
            "nominal_physics_normalized_state_mse": nominal_mse,
            "nominal_physics_normalized_state_rmse": math.sqrt(nominal_mse),
            "relative_improvement_over_nominal_physics": self._improvement(
                model_mse, nominal_mse
            ),
            "model_better_than_nominal_physics_fraction": self.model_better_than_nominal
            / self.sample_count,
            "persistence_normalized_state_mse": persistence_mse,
            "relative_improvement_over_persistence": self._improvement(
                model_mse, persistence_mse
            ),
            "model_better_than_persistence_fraction": self.model_better
            / self.sample_count,
            "domain_parameter_diagnostics": domain_diagnostics,
            "per_horizon_state_mse": model_horizon.tolist(),
            "per_horizon_state_rmse": state_rmse_horizon.tolist(),
            "per_step_state_rmse_growth": state_error_growth.tolist(),
            "mean_state_rmse_growth_per_step": float(state_error_growth.mean()),
            "rollout_fidelity_at_seconds": fidelity_at_seconds,
            "persistence_per_horizon_state_mse": persistence_horizon.tolist(),
            "horizon_groups": groups,
            "nominal_physics_per_horizon_state_mse": nominal_horizon.tolist(),
            "nominal_physics_horizon_groups": nominal_groups,
            "position_displacement": {
                "mse": float(position_horizon.mean()),
                "rmse_m": math.sqrt(float(position_horizon.mean())),
                "nominal_physics_mse": float(nominal_position_horizon.mean()),
                "nominal_physics_rmse_m": math.sqrt(
                    float(nominal_position_horizon.mean())
                ),
                "relative_improvement_over_nominal_physics": self._improvement(
                    float(position_horizon.mean()),
                    float(nominal_position_horizon.mean()),
                ),
                "persistence_mse": float(persistence_position_horizon.mean()),
                "relative_improvement_over_persistence": self._improvement(
                    float(position_horizon.mean()),
                    float(persistence_position_horizon.mean()),
                ),
                "per_horizon_mse": position_horizon.tolist(),
                "nominal_physics_per_horizon_mse": nominal_position_horizon.tolist(),
                "persistence_per_horizon_mse": persistence_position_horizon.tolist(),
                "horizon_groups": position_groups,
            },
            "physical_rmse": {
                "rotation_rad": math.sqrt(self.rotation_angle_squared / point_count),
                "rotation_deg": math.degrees(
                    math.sqrt(self.rotation_angle_squared / point_count)
                ),
                "roll_pitch_rad": torch.sqrt(
                    self.orientation_component_squared[:2] / point_count
                ).tolist(),
                "roll_pitch_deg": torch.rad2deg(
                    torch.sqrt(self.orientation_component_squared[:2] / point_count)
                ).tolist(),
                "yaw_rad": math.sqrt(
                    float(self.orientation_component_squared[2]) / point_count
                ),
                "yaw_deg": math.degrees(
                    math.sqrt(
                        float(self.orientation_component_squared[2]) / point_count
                    )
                ),
                "velocity_m_s": torch.sqrt(
                    self.velocity_squared / point_count
                ).tolist(),
                "angular_velocity_rad_s": torch.sqrt(
                    self.angular_squared / point_count
                ).tolist(),
                "imu_linear_acceleration_m_s2": torch.sqrt(
                    self.acceleration_squared / point_count
                ).tolist(),
                "depth_m": math.sqrt(self.depth_squared / point_count),
                "position_delta_ned_m": torch.sqrt(
                    self.position_squared / point_count
                ).tolist(),
            },
            "physical_rate_prediction": {
                "labels": list(RATE_LABELS),
                "rmse": torch.sqrt(self.rate_error_squared / point_count).tolist(),
                "nominal_physics_rmse": torch.sqrt(
                    self.nominal_rate_error_squared / point_count
                ).tolist(),
                "target_std": torch.sqrt(rate_target_ss / point_count).tolist(),
                "r2": rate_r2.tolist(),
                "correlation": rate_correlation.tolist(),
            },
        }


@torch.no_grad()
def evaluate(
    model: PhysicsIntegratedStage2,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    accumulator = _MetricAccumulator(
        model.backbone.config.horizon_steps,
        model.sample_rate_hz,
    )
    loss_names = (
        "total",
        "rollout",
        "rate",
        "near_horizon_rate",
        "terminal_trajectory",
        "latent",
        "residual_magnitude",
        "residual_bias_magnitude",
        "residual_action_magnitude",
        "residual_smoothness",
        "residual_near_limit_fraction",
        "latent_response_magnitude",
        "latent_response_smoothness",
        "domain_parameter_deviation",
        "tilt",
        "yaw",
        "velocity",
        "roll_pitch_rate",
        "yaw_rate",
        "acceleration",
        "depth",
        "position",
    )
    loss_sums = {name: 0.0 for name in loss_names}
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        output = model(
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
            batch["future_observation"],
        )
        nominal_output = model.rollout_from_context(
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
            include_residual=False,
        )
        losses = stage2_loss(
            model,
            output,
            batch["future_observation"],
            batch["future_position_delta_ned_m"],
        )
        for name in loss_names:
            loss_sums[name] += float(losses[name].cpu()) * batch_size
        accumulator.update(
            model,
            output,
            nominal_output,
            batch["future_observation"],
            batch["future_position_delta_ned_m"],
            batch["observation_history"][:, -1:],
        )
    result = accumulator.result()
    result["loss"] = {
        name: value / accumulator.sample_count for name, value in loss_sums.items()
    }
    return result


@torch.no_grad()
def audit_conditioning(
    model: PhysicsIntegratedStage2,
    loader: Iterable[Mapping[str, Tensor]],
    standardizer: Standardizer,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Ablate candidate actions without claiming counterfactual truth.

    Validation windows are ordered by episode and overlap heavily.  Local
    action ablation is a one-step temporal delay within each sample; the
    cross-episode replacements remain provenance-changing diagnostics only.
    """
    model.eval()
    names = (
        "factual",
        "held_action",
        "zero_action",
        "local_action_shift",
        "cross_episode_action",
        "matched_cross_episode_action",
    )
    error_sums = {name: 0.0 for name in names}
    better_counts = {name: 0 for name in names if name != "factual"}
    displacement_sums = {name: 0.0 for name in names if name != "factual"}
    input_displacement_sums = {name: 0.0 for name in names if name != "factual"}
    comparison_counts = {name: 0 for name in names if name != "factual"}
    horizon_steps = model.backbone.config.horizon_steps
    horizon_error_sums = {
        name: torch.zeros(horizon_steps, dtype=torch.float64) for name in names
    }
    horizon_better_counts = {
        name: torch.zeros(horizon_steps, dtype=torch.int64)
        for name in names
        if name != "factual"
    }
    boundaries = _horizon_boundaries(horizon_steps)
    group_better_counts = {
        name: {group: 0 for group, start, stop in boundaries if stop > start}
        for name in names
        if name != "factual"
    }
    sample_count = 0
    zero_action = torch.as_tensor(
        -standardizer.action_mean / standardizer.action_std,
        dtype=torch.float32,
        device=device,
    )
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        episode_index = raw_batch.get("episode_index")
        if episode_index is not None:
            episode_index = episode_index.to(device=device, dtype=torch.int64)
        factual = model(
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
        )
        factual_error = _rollout_error(
            model,
            factual,
            batch["future_observation"],
            batch["future_position_delta_ned_m"],
        )
        held_action = batch["action_history"][:, -1:].expand_as(batch["future_action"])
        alternatives: dict[str, Mapping[str, Tensor]] = {
            "held_action": model(
                batch["observation_history"],
                batch["action_history"],
                held_action,
            ),
            "zero_action": model(
                batch["observation_history"],
                batch["action_history"],
                zero_action.expand_as(batch["future_action"]),
            ),
        }
        input_displacements: dict[str, Tensor] = {
            "held_action": (held_action - batch["future_action"])
            .square()
            .mean(dim=(1, 2))
            .sqrt(),
            "zero_action": (
                zero_action.expand_as(batch["future_action"]) - batch["future_action"]
            )
            .square()
            .mean(dim=(1, 2))
            .sqrt(),
        }
        if batch_size >= 2:
            local_action, _ = _delayed_action_plan(
                batch["action_history"],
                batch["future_action"],
            )
            alternatives["local_action_shift"] = model(
                batch["observation_history"],
                batch["action_history"],
                local_action,
            )
            input_displacements["local_action_shift"] = (
                (local_action - batch["future_action"]).square().mean(dim=(1, 2)).sqrt()
            )
            cross_indices = (
                _cross_episode_indices(episode_index)
                if episode_index is not None
                else None
            )
            if cross_indices is not None:
                cross_action = batch["future_action"][cross_indices]
                alternatives["cross_episode_action"] = model(
                    batch["observation_history"],
                    batch["action_history"],
                    cross_action,
                )
                input_displacements["cross_episode_action"] = (
                    (cross_action - batch["future_action"])
                    .square()
                    .mean(dim=(1, 2))
                    .sqrt()
                )
            matched_indices = (
                _matched_episode_indices(episode_index, batch["future_action"])
                if episode_index is not None
                else None
            )
            if matched_indices is not None:
                matched_action = batch["future_action"][matched_indices]
                alternatives["matched_cross_episode_action"] = model(
                    batch["observation_history"],
                    batch["action_history"],
                    matched_action,
                )
                input_displacements["matched_cross_episode_action"] = (
                    (matched_action - batch["future_action"])
                    .square()
                    .mean(dim=(1, 2))
                    .sqrt()
                )
        error_sums["factual"] += float(factual_error.mean(dim=1).sum().cpu())
        horizon_error_sums["factual"] += factual_error.double().sum(dim=0).cpu()
        factual_physical = factual["predicted_future_physical_observation"]
        for name, alternative in alternatives.items():
            alternative_error = _rollout_error(
                model,
                alternative,
                batch["future_observation"],
                batch["future_position_delta_ned_m"],
            )
            error_sums[name] += float(alternative_error.mean(dim=1).sum().cpu())
            horizon_error_sums[name] += alternative_error.double().sum(dim=0).cpu()
            better_counts[name] += int(
                (factual_error.mean(dim=1) < alternative_error.mean(dim=1)).sum().cpu()
            )
            horizon_better_counts[name] += (
                (factual_error < alternative_error).sum(dim=0).cpu()
            )
            for group, start, stop in boundaries:
                if stop > start:
                    group_better_counts[name][group] += int(
                        (
                            factual_error[:, start:stop].mean(dim=1)
                            < alternative_error[:, start:stop].mean(dim=1)
                        )
                        .sum()
                        .cpu()
                    )
            factual_as_target = model.normalize_observation(factual_physical)
            displacement_sums[name] += float(
                _rollout_error(
                    model,
                    alternative,
                    factual_as_target,
                    factual["predicted_position_delta_ned_m"],
                )
                .mean(dim=1)
                .sqrt()
                .sum()
                .cpu()
            )
            input_displacement_sums[name] += float(
                input_displacements[name].sum().cpu()
            )
            comparison_counts[name] += batch_size
        sample_count += batch_size
    if sample_count < 1:
        raise ValueError("empty conditioning audit loader")
    result: dict[str, object] = {
        "sample_count": sample_count,
        "factual_state_mse": error_sums["factual"] / sample_count,
    }
    for name in names[1:]:
        denominator = comparison_counts[name]
        if denominator == 0:
            result[name] = None
            continue
        result[name] = {
            "comparison_count": denominator,
            "state_mse": error_sums[name] / denominator,
            "factual_better_fraction": better_counts[name] / denominator,
            "conditioning_input_rms_displacement": input_displacement_sums[name]
            / denominator,
            "normalized_state_rollout_rms_displacement": displacement_sums[name]
            / denominator,
            "per_horizon_state_mse": (horizon_error_sums[name] / denominator).tolist(),
            "per_horizon_factual_better_fraction": (
                horizon_better_counts[name].double() / denominator
            ).tolist(),
            "horizon_group_factual_better_fraction": {
                group: count / denominator
                for group, count in group_better_counts[name].items()
            },
        }
    result["factual_per_horizon_state_mse"] = (
        horizon_error_sums["factual"] / sample_count
    ).tolist()
    return result


@torch.no_grad()
def audit_control_derivatives(
    model: PhysicsIntegratedStage2,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    standardized_perturbation: float = 0.10,
    perturbed_steps: int = 3,
    maximum_batches: int = 8,
) -> dict[str, object]:
    """Measure local rate and trajectory sensitivity without claiming truth.

    NED displacement is reported for inspection. Readiness uses displacement
    rotated into the current body frame, since a valid surge response need not
    point north when the vehicle starts with nonzero yaw.
    """
    if standardized_perturbation <= 0.0 or perturbed_steps < 1 or maximum_batches < 1:
        raise ValueError("invalid derivative audit configuration")
    model.eval()
    horizons = tuple(
        index for index in (0, 2, 4) if index < model.backbone.config.horizon_steps
    )
    rate_sums = torch.zeros(
        (len(horizons), len(ACTION_LABELS), len(RATE_LABELS)),
        dtype=torch.float64,
    )
    nominal_rate_sums = torch.zeros_like(rate_sums)
    learned_correction_rate_sums = torch.zeros_like(rate_sums)
    ned_response_sums = torch.zeros(
        (len(horizons), len(ACTION_LABELS), 4), dtype=torch.float64
    )
    command_response_sums = torch.zeros_like(ned_response_sums)
    sample_count = 0
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        for action_index in range(len(ACTION_LABELS)):
            positive = batch["future_action"].clone()
            negative = batch["future_action"].clone()
            stop = min(perturbed_steps, positive.shape[1])
            positive[:, :stop, action_index] += standardized_perturbation
            negative[:, :stop, action_index] -= standardized_perturbation
            positive_output = model(
                batch["observation_history"],
                batch["action_history"],
                positive,
            )
            negative_output = model(
                batch["observation_history"],
                batch["action_history"],
                negative,
            )
            physical_width = (
                2.0 * standardized_perturbation * model.action_std[action_index]
            )
            derivative = (
                positive_output["predicted_rate_body"]
                - negative_output["predicted_rate_body"]
            ) / physical_width
            nominal_derivative = (
                positive_output["nominal_rate_body"]
                - negative_output["nominal_rate_body"]
            ) / physical_width
            learned_correction_derivative = derivative - nominal_derivative
            position_difference_ned = (
                positive_output["predicted_position_delta_ned_m"]
                - negative_output["predicted_position_delta_ned_m"]
            ) / physical_width
            current_rotation = rotation6d_to_matrix(
                positive_output["current_physical_observation"][..., ROTATION_6D_SLICE]
            )
            position_difference_body = (
                current_rotation.transpose(-1, -2).unsqueeze(1)
                @ position_difference_ned.unsqueeze(-1)
            ).squeeze(-1)
            positive_rotation = rotation6d_to_matrix(
                positive_output["predicted_future_physical_observation"][
                    ..., ROTATION_6D_SLICE
                ]
            )
            negative_rotation = rotation6d_to_matrix(
                negative_output["predicted_future_physical_observation"][
                    ..., ROTATION_6D_SLICE
                ]
            )
            yaw_difference = (
                _wrap_angle(
                    rotation_roll_pitch_yaw(positive_rotation)[..., 2]
                    - rotation_roll_pitch_yaw(negative_rotation)[..., 2]
                )
                / physical_width
            )
            ned_response = torch.cat(
                (position_difference_ned, yaw_difference.unsqueeze(-1)), dim=-1
            )
            command_response = torch.cat(
                (position_difference_body, yaw_difference.unsqueeze(-1)), dim=-1
            )
            for output_index, horizon in enumerate(horizons):
                rate_sums[output_index, action_index] += (
                    derivative[:, horizon].double().sum(dim=0).cpu()
                )
                nominal_rate_sums[output_index, action_index] += (
                    nominal_derivative[:, horizon].double().sum(dim=0).cpu()
                )
                learned_correction_rate_sums[output_index, action_index] += (
                    learned_correction_derivative[:, horizon].double().sum(dim=0).cpu()
                )
                ned_response_sums[output_index, action_index] += (
                    ned_response[:, horizon].double().sum(dim=0).cpu()
                )
                command_response_sums[output_index, action_index] += (
                    command_response[:, horizon].double().sum(dim=0).cpu()
                )
        sample_count += batch_size
    if sample_count < 1:
        raise ValueError("empty derivative audit loader")
    rate_jacobian = rate_sums / sample_count
    nominal_rate_jacobian = nominal_rate_sums / sample_count
    learned_correction_rate_jacobian = learned_correction_rate_sums / sample_count
    ned_response_jacobian = ned_response_sums / sample_count
    command_response_jacobian = command_response_sums / sample_count
    # Roll is passively stabilized and is not a commanded trajectory axis.
    # Readiness therefore uses only surge, sway, heave, and yaw.
    controlled_action_indices = (0, 1, 2, 3)
    expected_rate_indices = (0, 1, 2, 5)
    expected_signs = (1.0, 1.0, -1.0, 1.0)
    first_step_rate_direct = torch.stack(
        tuple(
            rate_jacobian[0, action_index, rate_index] * sign
            for action_index, rate_index, sign in zip(
                controlled_action_indices,
                expected_rate_indices,
                expected_signs,
                strict=True,
            )
        )
    )
    first_step_controlled_rate = (
        rate_jacobian[0, list(controlled_action_indices)].abs().clone()
    )
    first_step_rate_off_axis = first_step_controlled_rate.clone()
    for row_index, rate_index in enumerate(expected_rate_indices):
        first_step_rate_off_axis[row_index, rate_index] = 0.0
    first_step_rate_off_axis_ratio = float(
        first_step_rate_off_axis.sum()
        / first_step_controlled_rate.sum().clamp_min(1e-12)
    )

    expected_response_indices = (0, 1, 2, 3)
    command_direct = torch.stack(
        tuple(
            command_response_jacobian[:, action_index, response_index] * sign
            for action_index, response_index, sign in zip(
                controlled_action_indices,
                expected_response_indices,
                expected_signs,
                strict=True,
            )
        ),
        dim=1,
    )
    controlled_response = (
        command_response_jacobian[:, list(controlled_action_indices)].abs().clone()
    )
    response_off_axis = controlled_response.clone()
    for row_index, response_index in enumerate(expected_response_indices):
        response_off_axis[:, row_index, response_index] = 0.0
    response_off_axis_ratio = response_off_axis.sum(dim=(1, 2)) / (
        controlled_response.sum(dim=(1, 2)).clamp_min(1e-12)
    )
    all_finite = bool(
        torch.isfinite(rate_jacobian).all()
        and torch.isfinite(nominal_rate_jacobian).all()
        and torch.isfinite(learned_correction_rate_jacobian).all()
        and torch.isfinite(ned_response_jacobian).all()
        and torch.isfinite(command_response_jacobian).all()
    )
    return {
        "sample_count": sample_count,
        "standardized_perturbation": standardized_perturbation,
        "perturbed_steps": perturbed_steps,
        "horizon_steps": [index + 1 for index in horizons],
        "horizon_seconds": [(index + 1) * model.dt for index in horizons],
        "rate_labels": list(RATE_LABELS),
        "action_labels": list(ACTION_LABELS),
        "rate_jacobian_per_physical_modal_action": rate_jacobian.tolist(),
        "nominal_rate_jacobian_per_physical_modal_action": nominal_rate_jacobian.tolist(),
        "learned_correction_rate_jacobian_per_physical_modal_action": (
            learned_correction_rate_jacobian.tolist()
        ),
        "expected_first_step_rate_direct_signed_gain": first_step_rate_direct.tolist(),
        "ned_response_labels": [
            "position_delta_n_m",
            "position_delta_e_m",
            "position_delta_d_m",
            "yaw_delta_rad",
        ],
        "ned_response_jacobian_per_physical_modal_action": (
            ned_response_jacobian.tolist()
        ),
        "command_response_frame": "current_body_frd_plus_wrapped_yaw",
        "command_response_labels": [
            "position_delta_body_x_m",
            "position_delta_body_y_m",
            "position_delta_body_z_m",
            "yaw_delta_rad",
        ],
        "command_response_jacobian_per_physical_modal_action": (
            command_response_jacobian.tolist()
        ),
        "expected_direct_signed_response_per_horizon": command_direct.tolist(),
        "controlled_action_labels": [
            ACTION_LABELS[index] for index in controlled_action_indices
        ],
        "all_expected_first_step_rate_direct_signs_positive": bool(
            (first_step_rate_direct > 0.0).all()
        ),
        "first_step_rate_off_axis_absolute_fraction": (first_step_rate_off_axis_ratio),
        "first_step_rate_jacobian_l1": float(rate_jacobian[0].abs().sum()),
        "first_checked_horizon_nominal_rate_jacobian_l1": float(
            nominal_rate_jacobian[0].abs().sum()
        ),
        "first_checked_horizon_learned_correction_rate_jacobian_l1": float(
            learned_correction_rate_jacobian[0].abs().sum()
        ),
        "all_checked_horizon_direct_signs_positive": bool((command_direct > 0.0).all()),
        "off_axis_fraction_per_horizon": response_off_axis_ratio.tolist(),
        "maximum_checked_horizon_off_axis_fraction": float(
            response_off_axis_ratio.max()
        ),
        "first_checked_horizon_response_jacobian_l1": float(
            command_response_jacobian[0].abs().sum()
        ),
        "all_finite": all_finite,
        "counterfactual_ground_truth": False,
        "interpretation": "learned local sensitivity audit only",
    }


def _load_stage1_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[PredictiveStateJEPA, dict[str, object]]:
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Stage-1 checkpoint must contain a dictionary")
    required = (
        "feature_contract_id",
        "action_labels",
        "model_configuration",
        "model_state_dict",
        "standardizer",
        "history_steps",
        "horizon_steps",
        "sample_rate_hz",
        "recorder_sample_rate_hz",
        "model_step_stride",
        "temporal_resampling_id",
        "action_contract_id",
        "actuator_model",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise KeyError(f"Stage-1 checkpoint is missing fields: {missing}")
    if checkpoint.get("implementation") != STAGE1_IMPLEMENTATION:
        raise ValueError("Stage-1 checkpoint does not use the retained underwater TCN")
    readiness = checkpoint.get("training_readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("Stage-1 checkpoint is missing training diagnostics")
    if checkpoint["feature_contract_id"] != FEATURE_CONTRACT_ID:
        raise ValueError("Stage-1 feature contract does not match the current loader")
    if tuple(checkpoint["action_labels"]) != ACTION_LABELS:
        raise ValueError("Stage-1 action labels do not match the modal contract")
    if checkpoint["action_contract_id"] != ACTION_CONTRACT_ID:
        raise ValueError("Stage-1 action contract does not match the current loader")
    if checkpoint["actuator_model"] != ACTUATOR_MODEL_ID:
        raise ValueError("Stage-1 actuator model is not the retained modal contract")
    if checkpoint["temporal_resampling_id"] != TEMPORAL_RESAMPLING_ID:
        raise ValueError("Stage-1 temporal resampling contract is unsupported")
    backbone = PredictiveStateJEPA(Stage1Config(**checkpoint["model_configuration"]))
    backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    backbone.requires_grad_(False)
    backbone.eval()
    backbone.to(device)
    return backbone, checkpoint


def _random_backbone_from_contract(
    config: Stage1Config,
    *,
    seed: int,
    device: torch.device,
) -> PredictiveStateJEPA:
    """Build a scratch encoder without perturbing matched decoder initialization."""
    with torch.random.fork_rng(devices=[]):
        torch.random.manual_seed(seed)
        backbone = PredictiveStateJEPA(config)
    backbone.to(device)
    return backbone


def load_stage2_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PhysicsIntegratedStage2, dict[str, object]]:
    """Load the retained Stage-2 checkpoint without legacy format branches."""
    target_device = torch.device(device)
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("implementation") != IMPLEMENTATION
    ):
        raise ValueError("checkpoint is not the retained physics-integrated Stage 2")
    required_contract = (
        "feature_contract_id",
        "action_contract_id",
        "actuator_model",
        "sample_rate_hz",
        "recorder_sample_rate_hz",
        "model_step_stride",
        "temporal_resampling_id",
        "model_step_seconds",
        "action_interval_seconds",
    )
    missing_contract = [name for name in required_contract if name not in checkpoint]
    if missing_contract:
        raise ValueError(
            "Stage-2 checkpoint is missing the unified contract: " f"{missing_contract}"
        )
    if checkpoint["feature_contract_id"] != FEATURE_CONTRACT_ID:
        raise ValueError("Stage-2 feature contract does not match the current loader")
    if checkpoint["action_contract_id"] != ACTION_CONTRACT_ID:
        raise ValueError("Stage-2 action contract does not match the current loader")
    if checkpoint["actuator_model"] != ACTUATOR_MODEL_ID:
        raise ValueError("Stage-2 actuator model is not the retained modal contract")
    if tuple(checkpoint.get("action_labels", ())) != ACTION_LABELS:
        raise ValueError("Stage-2 action labels do not match the modal contract")
    recorder_rate = float(checkpoint["recorder_sample_rate_hz"])
    model_rate = float(checkpoint["sample_rate_hz"])
    stride = float(checkpoint["model_step_stride"])
    if stride <= 0.0 or recorder_rate <= 0.0 or model_rate <= 0.0:
        raise ValueError("Stage-2 checkpoint has invalid sampling rates")
    expected_step = 1.0 / model_rate
    if not math.isclose(model_rate, recorder_rate / stride, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Stage-2 model rate does not match recorder stride")
    if checkpoint["temporal_resampling_id"] != TEMPORAL_RESAMPLING_ID:
        raise ValueError("Stage-2 temporal resampling contract is unsupported")
    if not math.isclose(
        float(checkpoint["model_step_seconds"]),
        expected_step,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ) or not math.isclose(
        float(checkpoint["action_interval_seconds"]),
        expected_step,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("Stage-2 action interval does not match model rate")
    stage2_configuration = dict(checkpoint["stage2_configuration"])
    # Remove the short-lived duplicated potential-limit field. The retained
    # contract derives it from dt and the physical rate limit.
    stage2_configuration.pop("latent_response_potential_limit", None)
    stage2_configuration.pop("latent_control_authority_loss_weight", None)
    stage2_configuration.pop("latent_control_authority_minimum_ratio", None)
    # Historical successful response checkpoints decoded every z_hat relative
    # to the fixed current z_t as a per-step rate. Preserve that exact behavior
    # instead of silently applying the response-potential phase convention.
    if (
        stage2_configuration.get("future_latent_response_enabled", False)
        and "latent_response_feature" not in stage2_configuration
    ):
        stage2_configuration["latent_response_feature"] = (
            "legacy_cumulative_from_context"
        )
    backbone = PredictiveStateJEPA(
        Stage1Config(**checkpoint["stage1_model_configuration"])
    )
    model = PhysicsIntegratedStage2(
        backbone,
        Standardizer.from_dict(checkpoint["standardizer"]),
        sample_rate_hz=float(checkpoint["sample_rate_hz"]),
        config=Stage2Config(**stage2_configuration),
        representation_training_mode=str(
            checkpoint.get(
                "representation_training_mode", "predictive_pretrained_frozen"
            )
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(target_device)
    return model, checkpoint


def _assert_same_standardizer(rebuilt: Standardizer, checkpoint_value: object) -> None:
    if not isinstance(checkpoint_value, Mapping):
        raise TypeError("checkpoint standardizer must be a mapping")
    current = rebuilt.as_dict()
    if set(current) != set(checkpoint_value):
        raise ValueError("Stage-1 and Stage-2 standardizer fields do not match")
    for name, current_value in current.items():
        expected = np.asarray(checkpoint_value[name])
        actual = np.asarray(current_value)
        if expected.shape != actual.shape or not np.allclose(
            actual, expected, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"Stage-1 and Stage-2 standardizers differ for {name!r}")


def _datasets(
    manifest_path: str,
    checkpoint: Mapping[str, object],
    *,
    train_stride: int,
    evaluation_stride: int,
) -> tuple[
    dict[str, list[object]],
    dict[str, MotorJepaWindowDataset],
    Standardizer,
    float,
    int,
    int,
]:
    archived_partitions = {
        name: load_partition_sequences(manifest_path, name)
        for name in ("train", "validation", "test")
    }
    recorder_rate = common_sample_rate(sum(archived_partitions.values(), []))
    rate = float(checkpoint["sample_rate_hz"])
    partitions = {
        name: resample_episode_sequences(episodes, rate)
        for name, episodes in archived_partitions.items()
    }
    # Stage 2 consumes a frozen Stage-1 representation. Its feature scale is
    # therefore part of the Stage-1 checkpoint contract, not a statistic to
    # refit on an arbitrary Stage-2 subset. Re-fitting here made legitimate
    # subset smoke experiments either fail this assertion or silently change
    # the meaning of every frozen encoder input.
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    history_steps = int(checkpoint["history_steps"])
    horizon_steps = int(checkpoint["horizon_steps"])
    if not math.isclose(
        float(checkpoint["model_step_stride"]),
        recorder_rate / rate,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("dataset model rate differs from Stage-1 checkpoint")
    if checkpoint.get("temporal_resampling_id") != TEMPORAL_RESAMPLING_ID:
        raise ValueError("dataset temporal resampling contract differs from Stage 1")
    if not math.isclose(
        float(checkpoint["recorder_sample_rate_hz"]),
        recorder_rate,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("dataset recorder rate differs from Stage-1 checkpoint")
    datasets = {
        name: MotorJepaWindowDataset(
            episodes,
            standardizer,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            model_step_stride=1,
            stride=train_stride if name == "train" else evaluation_stride,
        )
        for name, episodes in partitions.items()
    }
    return partitions, datasets, standardizer, rate, history_steps, horizon_steps


def _loader(
    dataset: MotorJepaWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
    action_focused_fraction: float = 0.0,
) -> DataLoader:
    if not 0.0 <= action_focused_fraction <= 1.0:
        raise ValueError("action-focused sampling fraction must be in [0, 1]")
    sampler = None
    if action_focused_fraction > 0.0:
        weights = dataset.action_sampling_weights(
            focused_fraction=action_focused_fraction
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _training_readiness(
    validation: Mapping[str, object],
    conditioning: Mapping[str, object],
    derivative_audit: Mapping[str, object],
    *,
    future_latent_response_enabled: bool = False,
) -> dict[str, object]:
    cross_episode_action = conditioning["cross_episode_action"]
    matched_cross_episode_action = conditioning["matched_cross_episode_action"]
    local_action_shift = conditioning["local_action_shift"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            cross_episode_action,
            matched_cross_episode_action,
            local_action_shift,
        )
    ):
        raise TypeError("conditioning audit is malformed")
    forecast_checks = {
        "beats_nominal_physics": float(
            validation["relative_improvement_over_nominal_physics"]
        )
        > 0.0,
        "majority_beats_nominal_physics": float(
            validation["model_better_than_nominal_physics_fraction"]
        )
        > 0.5,
        "beats_persistence": float(validation["relative_improvement_over_persistence"])
        > 0.0,
        "majority_beats_persistence": float(
            validation["model_better_than_persistence_fraction"]
        )
        > 0.5,
        "finite_action_sensitivity": bool(derivative_audit["all_finite"]),
        "nonzero_action_sensitivity": float(
            derivative_audit["first_checked_horizon_response_jacobian_l1"]
        )
        > 1e-6,
        "nonzero_learned_correction_action_sensitivity": float(
            derivative_audit[
                "first_checked_horizon_learned_correction_rate_jacobian_l1"
            ]
        )
        > 1e-6,
        "controlled_axis_direct_signs": bool(
            derivative_audit["all_checked_horizon_direct_signs_positive"]
        ),
        "bounded_off_axis_jacobian": float(
            derivative_audit["maximum_checked_horizon_off_axis_fraction"]
        )
        <= 0.40,
    }
    conditioning_diagnostics = {
        "held_action_factual_better_fraction": float(
            conditioning["held_action"]["factual_better_fraction"]
        ),
        "cross_episode_action_factual_better_fraction": float(
            cross_episode_action["factual_better_fraction"]
        ),
        "matched_cross_episode_action_factual_better_fraction": float(
            matched_cross_episode_action["factual_better_fraction"]
        ),
        "temporal_action_shift_factual_better_fraction": float(
            local_action_shift["factual_better_fraction"]
        ),
        "action_replacement_metrics_are_timing_and_conditioning_diagnostics_only": True,
        "action_replacement_metrics_used_for_training": False,
        "action_replacement_metrics_used_for_checkpoint_selection": False,
        "action_replacement_metrics_used_for_readiness": False,
        "candidate_action_enters_physical_transition_directly": True,
        "domain_belief_fixed_over_imagined_horizon": True,
        "state_latent_action_conditioned_over_imagined_horizon": (
            future_latent_response_enabled
        ),
        "candidate_action_enters_physical_transition_via_latent": (
            future_latent_response_enabled
        ),
    }
    # Action replacements have no counterfactual target and therefore cannot
    # gate a factual world model. Readiness uses held-out factual rollouts and
    # structural derivatives; closed-loop tracking remains the deployment
    # criterion.
    required_checks = tuple(forecast_checks)
    return {
        "passed": all(forecast_checks[name] for name in required_checks),
        "checks": forecast_checks,
        "required_checks": list(required_checks),
        "conditioning_diagnostics": conditioning_diagnostics,
        "conditioning_diagnostics_are_counterfactual_claims": False,
        "control_derivative_audit_is_counterfactual_claim": False,
        "warnings": {},
        "thresholds": {
            "minimum_relative_improvement_over_nominal_physics": 0.0,
            "minimum_model_better_than_nominal_physics_fraction": 0.5,
            "minimum_relative_improvement_over_persistence": 0.0,
            "minimum_model_better_fraction": 0.5,
            "minimum_learned_correction_rate_jacobian_l1": 1e-6,
            "minimum_checked_horizon_response_jacobian_l1": 1e-6,
            "maximum_checked_horizon_off_axis_absolute_fraction": 0.40,
        },
    }


def _factual_rollout_selection_score(metrics: Mapping[str, object]) -> float:
    """Select checkpoints only by held-out factual physical rollout error."""
    factual = float(metrics["normalized_state_mse"])
    if not math.isfinite(factual) or factual < 0.0:
        raise ValueError("validation factual rollout MSE must be nonnegative")
    return factual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--config", default="")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--evaluation-stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--probe-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=0.10)
    parser.add_argument("--yaw-loss-weight", type=float, default=1.0)
    parser.add_argument("--position-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--future-latent-response",
        action="store_true",
        help=(
            "decode cumulative candidate latent responses into bounded surge, "
            "sway, heave, and yaw potentials and inject temporal increments"
        ),
    )
    parser.add_argument(
        "--latent-response-magnitude-loss-weight", type=float, default=0.002
    )
    parser.add_argument(
        "--latent-response-smoothness-loss-weight", type=float, default=0.005
    )
    parser.add_argument(
        "--representation-training-mode",
        choices=REPRESENTATION_TRAINING_MODES,
        default="predictive_pretrained_frozen",
        help=(
            "Use the frozen predictive representation, fine-tune it using only "
            "factual physical rollouts, or discard its weights and train the "
            "same encoder from scratch with that downstream objective."
        ),
    )
    parser.add_argument("--action-focused-sampling-fraction", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260805)
    config_argument, _ = parser.parse_known_args(argv)
    training_config = _apply_training_config(parser, config_argument.config)
    args = parser.parse_args(argv)
    if (
        args.epochs < 1
        or args.batch_size < 1
        or min(args.train_stride, args.evaluation_stride) < 1
    ):
        parser.error("epochs, batch size, and strides must be positive")
    if args.probe_hidden_dim < 1 or args.num_workers < 0:
        parser.error("probe hidden dim must be positive and workers non-negative")
    if (
        min(
            args.learning_rate,
            args.grad_clip_norm,
        )
        <= 0.0
        or args.latent_loss_weight < 0.0
        or args.yaw_loss_weight < 0.0
        or args.position_loss_weight < 0.0
        or args.latent_response_magnitude_loss_weight < 0.0
        or args.latent_response_smoothness_loss_weight < 0.0
        or args.weight_decay < 0.0
    ):
        parser.error("invalid Stage-2 optimizer or loss configuration")
    if not 0.0 <= args.action_focused_sampling_fraction <= 1.0:
        parser.error("action-focused sampling fraction must be in [0, 1]")

    _seed_everything(args.seed)
    device = _device(args.device)
    backbone, stage1_checkpoint = _load_stage1_checkpoint(
        args.stage1_checkpoint, device
    )
    partitions, datasets, standardizer, rate, history_steps, horizon_steps = _datasets(
        args.manifest,
        stage1_checkpoint,
        train_stride=args.train_stride,
        evaluation_stride=args.evaluation_stride,
    )
    if horizon_steps != backbone.config.horizon_steps:
        raise ValueError("checkpoint horizon metadata differs from model configuration")

    predictive_pretraining_used = (
        args.representation_training_mode != "supervised_from_scratch"
    )
    representation_frozen = (
        args.representation_training_mode == "predictive_pretrained_frozen"
    )
    if not predictive_pretraining_used:
        backbone = _random_backbone_from_contract(
            backbone.config,
            seed=args.seed + 1,
            device=device,
        )

    # Both trainable-representation variants use only downstream factual
    # dynamics supervision. This makes pretrained initialization the sole
    # difference between the matched fine-tuned and scratch ablations.
    effective_latent_loss_weight = (
        args.latent_loss_weight
        if representation_frozen and not args.future_latent_response
        else 0.0
    )

    config = Stage2Config(
        probe_hidden_dim=args.probe_hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        latent_loss_weight=effective_latent_loss_weight,
        yaw_loss_weight=args.yaw_loss_weight,
        position_loss_weight=args.position_loss_weight,
        future_latent_response_enabled=args.future_latent_response,
        latent_response_magnitude_loss_weight=(
            args.latent_response_magnitude_loss_weight
        ),
        latent_response_smoothness_loss_weight=(
            args.latent_response_smoothness_loss_weight
        ),
    )
    model = PhysicsIntegratedStage2(
        backbone,
        standardizer,
        sample_rate_hz=rate,
        config=config,
        representation_training_mode=args.representation_training_mode,
    ).to(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Stage-2 has no trainable parameters")
    frozen_modules = (
        model.backbone.dynamics_context_encoder,
        model.backbone.dynamics_context_normalization,
        model.backbone.observation_encoder,
        model.backbone.target_observation_encoder,
    )
    supervised_modules = model._supervised_representation_modules()
    if representation_frozen:
        if any(
            parameter.requires_grad
            for module in frozen_modules
            for parameter in module.parameters()
        ):
            raise RuntimeError(
                "predictively pretrained observation/history representation "
                "must remain frozen"
            )
    elif not all(
        parameter.requires_grad
        for module in supervised_modules
        for parameter in module.parameters()
    ):
        raise RuntimeError(
            "scratch observation/history representation must remain trainable"
        )
    if any(
        parameter.requires_grad
        for parameter in model.backbone.target_observation_encoder.parameters()
    ):
        raise RuntimeError("target observation encoder must remain frozen")
    transition_parameters = tuple(
        parameter
        for module in model._trainable_transition_modules()
        for parameter in module.parameters()
    )
    transition_frozen_for_response = (
        representation_frozen and config.future_latent_response_enabled
    )
    if transition_frozen_for_response:
        if any(parameter.requires_grad for parameter in transition_parameters):
            raise RuntimeError(
                "pretrained z_hat transition must remain frozen in response mode"
            )
    elif not all(parameter.requires_grad for parameter in transition_parameters):
        raise RuntimeError("Stage-2 action transition must remain trainable")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        num_workers=args.num_workers,
        action_focused_fraction=args.action_focused_sampling_fraction,
    )
    validation_loader = _loader(
        datasets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )
    test_loader = _loader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_value = math.inf
    best_model_state: dict[str, Tensor] | None = None
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device)
        validation_metrics = evaluate(model, validation_loader, device=device)
        validation_value = _factual_rollout_selection_score(validation_metrics)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.6f} "
            f"validation_selection={validation_value:.6f} "
            f"state_mse={float(validation_metrics['normalized_state_mse']):.6f} "
            f"vs_nominal={float(validation_metrics['relative_improvement_over_nominal_physics']):+.2%} "
            f"vs_persistence={float(validation_metrics['relative_improvement_over_persistence']):+.2%}",
            flush=True,
        )
        if validation_value < best_value:
            best_value = validation_value
            best_model_state = deepcopy(model.state_dict())
    if best_model_state is None:
        raise RuntimeError("Stage 2 produced no model checkpoint")
    model.load_state_dict(best_model_state, strict=True)

    selected_validation = evaluate(model, validation_loader, device=device)
    validation_conditioning = audit_conditioning(
        model, validation_loader, standardizer, device=device
    )
    test_metrics = evaluate(model, test_loader, device=device)
    test_conditioning = audit_conditioning(
        model, test_loader, standardizer, device=device
    )
    validation_derivatives = audit_control_derivatives(
        model, validation_loader, device=device
    )
    test_derivatives = audit_control_derivatives(model, test_loader, device=device)
    readiness = _training_readiness(
        selected_validation,
        validation_conditioning,
        validation_derivatives,
        future_latent_response_enabled=config.future_latent_response_enabled,
    )
    created_at = datetime.now().astimezone().isoformat()
    checkpoint_path = output_dir / "stage2_physics_probe.pt"
    checkpoint = {
        "implementation": IMPLEMENTATION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "observation_labels": list(OBSERVATION_LABELS),
        "action_labels": list(ACTION_LABELS),
        "created_at": created_at,
        "stage": 2,
        "stage1_checkpoint_path": str(
            Path(args.stage1_checkpoint).expanduser().resolve()
        ),
        "stage1_implementation": stage1_checkpoint.get("implementation"),
        "stage1_training_readiness": stage1_checkpoint.get("training_readiness"),
        "training_config": training_config,
        "representation_training_mode": args.representation_training_mode,
        "predictive_pretraining_used": predictive_pretraining_used,
        "representation_frozen": representation_frozen,
        "representation_downstream_objective": (
            "frozen_predictive_representation_plus_physical_and_latent_losses"
            if representation_frozen
            else "factual_physical_rollout_only"
        ),
        "representation_initialization": (
            "stage1_predictive_checkpoint"
            if predictive_pretraining_used
            else f"random_seed_{args.seed + 1}"
        ),
        "requested_latent_loss_weight": args.latent_loss_weight,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "stage1_model_configuration": asdict(backbone.config),
        "stage2_configuration": asdict(config),
        "model_state_dict": model.state_dict(),
        "probe_state_dict": model.probe.state_dict(),
        "standardizer": standardizer.as_dict(),
        "manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "sample_rate_hz": rate,
        "recorder_sample_rate_hz": stage1_checkpoint["recorder_sample_rate_hz"],
        "model_step_stride": stage1_checkpoint["model_step_stride"],
        "temporal_resampling_id": stage1_checkpoint["temporal_resampling_id"],
        "model_step_seconds": 1.0 / rate,
        "action_interval_seconds": 1.0 / rate,
        "action_contract_id": ACTION_CONTRACT_ID,
        "actuator_model": ACTUATOR_MODEL_ID,
        "selected_validation_metrics": selected_validation,
        "validation_conditioning_audit": validation_conditioning,
        "test_metrics": test_metrics,
        "test_conditioning_audit": test_conditioning,
        "validation_control_derivative_audit": validation_derivatives,
        "test_control_derivative_audit": test_derivatives,
        "training_readiness": readiness,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key
        not in {
            "model_state_dict",
            "probe_state_dict",
        }
    }
    report.update(
        {
            "objective": (
                "held_out_factual_multi_step_trajectory_and_near_term_rate_"
                "decoding_with_context_domain_correction_and_action_"
                "conditioned_residual_dynamics"
            ),
            "evaluation_protocol": {
                "checkpoint_selection": "held_out_factual_rollout",
                "reported_primary_metrics": [
                    "rollout_fidelity_at_seconds",
                    "per_horizon_state_rmse",
                    "per_step_state_rmse_growth",
                    "physical_rmse",
                ],
                "local_action_replacement_role": "timing_and_conditioning_diagnostic_only",
                "local_action_replacement_is_not_counterfactual_ground_truth": True,
                "closed_loop_tracking_is_deployment_criterion": True,
            },
            "dynamics_context_encoder_frozen": representation_frozen,
            "observation_encoder_frozen": representation_frozen,
            "target_observation_encoder_frozen": True,
            "observation_representation_frozen": representation_frozen,
            "nominal_dynamics_frozen": True,
            "residual_probe_bounded": True,
            "action_transition_trainable": not transition_frozen_for_response,
            "candidate_action_transition": (
                "fixed_history_belief_then_context_corrected_first_order_"
                "actuator_and_nominal_hydrodynamics_plus_bounded_residual"
            ),
            "candidate_action_latent_used_by_physical_rollout": (
                config.future_latent_response_enabled
            ),
            "domain_belief_fixed_over_imagined_horizon": True,
            "state_latent_action_conditioned_over_imagined_horizon": (
                config.future_latent_response_enabled
            ),
            "stage2_loss_terms": {
                "factual_near_term_rate": config.rate_loss_weight,
                "factual_near_term_rate_extra_weight": config.near_horizon_rate_loss_weight,
                "multi_step_physical_rollout": config.trajectory_loss_weight,
                "terminal_trajectory": config.terminal_trajectory_loss_weight,
                "domain_parameter_deviation": config.domain_parameter_loss_weight,
                "action_conditioned_future_latent": config.latent_loss_weight,
                "latent_response_magnitude": (
                    config.latent_response_magnitude_loss_weight
                ),
                "latent_response_smoothness": (
                    config.latent_response_smoothness_loss_weight
                ),
            },
            "future_latent_role": (
                "candidate_four_modal_response_potential_from_z_hat_minus_z_t_"
                "with_temporal_increment"
                if config.future_latent_response_enabled
                else "self_supervised_action_prediction_auxiliary"
            ),
            "actuator_model": {
                "type": "first_order",
                "time_constant_seconds": config.actuator_time_constant_seconds,
                "terminal_state_update": (
                    "a_end=a_start+(1-exp(-dt/tau))*(a_command-a_start)"
                ),
                "plant_interval_action": (
                    "a_mean=a_start+"
                    "(1-(tau/dt)*(1-exp(-dt/tau)))*(a_command-a_start)"
                ),
                "initial_state": "history_reconstructed_modal_action",
            },
            "physics_rate_structure": (
                "fixed_nominal_prior_plus_bounded_context_conditioned_"
                "M_control_D_and_actuator_corrections_plus_bounded_"
                "state_context_bias_and_6x5_action_wrench_residual"
            ),
            "observation_acceleration_contract": {
                "feature": "gravity_compensated_dynamic_body_acceleration_m_s2",
                "archive_source": "imu_linear_acceleration_body_frd_m_s2_specific_force",
                "conversion": "a_body = specific_force_body + R_body_to_ned.T @ [0,0,g]",
            },
            "nominal_hydrodynamic_prior": {
                "equation": (
                    "M nu_dot = tau_control - C(nu)nu - D1 nu "
                    "- D2 abs(nu)nu + tau_b + delta_tau"
                ),
                "learned_term": (
                    "bounded_context_state_bias_plus_bounded_6x5_action_"
                    "wrench_residual"
                ),
                "nominal_parameters_trainable": False,
                "context_domain_correction": {
                    "mass_scale_limit": math.exp(config.domain_mass_log_scale_limit),
                    "control_scale_limit": math.exp(
                        config.domain_control_log_scale_limit
                    ),
                    "damping_scale_limit": math.exp(
                        config.domain_damping_log_scale_limit
                    ),
                    "actuator_time_constant_scale_limit": math.exp(
                        config.domain_actuator_log_scale_limit
                    ),
                    "fixed_over_imagined_horizon": True,
                    "interpreted_as_identified_physical_truth": False,
                },
                "residual_wrench_limit": list(config.residual_wrench_limit),
                "residual_action_wrench_limit": list(
                    config.residual_action_wrench_limit
                ),
            },
            "control_direction_prior": {
                "controlled_axes": ["surge", "sway", "heave", "yaw"],
                "known_rate_signs": [1, 1, -1, 1],
                "roll_is_readiness_axis": False,
            },
            "primary_physical_targets": [
                "position_delta_ned",
                "body_velocity_xyz",
                "wrapped_yaw_delta",
                "yaw_rate",
            ],
            "low_weight_safety_targets": [
                "roll_pitch_orientation",
                "roll_pitch_rate",
            ],
            "action_focused_sampling_fraction": (args.action_focused_sampling_fraction),
            "episode_counts": {name: len(value) for name, value in partitions.items()},
            "window_counts": {name: len(value) for name, value in datasets.items()},
            "history": history,
            "best_validation_factual_rollout_mse": best_value,
            "checkpoint_selection": {
                "metric": "validation_metrics.normalized_state_mse",
                "source": "held_out_factual_multi_step_rollout",
                "action_replacement_metrics_used": False,
                "closed_loop_tracking_required_for_deployment": True,
            },
            "checkpoint_path": str(checkpoint_path),
        }
    )
    report_path = output_dir / "stage2_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(checkpoint_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
