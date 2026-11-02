"""JEPA-conditioned structured 4-DoF underwater dynamics.

This module is an independent alternative to ``learning.stage2``.  It does
not decode an unconstrained future trajectory and it does not imitate the
collection NMPC.  A frozen Stage-1 history belief instead selects a compact
set of coefficients for a control-oriented model:

    motor lag -> linear/signed-square thruster allocation
              -> context-conditioned control gain
              -> linear/quadratic body-rate dynamics
              -> exact planar pose kinematics

The coefficient set is fixed for the complete imagined horizon.  Candidate
actions therefore have a differentiable, structurally enforced path to the
predicted trajectory, while hidden drag, buoyancy/current bias, actuator lag,
and small cross-axis coupling can adapt from causal observation/action history.
No physics-domain labels or ground-truth state are consumed.
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from disaware_learning.learning.motor_jepa_data import (
    ACTION_CONTRACT_ID,
    ACTION_LABELS,
    FEATURE_CONTRACT_ID,
    IMU_GYRO_SLICE,
    MotorJepaWindowDataset,
    OBSERVATION_LABELS,
    ROTATION_6D_SLICE,
    Standardizer,
    VIO_VELOCITY_SLICE,
    common_sample_rate,
    load_partition_sequences,
)
from disaware_learning.learning.stage1 import (
    IMPLEMENTATION as STAGE1_IMPLEMENTATION,
    PredictiveStateJEPA,
    Stage1Config,
)


IMPLEMENTATION = "jepa_structured_4dof_hydrodynamics_v1"
MOTION_ORDER = ("surge_velocity", "sway_velocity", "heave_velocity_down", "yaw_rate")
CONTROLLED_ACTION_INDICES = (0, 1, 2, 3)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return result


def _positive_tuple(values: object, width: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {width} positive values") from error
    if (
        len(result) != width
        or not all(math.isfinite(value) and value > 0.0 for value in result)
    ):
        raise ValueError(f"{name} must contain {width} positive values")
    return result


def _nonnegative_tuple(values: object, width: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {width} nonnegative values") from error
    if (
        len(result) != width
        or not all(math.isfinite(value) and value >= 0.0 for value in result)
    ):
        raise ValueError(f"{name} must contain {width} nonnegative values")
    return result


def _wrap_angle(values: Tensor) -> Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


def yaw_from_rotation6d(values: Tensor) -> Tensor:
    """Extract yaw from the first body-to-NED rotation column."""
    if values.shape[-1] != 6:
        raise ValueError("rotation6d must contain six channels")
    return torch.atan2(values[..., 1], values[..., 0])


def modal_to_motor(values: Tensor) -> Tensor:
    """Reconstruct six BlueROV2 motor commands from five modal coordinates."""
    if values.shape[-1] != len(ACTION_LABELS):
        raise ValueError("modal action has the wrong channel count")
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


def motor_to_modal(values: Tensor) -> Tensor:
    """Project six motor values onto the standard five-dimensional manifold."""
    if values.shape[-1] != 6:
        raise ValueError("motor action must contain six channels")
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


def allocated_thruster_features(motor: Tensor) -> tuple[Tensor, Tensor]:
    """Return linear and signed-square 4-DoF generalized thrust features."""
    linear = motor_to_modal(motor)[..., :4]
    quadratic = motor_to_modal(torch.sign(motor) * motor.square())[..., :4]
    return linear, quadratic


@dataclass(frozen=True)
class StructuredDynamicsConfig:
    hidden_dim: int = 96
    # Initial values are broad physical scales, not fixed Stonefish labels.
    initial_linear_drag: tuple[float, ...] = (0.43, 0.50, 0.32, 0.88)
    initial_quadratic_drag: tuple[float, ...] = (0.40, 0.40, 0.25, 0.80)
    initial_linear_control_gain: tuple[float, ...] = (0.18, 0.22, 0.16, 0.05)
    initial_quadratic_control_gain: tuple[float, ...] = (0.20, 0.20, 0.15, 3.20)
    initial_actuator_time_constant_seconds: tuple[float, ...] = (
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
    )
    drag_log_scale_limit: float = math.log(2.0)
    control_log_scale_limit: float = math.log(2.0)
    actuator_log_scale_limit: float = math.log(3.0)
    acceleration_bias_limit: tuple[float, ...] = (0.12, 0.12, 0.08, 0.40)
    coupling_limit: float = 0.20
    velocity_error_scale: tuple[float, ...] = (0.10, 0.10, 0.08)
    yaw_rate_error_scale: float = 0.10
    position_error_scale: tuple[float, ...] = (0.30, 0.30, 0.20)
    yaw_error_scale: float = 0.20
    rate_error_scale: tuple[float, ...] = (0.20, 0.20, 0.15, 0.30)
    velocity_loss_weight: float = 1.0
    yaw_rate_loss_weight: float = 1.0
    position_loss_weight: float = 1.0
    yaw_loss_weight: float = 0.5
    rate_loss_weight: float = 0.10
    parameter_regularization_weight: float = 0.01
    base_parameter_regularization_weight: float = 0.001
    near_horizon_weight: float = 2.0
    terminal_weight: float = 0.5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        positive_fours = (
            "initial_linear_drag",
            "initial_quadratic_drag",
            "initial_linear_control_gain",
            "initial_quadratic_control_gain",
            "acceleration_bias_limit",
            "rate_error_scale",
        )
        positive_threes = ("velocity_error_scale", "position_error_scale")
        for name in positive_fours:
            object.__setattr__(self, name, _positive_tuple(getattr(self, name), 4, name))
        for name in positive_threes:
            object.__setattr__(self, name, _positive_tuple(getattr(self, name), 3, name))
        object.__setattr__(
            self,
            "initial_actuator_time_constant_seconds",
            _positive_tuple(
                self.initial_actuator_time_constant_seconds,
                6,
                "initial_actuator_time_constant_seconds",
            ),
        )
        positive_scalars = (
            "drag_log_scale_limit",
            "control_log_scale_limit",
            "actuator_log_scale_limit",
            "yaw_rate_error_scale",
            "yaw_error_scale",
            "velocity_loss_weight",
            "yaw_rate_loss_weight",
            "position_loss_weight",
            "yaw_loss_weight",
            "near_horizon_weight",
            "terminal_weight",
            "learning_rate",
            "grad_clip_norm",
        )
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be at least eight")
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in positive_scalars
        ):
            raise ValueError("structured dynamics scales must be positive and finite")
        if not 0.0 <= self.coupling_limit < 1.0:
            raise ValueError("coupling_limit must be in [0, 1)")
        for name in (
            "rate_loss_weight",
            "parameter_regularization_weight",
            "base_parameter_regularization_weight",
            "weight_decay",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.near_horizon_weight < 1.0:
            raise ValueError("near_horizon_weight must be at least one")


class StructuredCoefficientDecoder(nn.Module):
    """Decode one fixed set of hydrodynamic coefficients per history belief."""

    def __init__(self, belief_dim: int, config: StructuredDynamicsConfig) -> None:
        super().__init__()
        if belief_dim < 2:
            raise ValueError("belief_dim must be at least two")
        self.belief_dim = int(belief_dim)
        self.config = config
        # Four drag + four quadratic drag + eight control scales + six motor
        # time constants + four drift terms + twelve off-diagonal couplings.
        self.output_dim = 4 + 4 + 4 + 4 + 6 + 4 + 12
        self.network = nn.Sequential(
            nn.LayerNorm(belief_dim),
            nn.Linear(belief_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self.output_dim),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("coefficient decoder must end with a linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self.base_log_linear_drag = nn.Parameter(
            torch.log(torch.tensor(config.initial_linear_drag, dtype=torch.float32))
        )
        self.base_log_quadratic_drag = nn.Parameter(
            torch.log(torch.tensor(config.initial_quadratic_drag, dtype=torch.float32))
        )
        self.base_log_linear_control = nn.Parameter(
            torch.log(
                torch.tensor(config.initial_linear_control_gain, dtype=torch.float32)
            )
        )
        self.base_log_quadratic_control = nn.Parameter(
            torch.log(
                torch.tensor(config.initial_quadratic_control_gain, dtype=torch.float32)
            )
        )
        self.base_log_actuator_tau = nn.Parameter(
            torch.log(
                torch.tensor(
                    config.initial_actuator_time_constant_seconds,
                    dtype=torch.float32,
                )
            )
        )
        self.register_buffer(
            "initial_base_log_linear_drag", self.base_log_linear_drag.detach().clone()
        )
        self.register_buffer(
            "initial_base_log_quadratic_drag",
            self.base_log_quadratic_drag.detach().clone(),
        )
        self.register_buffer(
            "initial_base_log_linear_control",
            self.base_log_linear_control.detach().clone(),
        )
        self.register_buffer(
            "initial_base_log_quadratic_control",
            self.base_log_quadratic_control.detach().clone(),
        )
        self.register_buffer(
            "initial_base_log_actuator_tau",
            self.base_log_actuator_tau.detach().clone(),
        )
        self.register_buffer(
            "bias_limit",
            torch.tensor(config.acceleration_bias_limit, dtype=torch.float32),
        )
        basis = torch.zeros(12, 4, 4, dtype=torch.float32)
        index = 0
        for row in range(4):
            for column in range(4):
                if row != column:
                    basis[index, row, column] = 1.0
                    index += 1
        self.register_buffer("off_diagonal_basis", basis)

    def forward(self, belief: Tensor) -> dict[str, Tensor]:
        if belief.ndim != 2 or belief.shape[-1] != self.belief_dim:
            raise ValueError("history belief has the wrong shape")
        normalized = torch.tanh(self.network(belief))
        sections = torch.split(normalized, (4, 4, 4, 4, 6, 4, 12), dim=-1)
        drag_linear, drag_quadratic, control_linear, control_quadratic, tau, bias, coupling = sections
        cfg = self.config
        batch = belief.shape[0]
        identity = torch.eye(4, dtype=belief.dtype, device=belief.device).expand(
            batch, -1, -1
        )
        coupling_matrix = identity + cfg.coupling_limit * torch.einsum(
            "bk,kij->bij", coupling, self.off_diagonal_basis
        )
        return {
            "linear_drag": torch.exp(
                self.base_log_linear_drag + cfg.drag_log_scale_limit * drag_linear
            ),
            "quadratic_drag": torch.exp(
                self.base_log_quadratic_drag
                + cfg.drag_log_scale_limit * drag_quadratic
            ),
            "linear_control_gain": torch.exp(
                self.base_log_linear_control
                + cfg.control_log_scale_limit * control_linear
            ),
            "quadratic_control_gain": torch.exp(
                self.base_log_quadratic_control
                + cfg.control_log_scale_limit * control_quadratic
            ),
            "actuator_time_constant_seconds": torch.exp(
                self.base_log_actuator_tau
                + cfg.actuator_log_scale_limit * tau
            ),
            "acceleration_bias": bias * self.bias_limit,
            "coupling_matrix": coupling_matrix,
            "normalized_deviation": normalized,
        }

    def base_parameter_regularization(self) -> Tensor:
        values = (
            (self.base_log_linear_drag, self.initial_base_log_linear_drag),
            (self.base_log_quadratic_drag, self.initial_base_log_quadratic_drag),
            (self.base_log_linear_control, self.initial_base_log_linear_control),
            (
                self.base_log_quadratic_control,
                self.initial_base_log_quadratic_control,
            ),
            (self.base_log_actuator_tau, self.initial_base_log_actuator_tau),
        )
        # Penalize movement from the broad initializer, rather than pulling
        # every coefficient toward one in physical units.
        return torch.stack(
            tuple((value - initial).square().mean() for value, initial in values)
        ).mean()


class JepaStructuredDynamics(nn.Module):
    """Frozen JEPA belief plus structured 4-DoF hydrodynamic rollout."""

    def __init__(
        self,
        backbone: PredictiveStateJEPA,
        standardizer: Standardizer,
        *,
        sample_rate_hz: float,
        config: StructuredDynamicsConfig = StructuredDynamicsConfig(),
    ) -> None:
        super().__init__()
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")
        self.backbone = backbone
        self.config = config
        self.sample_rate_hz = float(sample_rate_hz)
        self.coefficients = StructuredCoefficientDecoder(
            backbone.config.belief_dim,
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
        self.register_buffer(
            "control_direction",
            torch.tensor((1.0, 1.0, -1.0, 1.0), dtype=torch.float32),
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    def train(self, mode: bool = True) -> "JepaStructuredDynamics":
        super().train(mode)
        self.backbone.eval()
        return self

    def denormalize_observation(self, values: Tensor) -> Tensor:
        return values * self.observation_std + self.observation_mean

    def denormalize_action(self, values: Tensor) -> Tensor:
        return values * self.action_std + self.action_mean

    def encode_belief(self, observation_history: Tensor, action_history: Tensor) -> Tensor:
        with torch.no_grad():
            context = self.backbone.encode_context(observation_history, action_history)
        return context[:, self.backbone.config.latent_dim :]

    def _rollout_physical(
        self,
        current_observation: Tensor,
        previous_modal_action: Tensor,
        future_modal_action: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        if current_observation.ndim != 2 or current_observation.shape[-1] != len(
            OBSERVATION_LABELS
        ):
            raise ValueError("current physical observation has the wrong shape")
        if previous_modal_action.shape != (len(current_observation), len(ACTION_LABELS)):
            raise ValueError("previous modal action has the wrong shape")
        if (
            future_modal_action.ndim != 3
            or future_modal_action.shape[0] != len(current_observation)
            or future_modal_action.shape[-1] != len(ACTION_LABELS)
        ):
            raise ValueError("future modal action has the wrong shape")

        velocity = current_observation[:, VIO_VELOCITY_SLICE]
        yaw_rate = current_observation[:, IMU_GYRO_SLICE.stop - 1 : IMU_GYRO_SLICE.stop]
        motion = torch.cat((velocity, yaw_rate), dim=-1)
        initial_yaw = yaw_from_rotation6d(current_observation[:, ROTATION_6D_SLICE])
        yaw_delta = torch.zeros_like(initial_yaw)
        position_delta = torch.zeros(
            len(current_observation),
            3,
            dtype=current_observation.dtype,
            device=current_observation.device,
        )
        effective_motor = modal_to_motor(previous_modal_action)
        tau = parameters["actuator_time_constant_seconds"]
        alpha = 1.0 - torch.exp(-self.dt / tau)

        future_motion = []
        future_rate = []
        future_position = []
        future_yaw = []
        future_effective_motor = []
        future_linear_feature = []
        future_quadratic_feature = []
        for step in range(future_modal_action.shape[1]):
            commanded_motor = modal_to_motor(future_modal_action[:, step])
            effective_motor = effective_motor + alpha * (commanded_motor - effective_motor)
            linear_feature, quadratic_feature = allocated_thruster_features(
                effective_motor
            )
            base_control = self.control_direction * (
                parameters["linear_control_gain"] * linear_feature
                + parameters["quadratic_control_gain"] * quadratic_feature
            )
            control = torch.einsum(
                "bij,bj->bi", parameters["coupling_matrix"], base_control
            )
            rate = (
                parameters["acceleration_bias"]
                - parameters["linear_drag"] * motion
                - parameters["quadratic_drag"] * motion.abs() * motion
                + control
            )
            current_yaw = initial_yaw + yaw_delta
            cosine = torch.cos(current_yaw)
            sine = torch.sin(current_yaw)
            world_velocity = torch.stack(
                (
                    cosine * motion[:, 0] - sine * motion[:, 1],
                    sine * motion[:, 0] + cosine * motion[:, 1],
                    motion[:, 2],
                ),
                dim=-1,
            )
            position_delta = position_delta + self.dt * world_velocity
            yaw_delta = yaw_delta + self.dt * motion[:, 3]
            motion = motion + self.dt * rate

            future_motion.append(motion)
            future_rate.append(rate)
            future_position.append(position_delta)
            future_yaw.append(yaw_delta)
            future_effective_motor.append(effective_motor)
            future_linear_feature.append(linear_feature)
            future_quadratic_feature.append(quadratic_feature)
        return {
            "predicted_motion": torch.stack(future_motion, dim=1),
            "predicted_rate": torch.stack(future_rate, dim=1),
            "predicted_position_delta_ned_m": torch.stack(future_position, dim=1),
            "predicted_yaw_delta_rad": torch.stack(future_yaw, dim=1),
            "effective_motor": torch.stack(future_effective_motor, dim=1),
            "linear_thruster_feature": torch.stack(future_linear_feature, dim=1),
            "quadratic_thruster_feature": torch.stack(
                future_quadratic_feature, dim=1
            ),
        }

    def forward(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
        *,
        belief_override: Tensor | None = None,
        zero_context: bool = False,
    ) -> dict[str, Tensor]:
        if observation_history.ndim != 3 or action_history.ndim != 3:
            raise ValueError("history tensors must be [batch, time, channel]")
        if observation_history.shape[:2] != action_history.shape[:2]:
            raise ValueError("observation and action histories must align")
        if future_action.ndim != 3 or future_action.shape[0] != observation_history.shape[0]:
            raise ValueError("future action batch must align with history")
        belief = (
            self.encode_belief(observation_history, action_history)
            if belief_override is None
            else belief_override
        )
        if belief.shape != (
            observation_history.shape[0],
            self.backbone.config.belief_dim,
        ):
            raise ValueError("belief override has the wrong shape")
        if zero_context:
            belief = torch.zeros_like(belief)
        parameters = self.coefficients(belief)
        current = self.denormalize_observation(observation_history[:, -1])
        previous_action = self.denormalize_action(action_history[:, -1])
        physical_future_action = self.denormalize_action(future_action)
        output = self._rollout_physical(
            current,
            previous_action,
            physical_future_action,
            parameters,
        )
        output.update(
            {
                "belief": belief,
                "current_physical_observation": current,
                "physical_future_action": physical_future_action,
                **parameters,
            }
        )
        return output

    def action_rate_jacobian(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
    ) -> Tensor:
        """Return d(first predicted body rate)/d(first physical modal action)."""
        candidate = future_action.detach().clone().requires_grad_(True)
        output = self(observation_history, action_history, candidate)
        rows = []
        for axis in range(4):
            gradient = torch.autograd.grad(
                output["predicted_rate"][:, 0, axis].sum(),
                candidate,
                retain_graph=axis < 3,
                create_graph=False,
            )[0][:, 0]
            # candidate is standardized: dx/du_physical = dx/du_std / std.
            rows.append(gradient / self.action_std)
        return torch.stack(rows, dim=1)


def _horizon_weights(model: JepaStructuredDynamics, steps: int, reference: Tensor) -> Tensor:
    time = torch.arange(1, steps + 1, dtype=reference.dtype, device=reference.device) * model.dt
    near = torch.where(time <= 1.0, model.config.near_horizon_weight, 1.0)
    near[-1] += model.config.terminal_weight
    return near / near.sum()


def structured_loss(
    model: JepaStructuredDynamics,
    output: Mapping[str, Tensor],
    future_observation: Tensor,
    future_position_delta_ned_m: Tensor,
) -> dict[str, Tensor]:
    target = model.denormalize_observation(future_observation)
    target_velocity = target[..., VIO_VELOCITY_SLICE]
    target_yaw_rate = target[..., IMU_GYRO_SLICE.stop - 1]
    target_motion = torch.cat((target_velocity, target_yaw_rate.unsqueeze(-1)), dim=-1)
    predicted_motion = output["predicted_motion"]
    if predicted_motion.shape != target_motion.shape:
        raise ValueError("predicted and target motion shapes differ")
    weights = _horizon_weights(model, predicted_motion.shape[1], predicted_motion)
    cfg = model.config

    velocity_scale = predicted_motion.new_tensor(cfg.velocity_error_scale)
    position_scale = predicted_motion.new_tensor(cfg.position_error_scale)
    rate_scale = predicted_motion.new_tensor(cfg.rate_error_scale)
    velocity_per_step = (
        (predicted_motion[..., :3] - target_velocity) / velocity_scale
    ).square().mean(dim=-1)
    yaw_rate_per_step = (
        (predicted_motion[..., 3] - target_yaw_rate) / cfg.yaw_rate_error_scale
    ).square()
    position_per_step = (
        (
            output["predicted_position_delta_ned_m"]
            - future_position_delta_ned_m
        )
        / position_scale
    ).square().mean(dim=-1)
    current_yaw = yaw_from_rotation6d(
        output["current_physical_observation"][:, ROTATION_6D_SLICE]
    )
    target_yaw = yaw_from_rotation6d(target[..., ROTATION_6D_SLICE])
    target_yaw_delta = _wrap_angle(target_yaw - current_yaw[:, None])
    yaw_per_step = (
        _wrap_angle(output["predicted_yaw_delta_rad"] - target_yaw_delta)
        / cfg.yaw_error_scale
    ).square()

    previous_motion = torch.cat(
        (
            output["current_physical_observation"][:, VIO_VELOCITY_SLICE],
            output["current_physical_observation"][:, IMU_GYRO_SLICE.stop - 1 : IMU_GYRO_SLICE.stop],
        ),
        dim=-1,
    )
    target_previous = torch.cat((previous_motion[:, None], target_motion[:, :-1]), dim=1)
    target_rate = (target_motion - target_previous) / model.dt
    rate_per_step = (
        (output["predicted_rate"] - target_rate) / rate_scale
    ).square().mean(dim=-1)

    def weighted(value: Tensor) -> Tensor:
        return (value.mean(dim=0) * weights).sum()

    velocity = weighted(velocity_per_step)
    yaw_rate = weighted(yaw_rate_per_step)
    position = weighted(position_per_step)
    yaw = weighted(yaw_per_step)
    rate = weighted(rate_per_step)
    parameter_regularization = output["normalized_deviation"].square().mean()
    base_regularization = model.coefficients.base_parameter_regularization()
    total = (
        cfg.velocity_loss_weight * velocity
        + cfg.yaw_rate_loss_weight * yaw_rate
        + cfg.position_loss_weight * position
        + cfg.yaw_loss_weight * yaw
        + cfg.rate_loss_weight * rate
        + cfg.parameter_regularization_weight * parameter_regularization
        + cfg.base_parameter_regularization_weight * base_regularization
    )
    return {
        "total": total,
        "velocity": velocity,
        "yaw_rate": yaw_rate,
        "position": position,
        "yaw": yaw,
        "rate": rate,
        "parameter_regularization": parameter_regularization,
        "base_parameter_regularization": base_regularization,
    }


def _move_batch(raw: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    required = (
        "observation_history",
        "action_history",
        "future_action",
        "future_observation",
        "future_position_delta_ned_m",
        "episode_index",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise KeyError(f"structured dynamics batch is missing: {missing}")
    result = {
        name: raw[name].to(device=device, non_blocking=True)
        for name in required
    }
    for name in required[:-1]:
        result[name] = result[name].float()
    return result


def train_one_epoch(
    model: JepaStructuredDynamics,
    loader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
        )
        losses = structured_loss(
            model,
            output,
            batch["future_observation"],
            batch["future_position_delta_ned_m"],
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("non-finite structured dynamics loss")
        losses["total"].backward()
        clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            model.config.grad_clip_norm,
        )
        optimizer.step()
        batch_size = len(batch["future_action"])
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("empty training loader")
    return {name: value / count for name, value in sums.items()}


def _different_episode_permutation(episode_index: Tensor) -> Tensor | None:
    episodes = episode_index.reshape(-1)
    if len(episodes) < 2 or torch.unique(episodes).numel() < 2:
        return None
    indices = torch.arange(len(episodes), device=episodes.device)
    result = torch.empty_like(indices)
    for row in range(len(indices)):
        candidates = indices[episodes != episodes[row]]
        result[row] = candidates[row % len(candidates)]
    return result


@torch.no_grad()
def evaluate(
    model: JepaStructuredDynamics,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    loss_sums: dict[str, float] = {}
    squared = torch.zeros(8, dtype=torch.float64)
    zero_squared = torch.zeros(8, dtype=torch.float64)
    shuffled_squared = torch.zeros(8, dtype=torch.float64)
    element_count = 0
    shuffled_count = 0
    sample_count = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        output = model(
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
        )
        losses = structured_loss(
            model,
            output,
            batch["future_observation"],
            batch["future_position_delta_ned_m"],
        )
        batch_size = len(batch["future_action"])
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.cpu()) * batch_size
        target = model.denormalize_observation(batch["future_observation"])
        target_motion = torch.cat(
            (
                target[..., VIO_VELOCITY_SLICE],
                target[..., IMU_GYRO_SLICE.stop - 1 : IMU_GYRO_SLICE.stop],
            ),
            dim=-1,
        )

        def errors(value: Mapping[str, Tensor]) -> Tensor:
            motion_error = (value["predicted_motion"] - target_motion).square().sum(dim=(0, 1))
            position_error = (
                value["predicted_position_delta_ned_m"]
                - batch["future_position_delta_ned_m"]
            ).square().sum(dim=(0, 1))
            yaw_target = _wrap_angle(
                yaw_from_rotation6d(target[..., ROTATION_6D_SLICE])
                - yaw_from_rotation6d(
                    value["current_physical_observation"][:, ROTATION_6D_SLICE]
                )[:, None]
            )
            yaw_error = _wrap_angle(
                value["predicted_yaw_delta_rad"] - yaw_target
            ).square().sum().reshape(1)
            return torch.cat((motion_error, position_error, yaw_error)).double().cpu()

        squared += errors(output)
        zero_output = model(
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
            zero_context=True,
        )
        zero_squared += errors(zero_output)
        permutation = _different_episode_permutation(batch["episode_index"])
        if permutation is not None:
            shuffled_output = model(
                batch["observation_history"],
                batch["action_history"],
                batch["future_action"],
                belief_override=output["belief"][permutation],
            )
            shuffled_squared += errors(shuffled_output)
            shuffled_count += batch_size * target_motion.shape[1]
        element_count += batch_size * target_motion.shape[1]
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("empty evaluation loader")
    labels = (*MOTION_ORDER, "position_n", "position_e", "position_d", "yaw")
    rmse = torch.sqrt(squared / element_count).numpy()
    zero_rmse = torch.sqrt(zero_squared / element_count).numpy()
    result: dict[str, object] = {
        "sample_count": sample_count,
        "loss": {name: value / sample_count for name, value in loss_sums.items()},
        "rmse": {name: float(value) for name, value in zip(labels, rmse, strict=True)},
        "zero_context_rmse": {
            name: float(value) for name, value in zip(labels, zero_rmse, strict=True)
        },
        "context_improvement_fraction": {
            name: float((baseline - factual) / max(baseline, 1e-12))
            for name, factual, baseline in zip(labels, rmse, zero_rmse, strict=True)
        },
    }
    if shuffled_count:
        shuffled_rmse = torch.sqrt(shuffled_squared / shuffled_count).numpy()
        result["shuffled_context_rmse"] = {
            name: float(value)
            for name, value in zip(labels, shuffled_rmse, strict=True)
        }
        result["factual_vs_shuffled_improvement_fraction"] = {
            name: float((shuffled - factual) / max(shuffled, 1e-12))
            for name, factual, shuffled in zip(labels, rmse, shuffled_rmse, strict=True)
        }
    return result


def audit_action_jacobian(
    model: JepaStructuredDynamics,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    maximum_batches: int = 16,
) -> dict[str, object]:
    model.eval()
    diagonal_values = []
    off_axis_values = []
    expected_sign = torch.tensor((1.0, 1.0, -1.0, 1.0), device=device)
    for index, raw in enumerate(loader):
        if index >= maximum_batches:
            break
        batch = _move_batch(raw, device)
        with torch.enable_grad():
            jacobian = model.action_rate_jacobian(
                batch["observation_history"],
                batch["action_history"],
                batch["future_action"],
            )
        controlled = jacobian[:, :, :4].detach()
        diagonal = controlled.diagonal(dim1=1, dim2=2)
        off_axis = controlled - torch.diag_embed(diagonal)
        diagonal_values.append(diagonal.cpu())
        off_axis_values.append(off_axis.cpu())
    if not diagonal_values:
        raise ValueError("action Jacobian audit received no batches")
    diagonal = torch.cat(diagonal_values)
    off_axis = torch.cat(off_axis_values)
    expected = expected_sign.cpu()[None]
    absolute_diagonal = diagonal.abs()
    off_axis_norm = torch.linalg.vector_norm(off_axis, dim=(1, 2))
    diagonal_norm = torch.linalg.vector_norm(diagonal, dim=1).clamp_min(1e-9)
    return {
        "sample_count": len(diagonal),
        "motion_order": list(MOTION_ORDER),
        "expected_diagonal_sign": expected_sign.cpu().tolist(),
        "diagonal_mean_per_axis": diagonal.mean(dim=0).tolist(),
        "diagonal_absolute_mean_per_axis": absolute_diagonal.mean(dim=0).tolist(),
        "expected_sign_fraction_per_axis": (
            diagonal * expected > 0.0
        ).float().mean(dim=0).tolist(),
        "finite_fraction": float(torch.isfinite(diagonal).float().mean()),
        "off_axis_to_diagonal_norm_mean": float(
            (off_axis_norm / diagonal_norm).mean()
        ),
    }


def load_stage1_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[PredictiveStateJEPA, Standardizer, dict[str, object]]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Stage-1 checkpoint must contain a dictionary")
    if checkpoint.get("implementation") != STAGE1_IMPLEMENTATION:
        raise ValueError("Stage-1 checkpoint implementation is incompatible")
    if checkpoint.get("feature_contract_id") != FEATURE_CONTRACT_ID:
        raise ValueError("Stage-1 observation contract is incompatible")
    if checkpoint.get("action_contract_id") != ACTION_CONTRACT_ID:
        raise ValueError("Stage-1 action contract is incompatible")
    if tuple(checkpoint.get("action_labels", ())) != ACTION_LABELS:
        raise ValueError("Stage-1 action labels are incompatible")
    model = PredictiveStateJEPA(Stage1Config(**checkpoint["model_configuration"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    return model, standardizer, checkpoint


def load_structured_dynamics_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    stage1_checkpoint_path: str | Path | None = None,
) -> tuple[JepaStructuredDynamics, dict[str, object]]:
    """Load the independent structured model with its frozen Stage-1 backbone."""
    target_device = _device(device) if isinstance(device, str) else device
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("implementation") != IMPLEMENTATION:
        raise ValueError("checkpoint is not a structured latent dynamics model")
    if checkpoint.get("feature_contract_id") != FEATURE_CONTRACT_ID:
        raise ValueError("structured checkpoint observation contract is incompatible")
    if checkpoint.get("action_contract_id") != ACTION_CONTRACT_ID:
        raise ValueError("structured checkpoint action contract is incompatible")
    resolved_stage1 = (
        Path(stage1_checkpoint_path).expanduser().resolve()
        if stage1_checkpoint_path is not None
        else Path(str(checkpoint["stage1_checkpoint_path"])).expanduser().resolve()
    )
    backbone, standardizer, stage1_checkpoint = load_stage1_checkpoint(
        resolved_stage1, device=target_device
    )
    if stage1_checkpoint.get("implementation") != checkpoint.get(
        "stage1_implementation"
    ):
        raise ValueError("structured checkpoint refers to a different Stage-1 model")
    model = JepaStructuredDynamics(
        backbone,
        standardizer,
        sample_rate_hz=float(checkpoint["sample_rate_hz"]),
        config=StructuredDynamicsConfig(**checkpoint["model_configuration"]),
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _build_datasets(
    manifest: str | Path,
    checkpoint: Mapping[str, object],
    standardizer: Standardizer,
    *,
    window_stride: int,
) -> tuple[dict[str, MotorJepaWindowDataset], float]:
    partitions = {
        name: load_partition_sequences(manifest, name)
        for name in ("train", "validation", "test")
    }
    recorder_rate = common_sample_rate(
        tuple(episode for values in partitions.values() for episode in values)
    )
    model_step_stride = int(checkpoint["model_step_stride"])
    model_rate = recorder_rate / model_step_stride
    if not math.isclose(
        model_rate,
        float(checkpoint["sample_rate_hz"]),
        rel_tol=0.02,
        abs_tol=0.02,
    ):
        raise ValueError("dataset model rate differs from Stage-1 checkpoint")
    datasets = {
        name: MotorJepaWindowDataset(
            episodes,
            standardizer,
            history_steps=int(checkpoint["history_steps"]),
            horizon_steps=int(checkpoint["horizon_steps"]),
            model_step_stride=model_step_stride,
            stride=window_stride,
        )
        for name, episodes in partitions.items()
    }
    return datasets, model_rate


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.window_stride < 1:
        raise ValueError("epochs, batch size, and window stride must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    _seed_everything(args.seed)
    device = _device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone, standardizer, stage1_checkpoint = load_stage1_checkpoint(
        args.stage1_checkpoint, device=device
    )
    datasets, model_rate = _build_datasets(
        args.manifest,
        stage1_checkpoint,
        standardizer,
        window_stride=args.window_stride,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_options)
    validation_loader = DataLoader(
        datasets["validation"], shuffle=False, **loader_options
    )
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_options)

    config = StructuredDynamicsConfig()
    model = JepaStructuredDynamics(
        backbone,
        standardizer,
        sample_rate_hz=model_rate,
        config=config,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=config.learning_rate * 0.05
    )
    history = []
    best_value = math.inf
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device=device
        )
        validation_metrics = evaluate(model, validation_loader, device=device)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        value = float(validation_metrics["loss"]["total"])  # type: ignore[index]
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.6f} validation={value:.6f}",
            flush=True,
        )
        if value < best_value:
            best_value = value
            best_state = deepcopy(model.state_dict())
        scheduler.step()
    if best_state is None:
        raise RuntimeError("structured dynamics training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    validation = evaluate(model, validation_loader, device=device)
    test = evaluate(model, test_loader, device=device)
    validation_jacobian = audit_action_jacobian(
        model, validation_loader, device=device
    )
    test_jacobian = audit_action_jacobian(model, test_loader, device=device)
    readiness = {
        "finite_action_jacobian": validation_jacobian["finite_fraction"] == 1.0,
        "correct_action_sign_all_axes": all(
            value >= 0.95
            for value in validation_jacobian["expected_sign_fraction_per_axis"]
        ),
        "context_beats_zero_context_motion_mean": float(
            np.mean(
                [
                    validation["context_improvement_fraction"][name]
                    for name in MOTION_ORDER
                ]
            )
        )
        > 0.0,
    }
    readiness["passed"] = all(readiness.values())

    checkpoint_path = output_dir / "structured_latent_dynamics.pt"
    checkpoint = {
        "implementation": IMPLEMENTATION,
        "created_at": datetime.now().astimezone().isoformat(),
        "model_configuration": asdict(config),
        "model_state_dict": model.state_dict(),
        "stage1_checkpoint_path": str(
            Path(args.stage1_checkpoint).expanduser().resolve()
        ),
        "stage1_implementation": stage1_checkpoint["implementation"],
        "manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "action_contract_id": ACTION_CONTRACT_ID,
        "observation_labels": list(OBSERVATION_LABELS),
        "action_labels": list(ACTION_LABELS),
        "motion_order": list(MOTION_ORDER),
        "sample_rate_hz": model_rate,
        "history_steps": int(stage1_checkpoint["history_steps"]),
        "horizon_steps": int(stage1_checkpoint["horizon_steps"]),
        "window_stride": args.window_stride,
        "validation_metrics": validation,
        "test_metrics": test,
        "validation_action_jacobian": validation_jacobian,
        "test_action_jacobian": test_jacobian,
        "training_readiness": readiness,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict"}
    }
    report.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "epoch_history": history,
            "window_counts": {name: len(value) for name, value in datasets.items()},
        }
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=_json_ready) + "\n",
        encoding="utf-8",
    )
    print(checkpoint_path)
    print(json.dumps({"validation": validation, "test": test, "readiness": readiness}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMPLEMENTATION",
    "JepaStructuredDynamics",
    "MOTION_ORDER",
    "StructuredCoefficientDecoder",
    "StructuredDynamicsConfig",
    "allocated_thruster_features",
    "audit_action_jacobian",
    "evaluate",
    "load_structured_dynamics_checkpoint",
    "load_stage1_checkpoint",
    "modal_to_motor",
    "motor_to_modal",
    "structured_loss",
    "train_one_epoch",
    "yaw_from_rotation6d",
]
