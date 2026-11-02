"""Fixed-nominal dynamics baseline for the retained MPPI controller.

This module intentionally reuses the JEPA-MPPI candidate generator, costs,
constraints, and receding-horizon update.  Only the trajectory model differs:
the learned history context and residual heads are replaced by the calibrated
nominal underwater prior stored in the Stage-2 checkpoint metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from disaware_learning.control.jepa_mppi import (
    JepaMppiConfig,
    ReferenceConditionedJepaMppi,
)
from disaware_learning.learning.motor_jepa_data import (
    ACTUATOR_MODEL_ID,
    ACTION_CONTRACT_ID,
    ACTION_LABELS,
    DEPTH_INDEX,
    FEATURE_CONTRACT_ID,
    IMU_GYRO_SLICE,
    OBSERVATION_LABELS,
    ROTATION_6D_SLICE,
    Standardizer,
    TEMPORAL_RESAMPLING_ID,
    VIO_VELOCITY_SLICE,
)
from disaware_learning.learning.stage2 import (
    IMPLEMENTATION as STAGE2_IMPLEMENTATION,
    NominalUnderwaterDynamics,
    Stage2Config,
    matrix_to_rotation6d,
    rotation6d_to_matrix,
    so3_exp,
)


IMPLEMENTATION_ID = "fixed_nominal_underwater_mppi_v1"
DYNAMICS_MODEL_ID = "fixed_nominal_underwater_prior_v1"


class FixedNominalUnderwaterRollout(nn.Module):
    """Roll out the fixed Stage-2 underwater prior without learned weights."""

    def __init__(
        self,
        standardizer: Standardizer,
        *,
        sample_rate_hz: float,
        config: Stage2Config = Stage2Config(),
    ) -> None:
        super().__init__()
        if sample_rate_hz <= 0.0 or not math.isfinite(sample_rate_hz):
            raise ValueError("sample rate must be positive and finite")
        self.config = config
        self.sample_rate_hz = float(sample_rate_hz)
        self.nominal = NominalUnderwaterDynamics(config)
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

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    def _denormalize_observation(self, values: Tensor) -> Tensor:
        return values * self.observation_std + self.observation_mean

    def _denormalize_action(self, values: Tensor) -> Tensor:
        return values * self.action_std + self.action_mean

    def _reconstruct_effective_action(
        self,
        action_history: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        physical_history = self._denormalize_action(action_history)
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
        return (
            decay.pow(history_steps - 1) * physical_history[:, 0]
            + (tail_weights * physical_history[:, 1:]).sum(dim=1)
        )

    def rollout_control_from_context(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
        *,
        context: Tensor | None = None,
        include_residual: bool = True,
    ) -> dict[str, Tensor]:
        """Predict a candidate using fixed physics only.

        ``context`` and ``include_residual`` are accepted solely for interface
        compatibility with JEPA-MPPI.  They never affect this baseline.
        """
        del context, include_residual
        if (
            observation_history.ndim != 3
            or action_history.ndim != 3
            or future_action.ndim != 3
        ):
            raise ValueError("nominal MPPI inputs must be [batch, time, channel]")
        batch_size = observation_history.shape[0]
        if (
            batch_size < 1
            or action_history.shape[:2] != observation_history.shape[:2]
            or future_action.shape[0] != batch_size
            or future_action.shape[1] < 1
            or observation_history.shape[-1] != len(OBSERVATION_LABELS)
            or action_history.shape[-1] != len(ACTION_LABELS)
            or future_action.shape[-1] != len(ACTION_LABELS)
        ):
            raise ValueError("nominal MPPI input shapes do not match the model contract")

        current = self._denormalize_observation(observation_history[:, -1])
        physical_action = self._denormalize_action(future_action)
        rotation = rotation6d_to_matrix(current[:, ROTATION_6D_SLICE])
        velocity = current[:, VIO_VELOCITY_SLICE]
        omega = current[:, IMU_GYRO_SLICE]
        depth = current[:, DEPTH_INDEX : DEPTH_INDEX + 1]

        actuator_tau = self.config.actuator_time_constant_seconds
        alpha_scalar = 1.0 - math.exp(-self.dt / actuator_tau)
        interval_alpha = 1.0 - (actuator_tau / self.dt) * alpha_scalar
        alpha = physical_action.new_full(
            (batch_size, len(ACTION_LABELS)), alpha_scalar
        )
        effective_action = self._reconstruct_effective_action(action_history, alpha)
        coefficients = self.nominal.prepare_coefficients(current)
        position_delta = current.new_zeros((batch_size, 3))

        future_rotation: list[Tensor] = []
        future_velocity: list[Tensor] = []
        future_omega: list[Tensor] = []
        future_acceleration: list[Tensor] = []
        future_depth: list[Tensor] = []
        future_position_delta: list[Tensor] = []
        for step in range(physical_action.shape[1]):
            action_innovation = physical_action[:, step] - effective_action
            interval_effective_action = (
                effective_action + interval_alpha * action_innovation
            )
            effective_action = effective_action + alpha * action_innovation
            rate, _ = self.nominal(
                velocity,
                omega,
                interval_effective_action,
                rotation,
                prepared_coefficients=coefficients,
            )
            linear_rate = rate[:, :3]
            angular_rate = rate[:, 3:]
            velocity = velocity + self.dt * linear_rate
            omega = omega + self.dt * angular_rate
            rotation = rotation @ so3_exp(self.dt * omega)
            ned_velocity = (rotation @ velocity.unsqueeze(-1)).squeeze(-1)
            position_delta = position_delta + self.dt * ned_velocity
            depth = depth + self.dt * ned_velocity[:, 2:3]
            body_acceleration = linear_rate + torch.cross(omega, velocity, dim=-1)

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
        return {
            "predicted_future_physical_observation": physical_future,
            "predicted_position_delta_ned_m": torch.stack(
                future_position_delta, dim=1
            ),
        }

    def forward(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
    ) -> dict[str, Tensor]:
        return self.rollout_control_from_context(
            observation_history,
            action_history,
            future_action,
        )


class ReferenceConditionedNominalMppi(ReferenceConditionedJepaMppi):
    """The retained MPPI search evaluated by fixed nominal dynamics."""

    @classmethod
    def from_checkpoint_metadata(
        cls,
        checkpoint_path: str | Path,
        *,
        config: JepaMppiConfig = JepaMppiConfig(),
        device: str | torch.device = "cpu",
    ) -> "ReferenceConditionedNominalMppi":
        # Materialize checkpoint tensors on the metadata-only device.  This
        # parses the contract/configuration while never loading learned tensor
        # storage into CPU or GPU memory.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="meta",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Stage-2 checkpoint must contain a mapping")
        required = (
            "implementation",
            "feature_contract_id",
            "action_contract_id",
            "actuator_model",
            "action_labels",
            "temporal_resampling_id",
            "standardizer",
            "history_steps",
            "horizon_steps",
            "stage2_configuration",
        )
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"checkpoint is missing nominal MPPI metadata: {missing}")
        if checkpoint["implementation"] != STAGE2_IMPLEMENTATION:
            raise ValueError("checkpoint is not the retained physics-integrated Stage 2")
        if checkpoint["feature_contract_id"] != FEATURE_CONTRACT_ID:
            raise ValueError("checkpoint feature contract is unsupported")
        if checkpoint["action_contract_id"] != ACTION_CONTRACT_ID:
            raise ValueError("checkpoint action contract is unsupported")
        if checkpoint["actuator_model"] != ACTUATOR_MODEL_ID:
            raise ValueError("checkpoint actuator contract is unsupported")
        if tuple(checkpoint["action_labels"]) != ACTION_LABELS:
            raise ValueError("checkpoint action labels do not match the modal contract")
        if checkpoint["temporal_resampling_id"] != TEMPORAL_RESAMPLING_ID:
            raise ValueError("checkpoint temporal resampling contract is unsupported")

        time_contract = cls._checkpoint_time_contract(checkpoint)
        standardizer_value = checkpoint["standardizer"]
        stage2_configuration = checkpoint["stage2_configuration"]
        if not isinstance(standardizer_value, Mapping) or not isinstance(
            stage2_configuration, Mapping
        ):
            raise TypeError("checkpoint standardizer and Stage-2 config must be mappings")
        standardizer = Standardizer.from_dict(dict(standardizer_value))
        fixed_model = FixedNominalUnderwaterRollout(
            standardizer,
            sample_rate_hz=time_contract["sample_rate_hz"],
            config=Stage2Config(**dict(stage2_configuration)),
        )
        return cls(
            fixed_model,
            standardizer,
            history_steps=int(checkpoint["history_steps"]),
            model_horizon_steps=int(checkpoint["horizon_steps"]),
            sample_rate_hz=time_contract["sample_rate_hz"],
            config=config,
            device=device,
            stage2_implementation=DYNAMICS_MODEL_ID,
        )
