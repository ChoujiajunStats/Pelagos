"""Dense latent Stage-1 training for predictive underwater dynamics."""

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
from torch.nn import functional
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_

from disaware_learning.learning.jepa_manifest import load_dataset_manifest
from disaware_learning.learning.motor_jepa_data import (
    ACTUATOR_MODEL_ID,
    ACTION_CONTRACT_ID,
    ACTION_LABELS,
    FEATURE_CONTRACT_ID,
    MotorJepaWindowDataset,
    OBSERVATION_LABELS,
    DEPTH_INDEX,
    IMU_GYRO_SLICE,
    Standardizer,
    TEMPORAL_RESAMPLING_ID,
    VIO_VELOCITY_SLICE,
    common_sample_rate,
    load_partition_sequences,
    resample_episode_sequences,
)


IMPLEMENTATION = "predictive_belief_jepa_stage1_response_context_support_query_v7"
CHECKPOINT_SELECTION = (
    "held_out_factual_prediction_objective_excluding_batch_local_sigreg"
)

RESPONSE_OBSERVATION_INDICES = tuple(
    range(VIO_VELOCITY_SLICE.start, VIO_VELOCITY_SLICE.stop)
) + tuple(range(IMU_GYRO_SLICE.start, IMU_GYRO_SLICE.stop)) + (DEPTH_INDEX,)
MOTION_WORK_POINT_INDICES = tuple(
    range(VIO_VELOCITY_SLICE.start, VIO_VELOCITY_SLICE.stop)
) + tuple(range(IMU_GYRO_SLICE.start, IMU_GYRO_SLICE.stop))


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


@dataclass(frozen=True)
class Stage1Config:
    observation_dim: int = len(OBSERVATION_LABELS)
    action_dim: int = len(ACTION_LABELS)
    horizon_steps: int = 30
    latent_dim: int = 32
    predictor_hidden_dim: int = 32
    state_channels: tuple[int, ...] = (24, 32, 32, 32)
    action_channels: tuple[int, ...] = (32, 32)
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    tcn_kernel_size: int = 3
    model_step_seconds: float = 0.10
    teacher_forcing_weight: float = 1.0
    rollout_weight: float = 1.0
    backward_transition_weight: float = 0.25
    response_prediction_weight: float = 0.25
    support_query_response_weight: float = 0.25
    support_query_steps: int = 7
    response_delta_std: tuple[float, ...] = (1.0,) * len(
        RESPONSE_OBSERVATION_INDICES
    )
    target_ema_decay: float = 0.99
    # This quadrature implementation returns about 0.40 at the collapsed
    # origin and about 0.01 for a finite batch of Gaussian samples. A unit
    # weight is required to keep the shared encoder away from the zero fixed
    # point; SkyJEPA's numeric SIGReg scale is not directly interchangeable.
    sigreg_weight: float = 1.0
    learning_rate: float = 5e-3
    final_learning_rate: float = 1e-4
    warmup_fraction: float = 0.10
    weight_decay: float = 1e-5
    grad_clip_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.observation_dim < 1 or self.action_dim != len(ACTION_LABELS):
            raise ValueError("Stage-1 requires observations and modal actuator actions")
        if self.horizon_steps < 1 or min(
            self.latent_dim, self.predictor_hidden_dim
        ) < 2:
            raise ValueError("invalid Stage-1 latent dimensions")
        if not self.state_channels or not self.action_channels:
            raise ValueError("TCN channel lists cannot be empty")
        if len(self.state_channels) != len(self.tcn_dilations):
            raise ValueError("history TCN channels must match the dilation schedule")
        if any(value < 1 for value in (*self.state_channels, *self.action_channels)):
            raise ValueError("TCN channels must be positive")
        if any(value < 1 for value in self.tcn_dilations):
            raise ValueError("TCN dilations must be positive")
        if self.tcn_kernel_size < 2:
            raise ValueError("TCN kernel size must be at least two")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.model_step_seconds <= 0.0:
            raise ValueError("model step must be positive")
        if not 0.0 < self.target_ema_decay < 1.0:
            raise ValueError("target EMA decay must be in (0, 1)")
        if min(
            self.sigreg_weight,
            self.teacher_forcing_weight,
            self.rollout_weight,
            self.backward_transition_weight,
            self.response_prediction_weight,
            self.support_query_response_weight,
            self.learning_rate,
            self.final_learning_rate,
            self.grad_clip_norm,
        ) <= 0.0:
            raise ValueError("training scales must be positive")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("final learning rate cannot exceed peak learning rate")
        if len(self.response_delta_std) != len(RESPONSE_OBSERVATION_INDICES) or any(
            value <= 0.0 or not math.isfinite(value)
            for value in self.response_delta_std
        ):
            raise ValueError("response delta scales must be finite and positive")
        if self.support_query_steps < 1:
            raise ValueError("support-query response requires at least one query step")

    @property
    def receptive_field_steps(self) -> int:
        return 1 + (self.tcn_kernel_size - 1) * sum(self.tcn_dilations)

    @property
    def belief_dim(self) -> int:
        """Width of the history/dynamics half of the exported context."""
        return self.latent_dim


class _CausalTemporalBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
        )
        self.normalization = nn.LayerNorm(output_channels)
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1)
        )

    def forward(self, values: Tensor) -> Tensor:
        residual = self.residual(values)
        encoded = self.convolution(functional.pad(values, (self.left_padding, 0)))
        encoded = self.normalization(encoded.transpose(1, 2)).transpose(1, 2)
        return residual + functional.gelu(encoded)


class _CausalTemporalEncoder(nn.Module):
    """Causal TCN returning one representation for every input timestep."""

    def __init__(
        self,
        input_dim: int,
        channels: tuple[int, ...],
        output_dim: int,
        *,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers = []
        previous = input_dim
        for width, dilation in zip(channels, dilations, strict=True):
            layers.append(
                _CausalTemporalBlock(
                    previous,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            previous = width
        self.layers = nn.ModuleList(layers)
        self.projection = nn.Linear(previous, output_dim)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("causal TCN expects [batch, time, channel]")
        encoded = values.transpose(1, 2)
        for layer in self.layers:
            encoded = layer(encoded)
        return self.projection(encoded.transpose(1, 2))


class _LatentDistributionRegularizer(nn.Module):
    """Batch-invariant characteristic-function Gaussian regularizer."""

    def __init__(
        self,
        feature_dim: int,
        *,
        projection_count: int = 128,
        grid_points: int = 17,
        grid_max: float = 3.0,
        seed: int = 20260803,
    ) -> None:
        super().__init__()
        if feature_dim < 2 or projection_count < 1:
            raise ValueError("invalid SIGReg dimensions")
        if grid_points < 3 or grid_points % 2 == 0 or grid_max <= 0.0:
            raise ValueError("SIGReg requires an odd positive grid")
        self.feature_dim = int(feature_dim)
        self.projection_count = int(projection_count)
        self.seed = int(seed)
        grid = torch.linspace(0.0, grid_max, grid_points)
        delta = grid_max / (grid_points - 1)
        quadrature = torch.full_like(grid, 2.0 * delta)
        quadrature[[0, -1]] = delta
        normal_characteristic = torch.exp(-0.5 * grid.square())
        self.register_buffer("grid", grid)
        self.register_buffer("normal_characteristic", normal_characteristic)
        self.register_buffer(
            "integration_weights", quadrature * normal_characteristic
        )
        self.register_buffer("projection_step", torch.zeros((), dtype=torch.long))

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim < 2 or values.shape[-1] != self.feature_dim:
            raise ValueError("SIGReg expects (..., samples, feature)")
        if values.shape[-2] < 2:
            raise ValueError("SIGReg requires at least two samples")
        generator = torch.Generator(device=values.device)
        projection_step = int(self.projection_step.item()) if self.training else 0
        generator.manual_seed(self.seed + projection_step)
        directions = torch.randn(
            self.feature_dim,
            self.projection_count,
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )
        directions /= torch.linalg.vector_norm(
            directions, dim=0, keepdim=True
        ).clamp_min(1e-9)
        if self.training:
            self.projection_step.add_(1)
        arguments = (values @ directions).unsqueeze(-1) * self.grid
        empirical_real = torch.cos(arguments).mean(dim=-3)
        empirical_imaginary = torch.sin(arguments).mean(dim=-3)
        error = (
            empirical_real - self.normal_characteristic
        ).square() + empirical_imaginary.square()
        return (error @ self.integration_weights).mean()


class PredictiveStateJEPA(nn.Module):
    """Action-conditioned state JEPA with a causal transition context.

    The EMA target contains observations only. Past observation deltas,
    executed actions, and action deltas form the history context. Candidate
    actions enter through the recurrent transition while the encoded history
    remains fixed over the imagined rollout.
    """

    def __init__(self, config: Stage1Config) -> None:
        super().__init__()
        self.config = config
        state_width = config.state_channels[-1]
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.observation_dim, state_width),
            nn.GELU(),
            nn.LayerNorm(state_width),
            nn.Linear(state_width, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.target_observation_encoder = deepcopy(self.observation_encoder)
        self.target_observation_encoder.requires_grad_(False)
        response_dim = len(RESPONSE_OBSERVATION_INDICES)
        # Each token exposes both the motion work point and the response to the
        # action executed over the preceding interval. The outer product makes
        # action-response gain and cross-axis coupling directly observable
        # without using simulator domain labels.
        dynamics_token_dim = (
            len(MOTION_WORK_POINT_INDICES)
            + config.observation_dim
            + 2 * config.action_dim
            + response_dim * config.action_dim
        )
        self.dynamics_context_encoder = _CausalTemporalEncoder(
            dynamics_token_dim,
            config.state_channels,
            config.latent_dim,
            kernel_size=config.tcn_kernel_size,
            dilations=config.tcn_dilations,
        )
        self.dynamics_context_normalization = nn.LayerNorm(config.latent_dim)
        action_layers: list[nn.Module] = []
        previous = 2 * config.action_dim
        for width in config.action_channels:
            action_layers.extend((nn.Linear(previous, width), nn.GELU()))
            previous = width
        action_layers.append(nn.Linear(previous, config.predictor_hidden_dim))
        self.action_embedding = nn.Sequential(*action_layers)
        self.transition_conditioning = nn.Sequential(
            nn.Linear(
                config.predictor_hidden_dim + config.latent_dim,
                config.predictor_hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.predictor_hidden_dim),
        )
        self.transition = nn.GRUCell(config.predictor_hidden_dim, config.latent_dim)
        self.transition_normalization = nn.LayerNorm(config.latent_dim)
        self.backward_transition_conditioning = nn.Sequential(
            nn.Linear(
                config.predictor_hidden_dim + config.latent_dim,
                config.predictor_hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.predictor_hidden_dim),
        )
        self.backward_transition = nn.GRUCell(
            config.predictor_hidden_dim, config.latent_dim
        )
        self.backward_transition_normalization = nn.LayerNorm(config.latent_dim)
        response_hidden = max(config.predictor_hidden_dim, 32)
        self.response_predictor = nn.Sequential(
            nn.Linear(
                config.latent_dim + response_dim + 2 * config.action_dim,
                response_hidden,
            ),
            nn.GELU(),
            nn.LayerNorm(response_hidden),
            nn.Linear(response_hidden, response_hidden),
            nn.GELU(),
            nn.Linear(response_hidden, response_dim),
        )
        self.register_buffer(
            "response_delta_std",
            torch.as_tensor(config.response_delta_std, dtype=torch.float32),
        )
        self.sigreg = _LatentDistributionRegularizer(
            config.latent_dim,
            seed=20260803,
        )

    @property
    def context_dim(self) -> int:
        return 2 * self.config.latent_dim

    def train(self, mode: bool = True) -> "PredictiveStateJEPA":
        super().train(mode)
        self.target_observation_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = self.config.target_ema_decay
        for target, online in zip(
            self.target_observation_encoder.parameters(),
            self.observation_encoder.parameters(),
            strict=True,
        ):
            target.lerp_(online, 1.0 - decay)
        for target, online in zip(
            self.target_observation_encoder.buffers(),
            self.observation_encoder.buffers(),
            strict=True,
        ):
            target.copy_(online)

    @staticmethod
    def _transition_context_tokens(observation: Tensor, action: Tensor) -> Tensor:
        if observation.shape[:2] != action.shape[:2]:
            raise ValueError("observation and action histories must be aligned")
        if observation.ndim != 3 or action.ndim != 3:
            raise ValueError("history tensors must be [batch, time, channel]")
        observation_delta = torch.cat(
            (
                torch.zeros_like(observation[:, :1]),
                observation[:, 1:] - observation[:, :-1],
            ),
            dim=1,
        )
        action_delta = torch.cat(
            (torch.zeros_like(action[:, :1]), action[:, 1:] - action[:, :-1]),
            dim=1,
        )
        motion_response = observation_delta[..., RESPONSE_OBSERVATION_INDICES]
        action_response = (
            motion_response.unsqueeze(-1) * action.unsqueeze(-2)
        ).flatten(start_dim=-2)
        return torch.cat(
            (
                observation[..., MOTION_WORK_POINT_INDICES],
                observation_delta,
                action,
                action_delta,
                action_response,
            ),
            dim=-1,
        )

    def encode_dynamics_context(
        self,
        observation_history: Tensor,
        action_history: Tensor,
    ) -> Tensor:
        tokens = self._transition_context_tokens(
            observation_history, action_history
        )
        # The final causal token is the only output that has observed the full
        # history. Averaging also includes short, left-padded prefixes and
        # dilutes the best available dynamics estimate.
        context = self.dynamics_context_encoder(tokens)[:, -1]
        return self.dynamics_context_normalization(context)

    def predict_teacher_forced_responses(
        self,
        dynamics_context: Tensor,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
        future_observation: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict causal one-step motion responses in standardized units."""
        previous_observation = torch.cat(
            (observation_history[:, -1:], future_observation[:, :-1]), dim=1
        )
        previous_motion = previous_observation[..., RESPONSE_OBSERVATION_INDICES]
        next_motion = future_observation[..., RESPONSE_OBSERVATION_INDICES]
        previous_action = torch.cat(
            (action_history[:, -1:], future_action[:, :-1]), dim=1
        )
        predicted = self.predict_responses_from_belief(
            dynamics_context,
            previous_motion,
            previous_action,
            future_action,
        )
        target = (next_motion - previous_motion) / self.response_delta_std
        return predicted, target

    def predict_responses_from_belief(
        self,
        dynamics_context: Tensor,
        previous_motion: Tensor,
        previous_action: Tensor,
        action: Tensor,
    ) -> Tensor:
        if (
            previous_motion.shape[:2] != action.shape[:2]
            or previous_action.shape != action.shape
        ):
            raise ValueError("response work points and actions must be aligned")
        if dynamics_context.shape != (
            action.shape[0],
            self.config.belief_dim,
        ):
            raise ValueError("response belief has the wrong shape")
        belief = dynamics_context[:, None].expand(-1, action.shape[1], -1)
        return self.response_predictor(
            torch.cat(
                (belief, previous_motion, action, action - previous_action), dim=-1
            )
        )

    def predict_support_query_responses(
        self,
        observation_history: Tensor,
        action_history: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Identify dynamics on a causal prefix and predict its trailing response.

        Unlike an episode-level window swap, support and query share the same
        operating point trajectory and actuator state. The belief cannot see
        the query response it is asked to explain.
        """
        if observation_history.shape[:2] != action_history.shape[:2]:
            raise ValueError("support-query histories must be aligned")
        if observation_history.ndim != 3 or observation_history.shape[1] < 3:
            raise ValueError("support-query response requires three history steps")
        query_steps = min(
            self.config.support_query_steps, observation_history.shape[1] - 2
        )
        split = observation_history.shape[1] - query_steps
        belief = self.encode_dynamics_context(
            observation_history[:, :split], action_history[:, :split]
        )
        previous_observation = observation_history[:, split - 1 : -1]
        query_observation = observation_history[:, split:]
        previous_motion = previous_observation[..., RESPONSE_OBSERVATION_INDICES]
        next_motion = query_observation[..., RESPONSE_OBSERVATION_INDICES]
        previous_action = action_history[:, split - 1 : -1]
        query_action = action_history[:, split:]
        predicted = self.predict_responses_from_belief(
            belief, previous_motion, previous_action, query_action
        )
        target = (next_motion - previous_motion) / self.response_delta_std
        return predicted, target, belief

    def encode_context(
        self,
        observation_history: Tensor,
        action_history: Tensor,
    ) -> Tensor:
        state = self.observation_encoder(observation_history[:, -1])
        dynamics_context = self.encode_dynamics_context(
            observation_history, action_history
        )
        return torch.cat((state, dynamics_context), dim=-1)

    def encode_action_windows(
        self,
        action_history: Tensor,
        future_action: Tensor,
    ) -> Tensor:
        if action_history.ndim != 3 or future_action.ndim != 3:
            raise ValueError("actions must be [batch, time, actuator]")
        if action_history.shape[0] != future_action.shape[0]:
            raise ValueError("past and future action batches must match")
        if action_history.shape[-1] != self.config.action_dim or future_action.shape[
            -1
        ] != self.config.action_dim:
            raise ValueError("action channel mismatch")
        previous = torch.cat((action_history[:, -1:], future_action[:, :-1]), dim=1)
        features = torch.cat((future_action, future_action - previous), dim=-1)
        return self.action_embedding(features)

    def rollout_action_embeddings(
        self,
        context: Tensor,
        action_embeddings: Tensor,
    ) -> Tensor:
        if context.ndim != 2 or action_embeddings.ndim != 3:
            raise ValueError("invalid latent rollout shapes")
        expected_context = self.context_dim
        if context.shape[-1] != expected_context:
            raise ValueError(
                f"context must contain state and belief ({expected_context} channels)"
            )
        hidden = context[:, : self.config.latent_dim]
        belief = context[:, self.config.latent_dim :]
        # Conditioning depends only on the candidate action and the fixed
        # history belief, not on the recurrent hidden state. Evaluate every
        # horizon step in one matrix operation instead of launching the same
        # small Linear/GELU/LayerNorm chain once per step.
        belief_sequence = belief.unsqueeze(1).expand(
            -1, action_embeddings.shape[1], -1
        )
        conditioned_actions = self.transition_conditioning(
            torch.cat((action_embeddings, belief_sequence), dim=-1)
        )
        values = []
        for conditioning in conditioned_actions.unbind(dim=1):
            hidden = self.transition(conditioning, hidden)
            hidden = self.transition_normalization(hidden)
            values.append(hidden)
        return torch.stack(values, dim=1)

    def predict_latents(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
    ) -> Tensor:
        hidden = self.encode_context(observation_history, action_history)
        encoded_action = self.encode_action_windows(action_history, future_action)
        return self.rollout_action_embeddings(hidden, encoded_action)

    @torch.no_grad()
    def transition_innovation(
        self,
        prefix_observation: Tensor,
        prefix_action: Tensor,
        executed_action: Tensor,
        observed_next_observation: Tensor,
    ) -> dict[str, Tensor]:
        """Measure one executed transition in the frozen Stage-1 latent space.

        Inputs must already use the Stage-1 standardization and depth anchor.
        The observed latent comes from the EMA target encoder used by the JEPA
        objective, while the prediction sees only the strict causal prefix and
        the action that was actually executed.
        """
        if prefix_observation.ndim != 3 or prefix_action.ndim != 3:
            raise ValueError("transition prefix must be [batch, time, channel]")
        if prefix_observation.shape[:2] != prefix_action.shape[:2]:
            raise ValueError("transition prefix observations and actions must align")
        if executed_action.ndim != 2 or observed_next_observation.ndim != 2:
            raise ValueError("executed action and next observation must be vectors")
        batch_size = prefix_observation.shape[0]
        if (
            executed_action.shape
            != (batch_size, self.config.action_dim)
            or observed_next_observation.shape
            != (batch_size, self.config.observation_dim)
        ):
            raise ValueError("executed transition has the wrong feature dimensions")
        predicted_latent = self.predict_latents(
            prefix_observation,
            prefix_action,
            executed_action.unsqueeze(1),
        )[:, 0]
        observed_latent = self.target_observation_encoder(
            observed_next_observation
        )
        return {
            "observed_latent": observed_latent,
            "predicted_latent": predicted_latent,
            "latent_innovation": observed_latent - predicted_latent,
        }

    def backward_teacher_forced_latents(
        self,
        dynamics_context: Tensor,
        action_embeddings: Tensor,
        current_target: Tensor,
        future_target: Tensor,
    ) -> Tensor:
        if dynamics_context.ndim != 2 or action_embeddings.ndim != 3:
            raise ValueError("invalid backward transition context")
        if current_target.shape != (future_target.shape[0], 1, self.config.latent_dim):
            raise ValueError("current target latent has the wrong shape")
        previous_target = torch.cat((current_target, future_target[:, :-1]), dim=1)
        next_target = future_target
        conditioning = self.backward_transition_conditioning(
            torch.cat(
                (
                    action_embeddings,
                    dynamics_context[:, None].expand(
                        -1, action_embeddings.shape[1], -1
                    ),
                ),
                dim=-1,
            )
        )
        backward = self.backward_transition(
            conditioning.reshape(-1, conditioning.shape[-1]),
            next_target.reshape(-1, next_target.shape[-1]),
        )
        backward = self.backward_transition_normalization(backward).reshape_as(
            previous_target
        )
        return backward

    def encode_target_windows(
        self,
        future_observation: Tensor,
    ) -> Tensor:
        horizon_steps = future_observation.shape[1]
        if horizon_steps != self.config.horizon_steps:
            raise ValueError("future observation horizon does not match Stage-1 config")
        with torch.no_grad():
            return self.target_observation_encoder(future_observation)

    def forward(
        self,
        observation_history: Tensor,
        action_history: Tensor,
        future_action: Tensor,
        future_observation: Tensor | None = None,
    ) -> dict[str, Tensor]:
        predicted = self.predict_latents(
            observation_history, action_history, future_action
        )
        output = {"predicted_latent": predicted}
        if future_observation is not None:
            with torch.no_grad():
                target = self.target_observation_encoder(future_observation)
                current_target = self.target_observation_encoder(
                    observation_history[:, -1:]
                )
            teacher_input = torch.cat((current_target, target[:, :-1]), dim=1)
            action_embedding = self.encode_action_windows(action_history, future_action)
            dynamics_context = self.encode_dynamics_context(
                observation_history, action_history
            )
            response, response_target = self.predict_teacher_forced_responses(
                dynamics_context,
                observation_history,
                action_history,
                future_action,
                future_observation,
            )
            output["predicted_response"] = response
            output["target_response"] = response_target
            support_query, support_query_target, support_belief = (
                self.predict_support_query_responses(
                    observation_history, action_history
                )
            )
            output["predicted_support_query_response"] = support_query
            output["target_support_query_response"] = support_query_target
            output["support_belief"] = support_belief
            teacher_conditioning = self.transition_conditioning(
                torch.cat(
                    (
                        action_embedding,
                        dynamics_context[:, None].expand(
                            -1, action_embedding.shape[1], -1
                        ),
                    ),
                    dim=-1,
                )
            )
            teacher = self.transition(
                teacher_conditioning.reshape(
                    -1, teacher_conditioning.shape[-1]
                ),
                teacher_input.reshape(-1, teacher_input.shape[-1]),
            )
            teacher = self.transition_normalization(teacher).reshape_as(target)
            output["target_latent"] = target
            output["teacher_forced_latent"] = teacher
            output["backward_teacher_forced_latent"] = (
                self.backward_teacher_forced_latents(
                    dynamics_context,
                    action_embedding,
                    current_target,
                    target,
                )
            )
            output["backward_target_latent"] = torch.cat(
                (current_target, target[:, :-1]), dim=1
            )
        return output

    def loss(
        self,
        output: Mapping[str, Tensor],
        *,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if not {"predicted_latent", "target_latent"}.issubset(output):
            raise ValueError("Stage-1 loss requires predicted and target latents")
        predicted = output["predicted_latent"]
        target = output["target_latent"]
        rollout_error = (predicted - target).square().mean(dim=2)
        teacher = output.get("teacher_forced_latent")
        if teacher is None:
            raise ValueError("Stage-1 loss requires teacher-forced latents")
        teacher_error = (teacher - target).square().mean(dim=2)
        backward = output.get("backward_teacher_forced_latent")
        backward_target = output.get("backward_target_latent")
        if backward is None or backward_target is None:
            raise ValueError("Stage-1 loss requires backward transition latents")
        backward_error = (backward - backward_target).square().mean(dim=2)
        response = output.get("predicted_response")
        response_target = output.get("target_response")
        if response is None or response_target is None:
            raise ValueError("Stage-1 loss requires physical response predictions")
        response_error = (response - response_target).square().mean(dim=2)
        support_query = output.get("predicted_support_query_response")
        support_query_target = output.get("target_support_query_response")
        if support_query is None or support_query_target is None:
            raise ValueError("Stage-1 loss requires causal support-query responses")
        support_query_error = (support_query - support_query_target).square().mean(
            dim=2
        )
        time_seconds = (
            torch.arange(1, predicted.shape[1] + 1, device=predicted.device)
            * self.config.model_step_seconds
        )
        horizon_weight = torch.where(
            time_seconds <= 0.2,
            3.0,
            torch.where(
                time_seconds <= 0.5,
                2.0,
                torch.where(
                    time_seconds <= 1.0,
                    1.5,
                    torch.where(time_seconds <= 2.0, 1.0, 0.5),
                ),
            ),
        )
        error = rollout_error
        if sample_weight is None:
            per_horizon = rollout_error.mean(dim=0)
            teacher_per_horizon = teacher_error.mean(dim=0)
            backward_per_horizon = backward_error.mean(dim=0)
            response_per_horizon = response_error.mean(dim=0)
            support_query_response = support_query_error.mean()
        else:
            if sample_weight.ndim != 1 or sample_weight.shape[0] != error.shape[0]:
                raise ValueError("sample weights must match batch size")
            weights = sample_weight.to(error).clamp_min(0.0)
            denominator = weights.sum().clamp_min(1e-9)
            per_horizon = (rollout_error * weights[:, None]).sum(dim=0) / denominator
            teacher_per_horizon = (teacher_error * weights[:, None]).sum(dim=0) / denominator
            backward_per_horizon = (
                backward_error * weights[:, None]
            ).sum(dim=0) / denominator
            response_per_horizon = (
                response_error * weights[:, None]
            ).sum(dim=0) / denominator
            support_query_response = (
                support_query_error * weights[:, None]
            ).sum() / (denominator * support_query_error.shape[1])
        rollout = (per_horizon * horizon_weight).sum() / horizon_weight.sum()
        teacher_forcing = (
            teacher_per_horizon * horizon_weight
        ).sum() / horizon_weight.sum()
        backward_transition = (
            backward_per_horizon * horizon_weight
        ).sum() / horizon_weight.sum()
        response_prediction = (
            response_per_horizon * horizon_weight
        ).sum() / horizon_weight.sum()
        prediction = rollout
        sigreg = (
            self.sigreg(predicted.permute(1, 0, 2))
            if predicted.shape[0] >= 2
            else prediction.new_zeros(())
        )
        total = (
            self.config.rollout_weight * rollout
            + self.config.teacher_forcing_weight * teacher_forcing
            + self.config.backward_transition_weight * backward_transition
            + self.config.response_prediction_weight * response_prediction
            + self.config.support_query_response_weight * support_query_response
            + self.config.sigreg_weight * sigreg
        )
        return {
            "total": total,
            "prediction": prediction,
            "rollout": rollout,
            "teacher_forcing": teacher_forcing,
            "backward_transition": backward_transition,
            "response_prediction": response_prediction,
            "support_query_response": support_query_response,
            "sigreg": sigreg,
            "per_horizon": per_horizon,
            "teacher_per_horizon": teacher_per_horizon,
            "backward_per_horizon": backward_per_horizon,
            "response_per_horizon": response_per_horizon,
        }



def _move_batch(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    required = (
        "observation_history",
        "action_history",
        "future_action",
        "future_observation",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"missing Stage-1 batch fields: {missing}")
    moved = {
        name: batch[name].to(device=device, dtype=torch.float32, non_blocking=True)
        for name in required
    }
    return moved


def train_one_epoch(
    model: PredictiveStateJEPA,
    loader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    model.train()
    sums = {
        "total": 0.0,
        "prediction": 0.0,
        "rollout": 0.0,
        "teacher_forcing": 0.0,
        "backward_transition": 0.0,
        "response_prediction": 0.0,
        "support_query_response": 0.0,
        "sigreg": 0.0,
    }
    sample_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(model(**batch))
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("non-finite Stage-1 loss")
        losses["total"].backward()
        clip_grad_norm_(model.parameters(), model.config.grad_clip_norm)
        optimizer.step()
        model.update_target_encoder()
        if scheduler is not None:
            scheduler.step()
        for name in sums:
            sums[name] += float(losses[name].detach().cpu()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("empty training loader")
    return {name: value / sample_count for name, value in sums.items()}


def _warmup_cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    final_ratio: float,
) -> float:
    if total_steps < 1 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("invalid learning-rate schedule lengths")
    if not 0.0 < final_ratio <= 1.0:
        raise ValueError("final learning-rate ratio must be in (0, 1]")
    bounded_step = min(max(int(step), 0), total_steps - 1)
    if warmup_steps and bounded_step < warmup_steps:
        return float(bounded_step + 1) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps - 1, 1)
    progress = float(bounded_step - warmup_steps) / float(decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return final_ratio + (1.0 - final_ratio) * cosine


@torch.no_grad()
def evaluate(
    model: PredictiveStateJEPA,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    sums = {
        "total": 0.0,
        "prediction": 0.0,
        "rollout": 0.0,
        "teacher_forcing": 0.0,
        "backward_transition": 0.0,
        "response_prediction": 0.0,
        "support_query_response": 0.0,
        "sigreg": 0.0,
    }
    per_step: Tensor | None = None
    sample_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        losses = model.loss(model(**batch))
        for name in sums:
            sums[name] += float(losses[name].cpu()) * batch_size
        current = losses["per_horizon"].cpu()
        weighted_current = current * batch_size
        per_step = weighted_current if per_step is None else per_step + weighted_current
        sample_count += batch_size
    if sample_count == 0 or per_step is None:
        raise ValueError("empty evaluation loader")
    per_step /= sample_count
    return {
        **{name: value / sample_count for name, value in sums.items()},
        "per_step": per_step.tolist(),
    }


@torch.no_grad()
def audit_response_prediction(
    model: PredictiveStateJEPA,
    loader: Iterable[Mapping[str, Tensor]],
    standardizer: Standardizer,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Measure response skill and whether it uses the factual history belief."""
    model.eval()
    response_dim = len(RESPONSE_OBSERVATION_INDICES)
    factual_sum = torch.zeros(response_dim, dtype=torch.float64)
    zero_sum = torch.zeros_like(factual_sum)
    shuffled_sum = torch.zeros_like(factual_sum)
    factual_better_count = 0
    comparison_count = 0
    shuffled_transition_count = 0
    transition_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        belief = model.encode_dynamics_context(
            batch["observation_history"], batch["action_history"]
        )
        predicted, target = model.predict_teacher_forced_responses(
            belief,
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
            batch["future_observation"],
        )
        factual_error = (predicted - target).square()
        zero_error = target.square()
        factual_sum += factual_error.sum(dim=(0, 1)).double().cpu()
        zero_sum += zero_error.sum(dim=(0, 1)).double().cpu()
        transition_count += target.shape[0] * target.shape[1]
        episode_index = raw_batch.get("episode_index")
        if episode_index is None:
            continue
        replacement = _cross_episode_indices(
            episode_index.to(device=device, dtype=torch.int64)
        )
        if replacement is None:
            continue
        shuffled, _ = model.predict_teacher_forced_responses(
            belief[replacement],
            batch["observation_history"],
            batch["action_history"],
            batch["future_action"],
            batch["future_observation"],
        )
        shuffled_error = (shuffled - target).square()
        shuffled_sum += shuffled_error.sum(dim=(0, 1)).double().cpu()
        factual_better_count += int(
            (
                factual_error.mean(dim=(1, 2))
                < shuffled_error.mean(dim=(1, 2))
            ).sum()
        )
        comparison_count += target.shape[0]
        shuffled_transition_count += target.shape[0] * target.shape[1]
    if transition_count == 0:
        raise ValueError("empty response-prediction audit loader")
    factual_mse = factual_sum / transition_count
    zero_mse = zero_sum / transition_count
    shuffled_mse = shuffled_sum / max(shuffled_transition_count, 1)
    physical_scale = torch.as_tensor(
        standardizer.observation_std[list(RESPONSE_OBSERVATION_INDICES)],
        dtype=torch.float64,
    ) * model.response_delta_std.double().cpu()
    labels = [OBSERVATION_LABELS[index] for index in RESPONSE_OBSERVATION_INDICES]
    return {
        "response_labels": labels,
        "transition_count": transition_count,
        "factual_normalized_mse_per_channel": factual_mse.tolist(),
        "zero_response_normalized_mse_per_channel": zero_mse.tolist(),
        "normalized_skill_over_zero_per_channel": (
            1.0 - factual_mse / zero_mse.clamp_min(1e-12)
        ).tolist(),
        "physical_rmse_per_step_per_channel": (
            factual_mse.sqrt() * physical_scale
        ).tolist(),
        "aggregate_factual_normalized_mse": float(factual_mse.mean()),
        "aggregate_zero_response_normalized_mse": float(zero_mse.mean()),
        "aggregate_normalized_skill_over_zero": float(
            1.0 - factual_mse.mean() / zero_mse.mean().clamp_min(1e-12)
        ),
        "shuffled_belief_comparison_count": comparison_count,
        "shuffled_belief_normalized_mse_per_channel": (
            shuffled_mse.tolist() if comparison_count else None
        ),
        "factual_better_than_shuffled_belief_fraction": (
            factual_better_count / comparison_count if comparison_count else None
        ),
        "counterfactual_ground_truth": False,
    }


@torch.no_grad()
def audit_support_query_prediction(
    model: PredictiveStateJEPA,
    loader: Iterable[Mapping[str, Tensor]],
    standardizer: Standardizer,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Audit causal history identification on held-out trailing transitions."""
    model.eval()
    response_dim = len(RESPONSE_OBSERVATION_INDICES)
    factual_sum = torch.zeros(response_dim, dtype=torch.float64)
    zero_sum = torch.zeros_like(factual_sum)
    shuffled_sum = torch.zeros_like(factual_sum)
    factual_better_count = 0
    comparison_count = 0
    transition_count = 0
    shuffled_transition_count = 0
    query_steps_seen: int | None = None
    support_steps_seen: int | None = None
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        predicted, target, belief = model.predict_support_query_responses(
            batch["observation_history"], batch["action_history"]
        )
        query_steps_seen = target.shape[1]
        support_steps_seen = batch["observation_history"].shape[1] - target.shape[1]
        factual_error = (predicted - target).square()
        zero_error = target.square()
        factual_sum += factual_error.sum(dim=(0, 1)).double().cpu()
        zero_sum += zero_error.sum(dim=(0, 1)).double().cpu()
        transition_count += target.shape[0] * target.shape[1]
        episode_index = raw_batch.get("episode_index")
        if episode_index is None:
            continue
        replacement = _cross_episode_indices(
            episode_index.to(device=device, dtype=torch.int64)
        )
        if replacement is None:
            continue
        split = support_steps_seen
        observation_history = batch["observation_history"]
        action_history = batch["action_history"]
        previous_motion = observation_history[:, split - 1 : -1][
            ..., RESPONSE_OBSERVATION_INDICES
        ]
        previous_action = action_history[:, split - 1 : -1]
        query_action = action_history[:, split:]
        shuffled = model.predict_responses_from_belief(
            belief[replacement], previous_motion, previous_action, query_action
        )
        shuffled_error = (shuffled - target).square()
        shuffled_sum += shuffled_error.sum(dim=(0, 1)).double().cpu()
        factual_better_count += int(
            (
                factual_error.mean(dim=(1, 2))
                < shuffled_error.mean(dim=(1, 2))
            ).sum()
        )
        comparison_count += target.shape[0]
        shuffled_transition_count += target.shape[0] * target.shape[1]
    if transition_count == 0:
        raise ValueError("empty support-query audit loader")
    factual_mse = factual_sum / transition_count
    zero_mse = zero_sum / transition_count
    shuffled_mse = shuffled_sum / max(shuffled_transition_count, 1)
    physical_scale = torch.as_tensor(
        standardizer.observation_std[list(RESPONSE_OBSERVATION_INDICES)],
        dtype=torch.float64,
    ) * model.response_delta_std.double().cpu()
    return {
        "response_labels": [
            OBSERVATION_LABELS[index] for index in RESPONSE_OBSERVATION_INDICES
        ],
        "support_steps": support_steps_seen,
        "query_steps": query_steps_seen,
        "transition_count": transition_count,
        "factual_normalized_mse_per_channel": factual_mse.tolist(),
        "zero_response_normalized_mse_per_channel": zero_mse.tolist(),
        "normalized_skill_over_zero_per_channel": (
            1.0 - factual_mse / zero_mse.clamp_min(1e-12)
        ).tolist(),
        "physical_rmse_per_step_per_channel": (
            factual_mse.sqrt() * physical_scale
        ).tolist(),
        "aggregate_factual_normalized_mse": float(factual_mse.mean()),
        "aggregate_zero_response_normalized_mse": float(zero_mse.mean()),
        "aggregate_normalized_skill_over_zero": float(
            1.0 - factual_mse.mean() / zero_mse.mean().clamp_min(1e-12)
        ),
        "shuffled_belief_comparison_count": comparison_count,
        "shuffled_belief_normalized_mse_per_channel": (
            shuffled_mse.tolist() if comparison_count else None
        ),
        "factual_better_than_shuffled_belief_fraction": (
            factual_better_count / comparison_count if comparison_count else None
        ),
        "causal_support_excludes_query_response": True,
        "counterfactual_ground_truth": False,
    }


@torch.no_grad()
def _cross_episode_indices(episode_index: Tensor) -> Tensor | None:
    """Return deterministic replacement indices whose episodes differ.

    A batch can contain several contiguous windows from one episode.  A plain
    ``roll(1)`` is therefore only a local-neighbour perturbation, not a
    shuffled-action audit.  This helper deliberately makes the provenance
    change explicit; callers skip the comparison when a batch has one episode.
    """
    episode_index = episode_index.reshape(-1)
    batch_size = int(episode_index.numel())
    if batch_size < 2 or torch.unique(episode_index).numel() < 2:
        return None
    indices = torch.arange(batch_size, device=episode_index.device)
    offset = max(batch_size // 2, 1)
    replacement = torch.roll(indices, shifts=offset, dims=0)
    for row in range(batch_size):
        if episode_index[replacement[row]] == episode_index[row]:
            candidates = indices[episode_index != episode_index[row]]
            replacement[row] = candidates[row % len(candidates)]
    return replacement


def _sequence_features(sequence: Tensor) -> Tensor:
    """Compact amplitude/variation descriptor for matched action replacement."""
    mean = sequence.mean(dim=1)
    rms = sequence.square().mean(dim=1).sqrt()
    if sequence.shape[1] > 1:
        delta_rms = sequence.diff(dim=1).square().mean(dim=1).sqrt()
    else:
        delta_rms = torch.zeros_like(mean)
    return torch.cat((mean, rms, delta_rms), dim=1)


@torch.no_grad()
def _matched_episode_indices(
    episode_index: Tensor,
    sequence: Tensor,
) -> Tensor | None:
    """Match each action window to a similar window from another episode."""
    episode_index = episode_index.reshape(-1)
    if episode_index.numel() < 2 or torch.unique(episode_index).numel() < 2:
        return None
    features = _sequence_features(sequence).float()
    distances = torch.cdist(features, features)
    same_episode = episode_index[:, None] == episode_index[None, :]
    distances = distances.masked_fill(same_episode, float("inf"))
    distances.fill_diagonal_(float("inf"))
    replacement = distances.argmin(dim=1)
    rows = torch.arange(len(replacement), device=distances.device)
    if not torch.isfinite(distances[rows, replacement]).all():
        return None
    return replacement


@torch.no_grad()
def audit_action_replacement(
    model: PredictiveStateJEPA,
    loader: Iterable[Mapping[str, Tensor]],
    standardizer: Standardizer,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Measure latent action sensitivity without requiring counterfactual data.

    The factual future motor sequence is compared with a physically zero motor
    sequence and with replacements from other episodes.  ``local_action_shift``
    is retained as a near-neighbour sensitivity check, but it is intentionally
    not used as the action-conditioning readiness metric.
    All alternatives use the same history and the same target latent, so this
    is an offline action-conditioning audit rather than a closed-loop claim.
    """
    model.eval()
    horizon = model.config.horizon_steps
    latent_dim = model.config.latent_dim
    alternative_names = (
        "zero",
        "local_action_shift",
        "cross_episode_action",
        "matched_cross_episode_action",
    )
    sums = {
        name: torch.zeros(horizon, dtype=torch.float64)
        for name in ("factual", *alternative_names)
    }
    margin_sums = {name: torch.zeros(horizon, dtype=torch.float64) for name in alternative_names}
    better_counts = {name: torch.zeros(horizon, dtype=torch.float64) for name in alternative_names}
    input_displacement_sums = {name: 0.0 for name in alternative_names}
    variance_sum = torch.zeros(horizon, latent_dim, dtype=torch.float64)
    variance_square_sum = torch.zeros_like(variance_sum)
    target_variance_sum = torch.zeros_like(variance_sum)
    target_variance_square_sum = torch.zeros_like(variance_sum)
    sample_count = 0
    comparison_counts = {name: 0 for name in alternative_names}
    batches = 0
    zero_action = torch.as_tensor(
        -standardizer.action_mean / standardizer.action_std,
        device=device,
        dtype=torch.float32,
    ).view(1, 1, -1)

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        episode_index = raw_batch.get("episode_index")
        if episode_index is not None:
            episode_index = episode_index.to(device=device, dtype=torch.int64)
        if batch_size < 1:
            continue
        factual_output = model(**batch)
        target = factual_output["target_latent"]
        factual = (factual_output["predicted_latent"] - target).square().mean(dim=2)
        zero_future_action = zero_action.expand_as(batch["future_action"])
        zero_prediction = model.predict_latents(
            batch["observation_history"],
            batch["action_history"],
            zero_future_action,
        )
        zero = (zero_prediction - target).square().mean(dim=2)

        sums["factual"] += factual.sum(dim=0).double().cpu()
        sums["zero"] += zero.sum(dim=0).double().cpu()
        margin_sums["zero"] += (zero - factual).sum(dim=0).double().cpu()
        better_counts["zero"] += (factual < zero).sum(dim=0).double().cpu()
        input_displacement_sums["zero"] += float(
            (zero_future_action - batch["future_action"])
            .square()
            .mean(dim=(1, 2))
            .sqrt()
            .sum()
            .cpu()
        )
        comparison_counts["zero"] += batch_size

        predicted = factual_output["predicted_latent"].double().cpu()
        target_cpu = target.double().cpu()
        variance_sum += predicted.sum(dim=0)
        variance_square_sum += predicted.square().sum(dim=0)
        target_variance_sum += target_cpu.sum(dim=0)
        target_variance_square_sum += target_cpu.square().sum(dim=0)
        sample_count += batch_size
        batches += 1

        alternatives: dict[str, Tensor] = {}
        if batch_size >= 2:
            alternatives["local_action_shift"] = torch.roll(
                batch["future_action"], shifts=1, dims=0
            )
            if episode_index is not None:
                cross_indices = _cross_episode_indices(episode_index)
                if cross_indices is not None:
                    alternatives["cross_episode_action"] = batch["future_action"][cross_indices]
                matched_indices = _matched_episode_indices(
                    episode_index, batch["future_action"]
                )
                if matched_indices is not None:
                    alternatives["matched_cross_episode_action"] = batch[
                        "future_action"
                    ][matched_indices]
        for name, replacement_action in alternatives.items():
            replacement_prediction = model.predict_latents(
                batch["observation_history"],
                batch["action_history"],
                replacement_action,
            )
            replacement_error = (replacement_prediction - target).square().mean(dim=2)
            sums[name] += replacement_error.sum(dim=0).double().cpu()
            margin_sums[name] += (replacement_error - factual).sum(dim=0).double().cpu()
            better_counts[name] += (factual < replacement_error).sum(dim=0).double().cpu()
            input_displacement_sums[name] += float(
                (replacement_action - batch["future_action"])
                .square()
                .mean(dim=(1, 2))
                .sqrt()
                .sum()
                .cpu()
            )
            comparison_counts[name] += batch_size

    if batches == 0 or sample_count == 0:
        raise ValueError("empty action-replacement audit loader")
    def _mean(values: Tensor, count: int) -> list[float]:
        return (values / max(count, 1)).tolist()

    def _variance(sum_values: Tensor, square_values: Tensor, count: int) -> list[float]:
        mean = sum_values / max(count, 1)
        return (square_values / max(count, 1) - mean.square()).clamp_min(0.0).mean(dim=1).tolist()

    audit: dict[str, object] = {
        "sample_count": sample_count,
        "comparison_counts": comparison_counts,
        "factual_mse_per_horizon": _mean(sums["factual"], sample_count),
        "zero_mse_per_horizon": _mean(sums["zero"], sample_count),
        "zero_margin_per_horizon": _mean(margin_sums["zero"], sample_count),
        "zero_factual_better_fraction_per_horizon": _mean(
            better_counts["zero"], sample_count
        ),
        "predicted_latent_variance_per_horizon": _variance(
            variance_sum, variance_square_sum, sample_count
        ),
        "target_latent_variance_per_horizon": _variance(
            target_variance_sum, target_variance_square_sum, sample_count
        ),
    }
    for name in alternative_names:
        count = comparison_counts[name]
        prefix = name
        audit[f"{prefix}_mse_per_horizon"] = _mean(sums[name], count) if count else None
        audit[f"{prefix}_margin_per_horizon"] = _mean(margin_sums[name], count) if count else None
        audit[f"{prefix}_factual_better_fraction_per_horizon"] = (
            _mean(better_counts[name], count) if count else None
        )
        audit[f"{prefix}_conditioning_input_rms_displacement"] = (
            input_displacement_sums[name] / count if count else None
        )
    return audit


@torch.no_grad()
def audit_context_replacement(
    model: PredictiveStateJEPA,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
) -> dict[str, object]:
    """Test whether the factual dynamics context improves factual prediction.

    A context from another episode is not a counterfactual physical target.
    This audit only checks whether the model uses its own recent transitions
    more effectively than a provenance-changing context replacement.
    """
    model.eval()
    horizon = model.config.horizon_steps
    factual_sum = torch.zeros(horizon, dtype=torch.float64)
    shuffled_sum = torch.zeros_like(factual_sum)
    zero_sum = torch.zeros_like(factual_sum)
    shuffled_better = torch.zeros_like(factual_sum)
    zero_better = torch.zeros_like(factual_sum)
    shuffled_context_displacement = 0.0
    factual_count = 0
    shuffled_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = batch["future_action"].shape[0]
        if batch_size < 1:
            continue
        output = model(**batch)
        target = output["target_latent"]
        factual = (output["predicted_latent"] - target).square().mean(dim=2)
        factual_sum += factual.sum(dim=0).double().cpu()
        factual_count += batch_size

        full_context = model.encode_context(
            batch["observation_history"], batch["action_history"]
        )
        state = full_context[:, : model.config.latent_dim]
        dynamics_context = full_context[:, model.config.latent_dim :]
        action_embeddings = model.encode_action_windows(
            batch["action_history"], batch["future_action"]
        )
        zero_context = torch.cat((state, torch.zeros_like(dynamics_context)), dim=-1)
        zero_prediction = model.rollout_action_embeddings(
            zero_context, action_embeddings
        )
        zero_error = (zero_prediction - target).square().mean(dim=2)
        zero_sum += zero_error.sum(dim=0).double().cpu()
        zero_better += (factual < zero_error).sum(dim=0).double().cpu()

        episode_index = raw_batch.get("episode_index")
        if episode_index is None:
            continue
        replacement = _cross_episode_indices(
            episode_index.to(device=device, dtype=torch.int64)
        )
        if replacement is None:
            continue
        replacement_dynamics_context = dynamics_context[replacement]
        shuffled_context = torch.cat(
            (state, replacement_dynamics_context), dim=-1
        )
        shuffled_prediction = model.rollout_action_embeddings(
            shuffled_context, action_embeddings
        )
        shuffled_error = (shuffled_prediction - target).square().mean(dim=2)
        shuffled_sum += shuffled_error.sum(dim=0).double().cpu()
        shuffled_better += (factual < shuffled_error).sum(dim=0).double().cpu()
        shuffled_context_displacement += float(
            (dynamics_context - replacement_dynamics_context)
            .square()
            .mean(dim=1)
            .sqrt()
            .sum()
            .cpu()
        )
        shuffled_count += batch_size
    if factual_count < 1:
        raise ValueError("empty context-replacement audit loader")

    def mean(values: Tensor, count: int) -> list[float] | None:
        return (values / count).tolist() if count else None

    return {
        "factual_count": factual_count,
        "shuffled_comparison_count": shuffled_count,
        "factual_mse_per_horizon": mean(factual_sum, factual_count),
        "zero_context_mse_per_horizon": mean(zero_sum, factual_count),
        "zero_context_factual_better_fraction_per_horizon": mean(
            zero_better, factual_count
        ),
        "shuffled_context_mse_per_horizon": mean(shuffled_sum, shuffled_count),
        "shuffled_context_factual_better_fraction_per_horizon": mean(
            shuffled_better, shuffled_count
        ),
        "shuffled_context_input_rms_displacement": (
            shuffled_context_displacement / shuffled_count
            if shuffled_count
            else None
        ),
        "counterfactual_ground_truth": False,
        "interpretation": "factual context-usage audit only",
    }


def _training_readiness(
    metrics: Mapping[str, object],
    action_audit: Mapping[str, object],
    context_audit: Mapping[str, object] | None = None,
    support_query_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    def mean_field(name: str) -> float:
        values = action_audit.get(name)
        if not isinstance(values, list) or not values:
            return math.nan
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    predicted_variance = mean_field("predicted_latent_variance_per_horizon")
    target_variance = mean_field("target_latent_variance_per_horizon")
    cross_episode_margin = mean_field("cross_episode_action_margin_per_horizon")
    cross_episode_better = mean_field(
        "cross_episode_action_factual_better_fraction_per_horizon"
    )
    matched_cross_episode_better = mean_field(
        "matched_cross_episode_action_factual_better_fraction_per_horizon"
    )
    local_shift_better = mean_field(
        "local_action_shift_factual_better_fraction_per_horizon"
    )
    zero_better = mean_field("zero_factual_better_fraction_per_horizon")
    sigreg = float(metrics["sigreg"])
    representation_checks = {
        "predicted_latent_variance": bool(predicted_variance >= 0.05),
        "target_latent_variance": bool(target_variance >= 0.05),
        "sigreg_below_collapse_fixed_point": bool(sigreg < 0.30),
    }
    conditioning_diagnostics = {
        "factual_beats_cross_episode_action": bool(
            cross_episode_margin > 0.0 and cross_episode_better >= 0.75
        ),
        "factual_beats_matched_cross_episode_action": bool(
            matched_cross_episode_better >= 0.65
        ),
        "factual_beats_zero_action": bool(zero_better >= 0.60),
    }
    if context_audit is not None:
        values = context_audit.get(
            "shuffled_context_factual_better_fraction_per_horizon"
        )
        context_better = (
            float(np.mean(np.asarray(values, dtype=np.float64)))
            if isinstance(values, list) and values
            else math.nan
        )
        conditioning_diagnostics["factual_beats_shuffled_dynamics_context"] = bool(
            context_better >= 0.55
        )
    else:
        context_better = math.nan
    if support_query_audit is not None:
        support_query_skill = float(
            support_query_audit["aggregate_normalized_skill_over_zero"]
        )
        support_query_better = float(
            support_query_audit["factual_better_than_shuffled_belief_fraction"]
        )
        conditioning_diagnostics["causal_support_predicts_query_response"] = bool(
            support_query_skill > 0.0 and support_query_better >= 0.55
        )
    else:
        support_query_skill = math.nan
        support_query_better = math.nan
    return {
        "passed": all(representation_checks.values()),
        "checks": representation_checks,
        "conditioning_diagnostics": conditioning_diagnostics,
        "conditioning_diagnostics_are_counterfactual_claims": False,
        "metrics": {
            "predicted_latent_variance_mean": predicted_variance,
            "target_latent_variance_mean": target_variance,
            "sigreg": sigreg,
            "cross_episode_action_margin_mean": cross_episode_margin,
            "cross_episode_action_factual_better_fraction_mean": cross_episode_better,
            "matched_cross_episode_action_factual_better_fraction_mean": (
                matched_cross_episode_better
            ),
            "local_action_shift_factual_better_fraction_mean": local_shift_better,
            "zero_action_factual_better_fraction_mean": zero_better,
            "shuffled_context_factual_better_fraction_mean": context_better,
            "support_query_skill_over_zero": support_query_skill,
            "support_query_factual_better_than_shuffled_belief_fraction": (
                support_query_better
            ),
        },
        "thresholds": {
            "minimum_latent_variance_mean": 0.05,
            "maximum_sigreg": 0.30,
            "minimum_cross_episode_action_margin": 0.0,
            "minimum_cross_episode_action_factual_better_fraction": 0.75,
            "minimum_matched_cross_episode_action_factual_better_fraction": 0.65,
            "minimum_zero_action_factual_better_fraction": 0.60,
            "target_shuffled_context_factual_better_fraction": 0.55,
            "target_support_query_skill_over_zero": 0.0,
            "minimum_support_query_factual_better_than_shuffled_fraction": 0.55,
        },
    }


def _factual_prediction_selection_score(
    metrics: Mapping[str, object], config: Stage1Config
) -> float:
    """Select checkpoints by held-out prediction, not batch-local SIGReg."""
    score = (
        config.rollout_weight * float(metrics["rollout"])
        + config.teacher_forcing_weight * float(metrics["teacher_forcing"])
        + config.backward_transition_weight
        * float(metrics["backward_transition"])
        + config.response_prediction_weight
        * float(metrics["response_prediction"])
        + config.support_query_response_weight
        * float(metrics["support_query_response"])
    )
    if not math.isfinite(score) or score < 0.0:
        raise ValueError("validation factual prediction score must be nonnegative")
    return score


def _datasets(
    manifest_path: str,
    history_seconds: float,
    horizon_seconds: float,
    model_rate_hz: float,
    train_stride: int,
    evaluation_stride: int,
):
    archived_partitions = {
        name: load_partition_sequences(manifest_path, name)
        for name in ("train", "validation", "test")
    }
    recorder_rate = common_sample_rate(sum(archived_partitions.values(), []))
    rate = float(model_rate_hz)
    if rate <= 0.0 or rate > recorder_rate + 0.2:
        raise ValueError("model rate must be positive and no faster than the recorder")
    partitions = {
        name: resample_episode_sequences(episodes, rate)
        for name, episodes in archived_partitions.items()
    }
    standardizer = Standardizer.fit(partitions["train"])
    history_steps = max(2, round(history_seconds * rate) + 1)
    horizon_steps = max(1, round(horizon_seconds * rate))
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
    return partitions, datasets, standardizer, rate, recorder_rate, history_steps, horizon_steps


def _response_delta_std(
    episodes: Iterable[object], standardizer: Standardizer
) -> tuple[float, ...]:
    """Fit train-only scales for one-step standardized motion responses."""
    indices = list(RESPONSE_OBSERVATION_INDICES)
    values: list[np.ndarray] = []
    for episode in episodes:
        observation = episode.observation[:, indices].astype(np.float64)
        valid = np.logical_and(episode.valid[1:], episode.valid[:-1])
        delta = np.diff(observation, axis=0)[valid]
        if len(delta):
            values.append(delta / standardizer.observation_std[indices])
    if not values:
        raise ValueError("cannot fit response scales without valid train transitions")
    scale = np.std(np.concatenate(values, axis=0), axis=0)
    scale = np.maximum(scale, 1e-3)
    if not np.all(np.isfinite(scale)):
        raise ValueError("non-finite response delta scales")
    return tuple(float(value) for value in scale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-seconds", type=float, default=2.48)
    parser.add_argument("--horizon-seconds", type=float, default=3.0)
    parser.add_argument("--model-rate-hz", type=float, default=10.0)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--evaluation-stride", type=int, default=2)
    parser.add_argument("--action-focused-sampling-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sigreg-weight", type=float, default=1.0)
    parser.add_argument("--backward-transition-weight", type=float, default=0.25)
    parser.add_argument("--response-prediction-weight", type=float, default=0.25)
    parser.add_argument("--support-query-response-weight", type=float, default=0.25)
    parser.add_argument("--support-query-seconds", type=float, default=0.7)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(argv)
    if min(
        args.history_seconds,
        args.horizon_seconds,
        args.learning_rate,
        args.final_learning_rate,
        args.backward_transition_weight,
        args.response_prediction_weight,
        args.support_query_response_weight,
        args.support_query_seconds,
    ) <= 0.0:
        parser.error("history, horizon, and learning rate must be positive")
    if not 0.0 <= args.warmup_fraction < 1.0:
        parser.error("warmup fraction must be in [0, 1)")
    if not 0.0 <= args.action_focused_sampling_fraction <= 1.0:
        parser.error("action-focused sampling fraction must be in [0, 1]")
    if args.final_learning_rate > args.learning_rate:
        parser.error("final learning rate cannot exceed peak learning rate")
    if args.epochs < 1 or args.batch_size < 1 or min(
        args.train_stride, args.evaluation_stride
    ) < 1 or args.model_rate_hz <= 0.0 or args.num_workers < 0:
        parser.error("epochs, batch size, and strides must be positive")
    _seed_everything(args.seed)
    device = _device(args.device)
    (
        partitions,
        datasets,
        standardizer,
        rate,
        recorder_rate,
        history_steps,
        horizon_steps,
    ) = _datasets(
        args.manifest,
        args.history_seconds,
        args.horizon_seconds,
        args.model_rate_hz,
        args.train_stride,
        args.evaluation_stride,
    )
    config = Stage1Config(
        horizon_steps=horizon_steps,
        model_step_seconds=1.0 / rate,
        sigreg_weight=args.sigreg_weight,
        backward_transition_weight=args.backward_transition_weight,
        response_prediction_weight=args.response_prediction_weight,
        support_query_response_weight=args.support_query_response_weight,
        support_query_steps=max(1, round(args.support_query_seconds * rate)),
        response_delta_std=_response_delta_std(partitions["train"], standardizer),
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        warmup_fraction=args.warmup_fraction,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
    )
    if history_steps > config.receptive_field_steps:
        raise ValueError(
            f"history has {history_steps} steps but the causal TCN receptive "
            f"field is only {config.receptive_field_steps}"
        )
    model = PredictiveStateJEPA(config).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    sampling_weights = datasets["train"].action_sampling_weights(
        focused_fraction=args.action_focused_sampling_fraction
    )
    sampler_generator = torch.Generator().manual_seed(args.seed)
    train_sampler = WeightedRandomSampler(
        torch.as_tensor(sampling_weights, dtype=torch.double),
        num_samples=len(datasets["train"]),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(datasets["train"], sampler=train_sampler, **loader_options)
    validation_loader = DataLoader(
        datasets["validation"],
        shuffle=False,
        **loader_options,
    )
    total_optimizer_steps = args.epochs * len(train_loader)
    if total_optimizer_steps < 2:
        raise ValueError("training schedule requires at least two optimizer steps")
    warmup_steps = min(
        int(round(config.warmup_fraction * total_optimizer_steps)),
        total_optimizer_steps - 1,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_multiplier(
            step,
            total_steps=total_optimizer_steps,
            warmup_steps=warmup_steps,
            final_ratio=config.final_learning_rate / config.learning_rate,
        ),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_value = math.inf
    best_state = None
    best_validation: dict[str, object] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        learning_rate_start = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scheduler=scheduler,
        )
        validation = evaluate(model, validation_loader, device=device)
        row = {
            "epoch": epoch,
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": validation,
        }
        history.append(row)
        selection_value = _factual_prediction_selection_score(validation, config)
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.6f} "
            f"validation={validation['total']:.6f} "
            f"selection={selection_value:.6f}",
            flush=True,
        )
        row["checkpoint_selection_value"] = selection_value
        if selection_value < best_value:
            best_value = selection_value
            best_state = deepcopy(model.state_dict())
            best_validation = deepcopy(validation)
    if best_state is None or best_validation is None:
        raise RuntimeError("Stage-1 produced no checkpoint")
    model.load_state_dict(best_state)
    selected_validation_metrics = evaluate(
        model,
        validation_loader,
        device=device,
    )
    validation_action_audit = audit_action_replacement(
        model,
        validation_loader,
        standardizer,
        device=device,
    )
    validation_context_audit = audit_context_replacement(
        model,
        validation_loader,
        device=device,
    )
    validation_response_audit = audit_response_prediction(
        model,
        validation_loader,
        standardizer,
        device=device,
    )
    validation_support_query_audit = audit_support_query_prediction(
        model,
        validation_loader,
        standardizer,
        device=device,
    )
    training_readiness = _training_readiness(
        selected_validation_metrics,
        validation_action_audit,
        validation_context_audit,
        validation_support_query_audit,
    )
    test_loader = DataLoader(
        datasets["test"], shuffle=False, **loader_options
    )
    test_metrics = evaluate(model, test_loader, device=device)
    test_action_audit = audit_action_replacement(
        model,
        test_loader,
        standardizer,
        device=device,
    )
    test_context_audit = audit_context_replacement(
        model,
        test_loader,
        device=device,
    )
    test_response_audit = audit_response_prediction(
        model,
        test_loader,
        standardizer,
        device=device,
    )
    test_support_query_audit = audit_support_query_prediction(
        model,
        test_loader,
        standardizer,
        device=device,
    )
    checkpoint_path = output_dir / "stage1_predictive_state_jepa.pt"
    checkpoint = {
        "implementation": IMPLEMENTATION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "observation_labels": list(OBSERVATION_LABELS),
        "action_labels": list(ACTION_LABELS),
        "created_at": datetime.now().astimezone().isoformat(),
        "model_configuration": asdict(config),
        "model_state_dict": model.state_dict(),
        "standardizer": standardizer.as_dict(),
        "manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "history_seconds": (history_steps - 1) / rate,
        "horizon_seconds": horizon_steps / rate,
        "receptive_field_steps": config.receptive_field_steps,
        "receptive_field_seconds": (config.receptive_field_steps - 1) / rate,
        "sample_rate_hz": rate,
        "recorder_sample_rate_hz": recorder_rate,
        "model_step_stride": recorder_rate / rate,
        "temporal_resampling_id": TEMPORAL_RESAMPLING_ID,
        "model_step_seconds": 1.0 / rate,
        "action_interval_seconds": 1.0 / rate,
        "action_contract_id": ACTION_CONTRACT_ID,
        "actuator_model": ACTUATOR_MODEL_ID,
        "training_schedule": {
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "peak_learning_rate": config.learning_rate,
            "final_learning_rate": config.final_learning_rate,
            "action_focused_sampling_fraction": args.action_focused_sampling_fraction,
            "sampling_weight_quantiles": np.quantile(
                sampling_weights, (0.0, 0.25, 0.5, 0.75, 1.0)
            ).tolist(),
        },
        "best_validation_selection_score": best_value,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "selected_validation_metrics": selected_validation_metrics,
        "validation_action_audit": validation_action_audit,
        "validation_context_audit": validation_context_audit,
        "validation_response_audit": validation_response_audit,
        "validation_support_query_audit": validation_support_query_audit,
        "test_metrics": test_metrics,
        "test_action_audit": test_action_audit,
        "test_context_audit": test_context_audit,
        "test_response_audit": test_response_audit,
        "test_support_query_audit": test_support_query_audit,
        "training_readiness": training_readiness,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "implementation": IMPLEMENTATION,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "observation_labels": list(OBSERVATION_LABELS),
        "action_labels": list(ACTION_LABELS),
        "stage": 1,
        "objective": (
            "observation_only_ema_target_response_sufficient_dynamics_context_"
            "within_history_causal_support_query_identification_"
            "action_conditioned_teacher_forcing_recursive_rollout_backward_"
            "transition_physical_response_plus_sigreg"
        ),
        "dynamics_context_contract": {
            "inputs": [
                "standardized_body_velocity_and_gyro_work_point",
                "standardized_observation_delta",
                "standardized_executed_modal_action",
                "standardized_executed_modal_action_delta",
                "standardized_motion_response_action_outer_product",
            ],
            "pooling": "final_full_history_causal_token",
            "fixed_over_imagined_horizon": True,
            "absolute_position_or_yaw_input": False,
            "trained_with_simulator_domain_labels": False,
        },
        "stage1_loss_terms": {
            "recursive_latent_rollout": config.rollout_weight,
            "teacher_forced_forward_transition": config.teacher_forcing_weight,
            "teacher_forced_backward_transition": (
                config.backward_transition_weight
            ),
            "sigreg": config.sigreg_weight,
            "teacher_forced_physical_response": config.response_prediction_weight,
            "causal_support_query_physical_response": (
                config.support_query_response_weight
            ),
        },
        "episode_counts": {name: len(value) for name, value in partitions.items()},
        "window_counts": {name: len(value) for name, value in datasets.items()},
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "history_seconds": (history_steps - 1) / rate,
        "horizon_seconds": horizon_steps / rate,
        "receptive_field_steps": config.receptive_field_steps,
        "receptive_field_seconds": (config.receptive_field_steps - 1) / rate,
        "sample_rate_hz": rate,
        "recorder_sample_rate_hz": recorder_rate,
        "model_step_stride": recorder_rate / rate,
        "temporal_resampling_id": TEMPORAL_RESAMPLING_ID,
        "model_step_seconds": 1.0 / rate,
        "action_interval_seconds": 1.0 / rate,
        "action_contract_id": ACTION_CONTRACT_ID,
        "actuator_model": ACTUATOR_MODEL_ID,
        "training_schedule": {
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "peak_learning_rate": config.learning_rate,
            "final_learning_rate": config.final_learning_rate,
            "action_focused_sampling_fraction": args.action_focused_sampling_fraction,
            "sampling_weight_quantiles": np.quantile(
                sampling_weights, (0.0, 0.25, 0.5, 0.75, 1.0)
            ).tolist(),
        },
        "history": history,
        "best_validation_selection_score": best_value,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "best_validation_metrics": best_validation,
        "selected_validation_metrics": selected_validation_metrics,
        "validation_action_audit": validation_action_audit,
        "validation_context_audit": validation_context_audit,
        "validation_response_audit": validation_response_audit,
        "validation_support_query_audit": validation_support_query_audit,
        "test_metrics": test_metrics,
        "test_action_audit": test_action_audit,
        "test_context_audit": test_context_audit,
        "test_response_audit": test_response_audit,
        "test_support_query_audit": test_support_query_audit,
        "training_readiness": training_readiness,
        "checkpoint_path": str(checkpoint_path),
    }
    report_path = output_dir / "stage1_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
