"""Runtime model and controller for the multi-rate SNN policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from disaware_learning.control.jepa_mppi import JepaMppiConfig, ReferenceTrajectory
from disaware_learning.learning.motor_jepa_data import (
    DEPTH_INDEX,
    OBSERVATION_LABELS,
    Standardizer,
    motor_output_to_modal,
)


IMPLEMENTATION_ID = "multirate_stateful_snn_residual_policy_v1"
ANCHORED_IMPLEMENTATION_ID = "multirate_anchored_snn_correction_policy_v2"
HISTORY_TOKEN_INDEX = 34
REFERENCE_TOKEN_INDEX = 35
CURRENT_ROLL_INDEX = 36
VIO_VALID_INDEX = 37
VIO_AGE_INDEX = 38
POLICY_INPUT_WIDTH = 39
SENSOR_FEATURE_WIDTH = 21
REFERENCE_POSITION_SLICE = slice(21, 24)
REFERENCE_VELOCITY_SLICE = slice(24, 27)
REFERENCE_YAW_INDEX = 27
REFERENCE_YAW_RATE_INDEX = 28
TOKEN_TIME_INDEX = 29
PREVIOUS_COMMAND_SLICE = slice(30, 34)
COMMAND_LIMIT = np.asarray((0.48, 0.48, 0.45, 0.11), dtype=np.float32)
POSITION_SCALE_M = np.asarray((0.50, 0.50, 0.30), dtype=np.float32)
VELOCITY_SCALE_M_S = np.asarray((0.15, 0.15, 0.12), dtype=np.float32)
YAW_SCALE_RAD = 0.35
YAW_RATE_SCALE_RAD_S = 0.20
ROLL_COMMAND_SCALE = 0.90
MOTOR_HARD_LIMIT = 0.90
MOTOR_PROJECTION_LIMIT = MOTOR_HARD_LIMIT - 1e-6
DEFAULT_CORRECTION_LIMIT = np.asarray((0.08, 0.08, 0.07, 0.02), dtype=np.float32)


def discrete_model_tick_due(*, macro_boundary: bool) -> bool:
    """Schedule synchronous inference from the exact recorder/model ratio."""
    return bool(macro_boundary)


@dataclass(frozen=True)
class AnchoredFeedbackConfig:
    """Stateless recovery controller used as the SNN action anchor."""

    reference_velocity_gain: tuple[float, float, float] = (
        1.9837,
        1.9122,
        1.8358,
    )
    velocity_error_gain: tuple[float, float, float] = (2.4535, 3.2691, 5.7648)
    position_error_gain: tuple[float, float, float] = (0.4578, 0.7119, 1.8786)
    position_command_limit: tuple[float, float, float] = (0.12, 0.12, 0.18)
    yaw_rate_feedforward_gain: float = 0.2423
    yaw_rate_error_gain: float = 0.1931
    yaw_error_gain: float = 0.0356
    yaw_position_command_limit: float = 0.03

    def __post_init__(self) -> None:
        for name in (
            "reference_velocity_gain",
            "velocity_error_gain",
            "position_error_gain",
            "position_command_limit",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values")
            if np.any(value < 0.0):
                raise ValueError(f"{name} cannot contain negative values")
        for name in (
            "yaw_rate_feedforward_gain",
            "yaw_rate_error_gain",
            "yaw_error_gain",
            "yaw_position_command_limit",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _surrogate_spike(membrane_minus_threshold: Tensor, slope: float) -> Tensor:
    hard = (membrane_minus_threshold >= 0.0).to(membrane_minus_threshold.dtype)
    soft = torch.sigmoid(float(slope) * membrane_minus_threshold)
    return hard.detach() + soft - soft.detach()


@dataclass
class _LIFState:
    membrane: Tensor
    spike: Tensor
    adaptation: Tensor


class AdaptiveLIFCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        membrane_decay: float = 0.90,
        adaptation_decay: float = 0.95,
        threshold: float = 1.0,
        surrogate_slope: float = 5.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.recurrent_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.membrane_decay_logit = nn.Parameter(
            torch.full(
                (hidden_size,),
                math.log(membrane_decay / (1.0 - membrane_decay)),
                dtype=torch.float32,
            )
        )
        self.adaptation_decay_logit = nn.Parameter(
            torch.full(
                (hidden_size,),
                math.log(adaptation_decay / (1.0 - adaptation_decay)),
                dtype=torch.float32,
            )
        )
        self.register_buffer("base_threshold", torch.tensor(float(threshold)))
        self.adaptation_gain_raw = nn.Parameter(torch.tensor(-1.5))
        self.surrogate_slope = float(surrogate_slope)

    def initial_state(self, inputs: Tensor) -> _LIFState:
        zeros = inputs.new_zeros((inputs.shape[0], self.hidden_size))
        return _LIFState(zeros, zeros, zeros)

    def forward(self, inputs: Tensor, state: _LIFState) -> _LIFState:
        adaptation_gain = torch.nn.functional.softplus(self.adaptation_gain_raw)
        threshold = self.base_threshold + adaptation_gain * state.adaptation
        current = self.input_projection(inputs) + self.recurrent_projection(state.spike)
        membrane_decay = torch.sigmoid(self.membrane_decay_logit)
        adaptation_decay = torch.sigmoid(self.adaptation_decay_logit)
        membrane = membrane_decay * state.membrane + current
        spike = _surrogate_spike(membrane - threshold, self.surrogate_slope)
        membrane = membrane - spike * threshold
        adaptation = adaptation_decay * state.adaptation + spike
        return _LIFState(membrane, spike, adaptation)


class ReferencePreviewEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        encoded = self.network(tokens.transpose(1, 2))
        return torch.nn.functional.silu(
            self.pool(torch.cat((encoded.mean(dim=-1), encoded[:, :, -1]), dim=-1))
        )


def _quantized_signed_events(
    sensor: Tensor,
    previous_sensor: Tensor,
    *,
    threshold: float,
    maximum_level: int,
    refresh: bool,
) -> Tensor:
    change = sensor if refresh else sensor - previous_sensor
    magnitude = torch.floor(change.abs() / threshold).clamp(max=maximum_level)
    signed = torch.sign(change) * magnitude / float(maximum_level)
    return torch.cat((torch.relu(signed), torch.relu(-signed)), dim=-1)


@dataclass
class MultiRateSnnState:
    layer1: _LIFState
    layer2: _LIFState
    previous_sensor: Tensor
    tick: int


class MultiRateSpikingResidualPolicy(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        history_steps: int,
        reference_hidden_size: int = 32,
        hidden_size_1: int = 72,
        hidden_size_2: int = 56,
        fusion_size: int = 64,
        event_threshold: float = 0.08,
        maximum_event_level: int = 4,
        refresh_interval_ticks: int = 5,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.history_steps = int(history_steps)
        self.event_threshold = float(event_threshold)
        self.maximum_event_level = int(maximum_event_level)
        self.refresh_interval_ticks = int(refresh_interval_ticks)
        self.reference_encoder = ReferencePreviewEncoder(
            self.input_size, reference_hidden_size
        )
        self.layer1 = AdaptiveLIFCell(2 * SENSOR_FEATURE_WIDTH + 3, hidden_size_1)
        self.layer2 = AdaptiveLIFCell(hidden_size_1, hidden_size_2)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size_2 + reference_hidden_size, fusion_size),
            nn.SiLU(),
            nn.Linear(fusion_size, 4),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def encode_reference(self, tokens: Tensor) -> Tensor:
        return self.reference_encoder(tokens)

    def initial_state(self, token: Tensor) -> MultiRateSnnState:
        return MultiRateSnnState(
            self.layer1.initial_state(token),
            self.layer2.initial_state(token),
            token.new_zeros((len(token), SENSOR_FEATURE_WIDTH)),
            0,
        )

    def step(
        self,
        token: Tensor,
        reference_context: Tensor,
        state: MultiRateSnnState,
    ) -> tuple[Tensor, MultiRateSnnState, dict[str, Tensor]]:
        refresh = state.tick % self.refresh_interval_ticks == 0
        sensor = token[:, :SENSOR_FEATURE_WIDTH]
        events = _quantized_signed_events(
            sensor,
            state.previous_sensor,
            threshold=self.event_threshold,
            maximum_level=self.maximum_event_level,
            refresh=refresh,
        )
        auxiliary = torch.stack(
            (
                token[:, VIO_VALID_INDEX],
                token[:, VIO_AGE_INDEX],
                token.new_full((len(token),), float(refresh)),
            ),
            dim=-1,
        )
        layer1 = self.layer1(torch.cat((events, auxiliary), dim=-1), state.layer1)
        layer2 = self.layer2(layer1.spike, state.layer2)
        prediction = self.fusion(torch.cat((layer2.membrane, reference_context), dim=-1))
        return (
            prediction,
            MultiRateSnnState(layer1, layer2, sensor, state.tick + 1),
            {
                "firing_rate": 0.5 * (layer1.spike.mean() + layer2.spike.mean()),
                "input_event_rate": (events > 0.0).to(events.dtype).mean(),
            },
        )


@dataclass(frozen=True)
class SnnPolicyResult:
    command: np.ndarray
    nominal_command: np.ndarray
    accepted: bool
    reason: str
    diagnostics: dict[str, Any]
    predicted_physical_observation: np.ndarray
    predicted_position_delta_ned_m: np.ndarray | None


def _project_command(command: np.ndarray, roll_command: float) -> np.ndarray:
    values = np.clip(np.asarray(command, dtype=np.float64), -COMMAND_LIMIT, COMMAND_LIMIT)
    surge, sway, heave, yaw = values
    horizontal = np.asarray(
        (
            yaw - surge + sway,
            -yaw - surge - sway,
            -yaw + surge + sway,
            yaw + surge - sway,
        )
    )
    scale = min(
        1.0,
        MOTOR_PROJECTION_LIMIT / max(float(np.max(np.abs(horizontal))), 1e-12),
    )
    values[(0, 1, 3),] *= scale
    heave_limit = min(
        float(COMMAND_LIMIT[2]),
        max(0.0, MOTOR_PROJECTION_LIMIT - abs(roll_command)),
    )
    values[2] = np.clip(values[2], -heave_limit, heave_limit)
    return values.astype(np.float32)


def anchored_feedback_command(
    current_observation: np.ndarray,
    current_position_ned_m: np.ndarray,
    reference: ReferenceTrajectory,
    roll_command: float,
    *,
    config: AnchoredFeedbackConfig = AnchoredFeedbackConfig(),
) -> np.ndarray:
    """Compute an absolute recoverable command from the first preview knot."""
    observation = np.asarray(current_observation, dtype=np.float64)
    position = np.asarray(current_position_ned_m, dtype=np.float64)
    if observation.shape != (len(OBSERVATION_LABELS),):
        raise ValueError("anchored feedback observation width mismatch")
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("anchored feedback position must contain three finite values")
    if not np.all(np.isfinite(observation)):
        raise ValueError("anchored feedback observation must be finite")

    first_column = observation[:3].copy()
    first_column /= max(float(np.linalg.norm(first_column)), 1e-9)
    second_column = observation[3:6].copy()
    second_column -= first_column * float(np.dot(first_column, second_column))
    second_column /= max(float(np.linalg.norm(second_column)), 1e-9)
    rotation_body_to_ned = np.column_stack(
        (first_column, second_column, np.cross(first_column, second_column))
    )
    position_error_ned = (
        np.asarray(reference.position_ned_m[0], dtype=np.float64) - position
    )
    position_error_body = rotation_body_to_ned.T @ position_error_ned
    reference_velocity = np.asarray(
        reference.linear_velocity_body_m_s[0], dtype=np.float64
    )
    measured_velocity = observation[6:9]
    position_term = np.clip(
        np.asarray(config.position_error_gain) * position_error_body,
        -np.asarray(config.position_command_limit),
        np.asarray(config.position_command_limit),
    )
    linear_frd = (
        np.asarray(config.reference_velocity_gain) * reference_velocity
        + np.asarray(config.velocity_error_gain)
        * (reference_velocity - measured_velocity)
        + position_term
    )
    current_yaw = float(math.atan2(observation[1], observation[0]))
    yaw_error = math.atan2(
        math.sin(float(reference.yaw_rad[0]) - current_yaw),
        math.cos(float(reference.yaw_rad[0]) - current_yaw),
    )
    reference_yaw_rate = float(reference.yaw_rate_rad_s[0])
    yaw_position_term = float(
        np.clip(
            config.yaw_error_gain * yaw_error,
            -config.yaw_position_command_limit,
            config.yaw_position_command_limit,
        )
    )
    yaw = (
        config.yaw_rate_feedforward_gain * reference_yaw_rate
        + config.yaw_rate_error_gain * (reference_yaw_rate - observation[11])
        + yaw_position_term
    )
    return _project_command(
        np.asarray((linear_frd[0], linear_frd[1], -linear_frd[2], yaw)),
        float(roll_command),
    )


class StatefulSnnPolicyController:
    """Deploy the 10 Hz JEPA residual teacher as bounded 25 Hz SNN updates."""

    def __init__(
        self,
        model: MultiRateSpikingResidualPolicy,
        standardizer: Standardizer,
        residual_mean: np.ndarray,
        residual_std: np.ndarray,
        *,
        config: JepaMppiConfig,
        device: torch.device,
        history_steps: int,
        horizon_steps: int,
        sample_rate_hz: float,
        teacher_rate_hz: float,
        checkpoint_path: Path,
        control_mode: str = "legacy_integrated_residual",
        feedback_config: AnchoredFeedbackConfig = AnchoredFeedbackConfig(),
        correction_limit: np.ndarray = DEFAULT_CORRECTION_LIMIT,
        correction_gain: np.ndarray | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.standardizer = standardizer
        self.residual_mean = np.asarray(residual_mean, dtype=np.float32)
        self.residual_std = np.asarray(residual_std, dtype=np.float32)
        self.config = config
        self.device = device
        self.history_steps = int(history_steps)
        self.model_horizon_steps = int(horizon_steps)
        self.rollout_horizon_steps = int(horizon_steps)
        self.sample_rate_hz = float(sample_rate_hz)
        self.teacher_rate_hz = float(teacher_rate_hz)
        self.checkpoint_path = str(checkpoint_path)
        self.control_mode = str(control_mode)
        if self.control_mode not in (
            "legacy_integrated_residual",
            "anchored_bounded_correction",
        ):
            raise ValueError(f"unknown SNN control mode: {self.control_mode}")
        self.feedback_config = feedback_config
        self.correction_limit = np.asarray(correction_limit, dtype=np.float32)
        if (
            self.correction_limit.shape != (4,)
            or not np.all(np.isfinite(self.correction_limit))
            or np.any(self.correction_limit <= 0.0)
        ):
            raise ValueError("SNN correction limits must contain four positive values")
        self.correction_gain = np.asarray(
            np.ones(4, dtype=np.float32)
            if correction_gain is None
            else correction_gain,
            dtype=np.float32,
        )
        if (
            self.correction_gain.shape != (4,)
            or not np.all(np.isfinite(self.correction_gain))
            or np.any(self.correction_gain < 0.0)
            or np.any(self.correction_gain > 1.0)
        ):
            raise ValueError("SNN correction gains must lie in [0, 1]")
        self.implementation_id = (
            ANCHORED_IMPLEMENTATION_ID
            if self.control_mode == "anchored_bounded_correction"
            else IMPLEMENTATION_ID
        )
        self._state: MultiRateSnnState | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: JepaMppiConfig,
        device: str | torch.device,
    ) -> "StatefulSnnPolicyController":
        path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        implementation = checkpoint.get("implementation")
        if implementation not in (
            "multirate_stateful_snn_controller_v1",
            "multirate_anchored_snn_controller_v2",
        ):
            raise ValueError("checkpoint is not a multi-rate SNN controller")
        if checkpoint.get("kind") != "multirate_snn":
            raise ValueError("checkpoint does not contain the SNN policy")
        metadata = checkpoint["dataset_metadata"]
        train_config = checkpoint["config"]
        input_width = int(metadata["input_width"])
        if input_width != POLICY_INPUT_WIDTH:
            raise ValueError(f"SNN input contract mismatch: {input_width}")
        history_steps = int(metadata["history_steps"])
        model = MultiRateSpikingResidualPolicy(
            input_width,
            history_steps=history_steps,
            reference_hidden_size=int(train_config["reference_hidden_size"]),
            hidden_size_1=int(train_config["snn_hidden_size_1"]),
            hidden_size_2=int(train_config["snn_hidden_size_2"]),
            fusion_size=int(train_config["fusion_size"]),
            event_threshold=float(train_config["event_threshold"]),
            maximum_event_level=int(train_config["maximum_event_level"]),
            refresh_interval_ticks=int(train_config["refresh_interval_ticks"]),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        stage2_path = Path(str(metadata["checkpoint_path"])).expanduser()
        if not stage2_path.is_file():
            raise FileNotFoundError(f"missing standardizer checkpoint: {stage2_path}")
        stage2 = torch.load(stage2_path, map_location="cpu", weights_only=False)
        control_mode = (
            "anchored_bounded_correction"
            if implementation == "multirate_anchored_snn_controller_v2"
            else "legacy_integrated_residual"
        )
        feedback_config = AnchoredFeedbackConfig(
            **checkpoint.get("feedback_config", {})
        )
        correction_limit = np.asarray(
            checkpoint.get("correction_limit", DEFAULT_CORRECTION_LIMIT),
            dtype=np.float32,
        )
        correction_gain = np.asarray(
            checkpoint.get("correction_gain", np.ones(4, dtype=np.float32)),
            dtype=np.float32,
        )
        return cls(
            model,
            Standardizer.from_dict(stage2["standardizer"]),
            np.asarray(checkpoint["residual_mean"], dtype=np.float32),
            np.asarray(checkpoint["residual_std"], dtype=np.float32),
            config=config,
            device=torch.device(device),
            history_steps=history_steps,
            horizon_steps=int(metadata["horizon_steps"]),
            sample_rate_hz=float(metadata["sample_rate_hz"]),
            teacher_rate_hz=float(metadata["reference_sample_rate_hz"]),
            checkpoint_path=path,
            control_mode=control_mode,
            feedback_config=feedback_config,
            correction_limit=correction_limit,
            correction_gain=correction_gain,
        )

    def reset(self, previous_command: np.ndarray | None = None) -> None:
        self._state = None

    def reset_plan(self, previous_command: np.ndarray | None = None) -> None:
        self._state = None

    def _tokens(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reference: ReferenceTrajectory,
        previous_command: np.ndarray,
        current_position_ned_m: np.ndarray,
        roll_command: float,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).copy()
        action = np.asarray(action, dtype=np.float32)
        if observation.shape != (self.history_steps, len(OBSERVATION_LABELS)):
            raise ValueError("SNN observation history shape mismatch")
        if action.shape != (self.history_steps, 6):
            raise ValueError("SNN motor history shape mismatch")
        observation[:, DEPTH_INDEX] -= observation[-1, DEPTH_INDEX]
        normalized_observation = (
            observation - self.standardizer.observation_mean[None, :]
        ) / self.standardizer.observation_std[None, :]
        modal = motor_output_to_modal(action)
        normalized_action = (
            modal - self.standardizer.action_mean[None, :]
        ) / self.standardizer.action_std[None, :]
        tokens = np.zeros(
            (self.history_steps + self.model_horizon_steps, POLICY_INPUT_WIDTH),
            dtype=np.float32,
        )
        tokens[: self.history_steps, :16] = normalized_observation
        tokens[: self.history_steps, 16:21] = normalized_action
        tokens[: self.history_steps, TOKEN_TIME_INDEX] = np.linspace(
            -1.0, 0.0, self.history_steps, dtype=np.float32
        )
        tokens[: self.history_steps, HISTORY_TOKEN_INDEX] = 1.0
        tokens[: self.history_steps, VIO_VALID_INDEX] = 1.0
        start = self.history_steps
        current_position = np.asarray(current_position_ned_m, dtype=np.float32)
        tokens[start:, REFERENCE_POSITION_SLICE] = (
            np.asarray(reference.position_ned_m, dtype=np.float32)
            - current_position[None, :]
        ) / POSITION_SCALE_M[None, :]
        tokens[start:, REFERENCE_VELOCITY_SLICE] = (
            np.asarray(reference.linear_velocity_body_m_s, dtype=np.float32)
            / VELOCITY_SCALE_M_S[None, :]
        )
        current_rotation = observation[-1, :6]
        current_yaw = float(np.arctan2(current_rotation[1], current_rotation[0]))
        reference_yaw = np.asarray(reference.yaw_rad, dtype=np.float32)
        tokens[start:, REFERENCE_YAW_INDEX] = np.arctan2(
            np.sin(reference_yaw - current_yaw), np.cos(reference_yaw - current_yaw)
        ) / YAW_SCALE_RAD
        tokens[start:, REFERENCE_YAW_RATE_INDEX] = (
            np.asarray(reference.yaw_rate_rad_s, dtype=np.float32)
            / YAW_RATE_SCALE_RAD_S
        )
        tokens[start:, TOKEN_TIME_INDEX] = np.arange(
            1, self.model_horizon_steps + 1, dtype=np.float32
        ) / self.model_horizon_steps
        tokens[start:, PREVIOUS_COMMAND_SLICE] = (
            np.asarray(previous_command, dtype=np.float32)[None, :]
            / COMMAND_LIMIT[None, :]
        )
        tokens[start:, CURRENT_ROLL_INDEX] = float(roll_command) / ROLL_COMMAND_SCALE
        tokens[start:, REFERENCE_TOKEN_INDEX] = 1.0
        return tokens

    @torch.no_grad()
    def _infer(self, tokens: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        values = torch.from_numpy(tokens[None]).to(self.device)
        reference_context = self.model.encode_reference(values[:, self.history_steps :])
        if self._state is None:
            history = values[:, : self.history_steps]
            state = self.model.initial_state(history[:, 0])
            prediction = None
            diagnostics: dict[str, Tensor] = {}
            firing_sum = 0.0
            event_sum = 0.0
            for index in range(self.history_steps):
                prediction, state, diagnostics = self.model.step(
                    history[:, index], reference_context, state
                )
                firing_sum += float(diagnostics["firing_rate"].cpu())
                event_sum += float(diagnostics["input_event_rate"].cpu())
            self._state = state
            summary = {
                "firing_rate": firing_sum / self.history_steps,
                "input_event_rate": event_sum / self.history_steps,
                "cold_history_unroll": 1.0,
            }
        else:
            prediction, self._state, diagnostics = self.model.step(
                values[:, self.history_steps - 1], reference_context, self._state
            )
            summary = {
                "firing_rate": float(diagnostics["firing_rate"].cpu()),
                "input_event_rate": float(diagnostics["input_event_rate"].cpu()),
                "cold_history_unroll": 0.0,
            }
        assert prediction is not None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        normalized = prediction.cpu().numpy()[0]
        return normalized * self.residual_std + self.residual_mean, summary

    def warmup(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        previous_command: np.ndarray,
        reference: ReferenceTrajectory,
        *,
        current_position_ned_m: np.ndarray,
        roll_command: float,
    ) -> float:
        started = time.perf_counter()
        token_anchor = np.asarray(previous_command, dtype=np.float32)
        if self.control_mode == "anchored_bounded_correction":
            token_anchor = anchored_feedback_command(
                np.asarray(observation, dtype=np.float32)[-1],
                current_position_ned_m,
                reference,
                float(roll_command),
                config=self.feedback_config,
            )
        tokens = self._tokens(
            observation,
            action,
            reference,
            token_anchor,
            current_position_ned_m,
            roll_command,
        )
        self._state = None
        self._infer(tokens)
        self._state = None
        return time.perf_counter() - started

    def optimize(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reference: ReferenceTrajectory,
        previous_command: np.ndarray,
        *,
        current_position_ned_m: np.ndarray,
        roll_command: float | np.ndarray = 0.0,
        **_: Any,
    ) -> SnnPolicyResult:
        started = time.perf_counter()
        roll = float(np.asarray(roll_command).reshape(-1)[0])
        previous = np.asarray(previous_command, dtype=np.float32)
        nominal = previous.copy()
        if self.control_mode == "anchored_bounded_correction":
            nominal = anchored_feedback_command(
                np.asarray(observation, dtype=np.float32)[-1],
                current_position_ned_m,
                reference,
                roll,
                config=self.feedback_config,
            )
        tokens = self._tokens(
            observation,
            action,
            reference,
            nominal,
            current_position_ned_m,
            roll,
        )
        inference_started = time.perf_counter()
        predicted_correction, spike = self._infer(tokens)
        inference_seconds = time.perf_counter() - inference_started
        if self.control_mode == "anchored_bounded_correction":
            applied_correction = np.clip(
                predicted_correction, -self.correction_limit, self.correction_limit
            )
            applied_correction = self.correction_gain * applied_correction
            update_fraction = 1.0
            desired = nominal + applied_correction
        else:
            update_fraction = min(1.0, self.teacher_rate_hz / self.sample_rate_hz)
            applied_correction = update_fraction * predicted_correction
            desired = previous + applied_correction
        slew = np.asarray(self.config.command_slew_per_second, dtype=np.float32)
        maximum_step = slew / self.sample_rate_hz
        desired = np.clip(desired, previous - maximum_step, previous + maximum_step)
        command = _project_command(desired, roll)
        total = time.perf_counter() - started
        diagnostics = {
            "timing_seconds": total,
            "core_optimization_seconds": inference_seconds,
            "postprocess_seconds": total - inference_seconds,
            "total_optimize_seconds": total,
            "nominal_cost": math.nan,
            "selected_cost": math.nan,
            "predicted_improvement": 0.0,
            "previous_command": previous.copy(),
            "nominal_feedback_command": nominal.copy(),
            "teacher_residual": predicted_correction.copy(),
            "predicted_correction": predicted_correction.copy(),
            "applied_correction": applied_correction.copy(),
            "correction_limit": self.correction_limit.copy(),
            "correction_gain": self.correction_gain.copy(),
            "control_mode": self.control_mode,
            "execution_update_fraction": update_fraction,
            "firing_rate": spike["firing_rate"],
            "input_event_rate": spike["input_event_rate"],
            "cold_history_unroll": bool(spike["cold_history_unroll"]),
        }
        return SnnPolicyResult(
            command=command,
            nominal_command=np.repeat(
                nominal[None, :], self.model_horizon_steps, axis=0
            ),
            accepted=True,
            reason=(
                "anchored_snn_correction_update"
                if self.control_mode == "anchored_bounded_correction"
                else "stateful_snn_policy_update"
            ),
            diagnostics=diagnostics,
            predicted_physical_observation=np.full(
                (self.rollout_horizon_steps, 16), np.nan, dtype=np.float32
            ),
            predicted_position_delta_ned_m=None,
        )
