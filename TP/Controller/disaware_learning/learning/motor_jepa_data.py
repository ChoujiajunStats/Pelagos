"""Episode-safe windows for motor-conditioned JEPA training."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as error:  # pragma: no cover - optional training dependency.
    raise ImportError("motor JEPA training requires PyTorch") from error

from disaware_learning.learning.jepa_manifest import load_dataset_manifest
from disaware_learning.data.episode_schema import OBSERVATION_FEATURE_CONTRACT_ID
from disaware_learning.estimation.pressure_depth import PRESSURE_DEPTH_SOURCE_ID

FEATURE_CONTRACT_ID = OBSERVATION_FEATURE_CONTRACT_ID
GRAVITY_M_S2 = 9.80665
OBSERVATION_LABELS = (
    # First two columns of the body-to-NED rotation matrix, flattened by
    # columns. The source quaternion remains in the episode archive.
    "vio_rotation6d_r00",
    "vio_rotation6d_r10",
    "vio_rotation6d_r20",
    "vio_rotation6d_r01",
    "vio_rotation6d_r11",
    "vio_rotation6d_r21",
    "vio_velocity_x_m_s",
    "vio_velocity_y_m_s",
    "vio_velocity_z_m_s",
    "imu_gyro_x_rad_s",
    "imu_gyro_y_rad_s",
    "imu_gyro_z_rad_s",
    "dynamic_accel_x_m_s2",
    "dynamic_accel_y_m_s2",
    "dynamic_accel_z_m_s2",
    "relative_depth_down_m",
)
ACTION_LABELS = (
    "actuator_modal_surge",
    "actuator_modal_sway",
    "actuator_modal_heave",
    "actuator_modal_yaw",
    "actuator_modal_roll",
)
# The archive retains six post-mixer motor demands. On the standard six-
# thruster allocation manifold these five generalized actuator coordinates are
# lossless; the omitted horizontal null-space component is identically zero.
ACTION_SOURCE_FIELD = "motor_output_normalized"
# Stage 2 and model-based controllers consume the same deployable action
# contract: ArduSub's six post-mixer motor demands projected onto the
# five-dimensional BlueROV2 mixer manifold. These are commands, not measured
# thrust or rotor speed; the downstream world model therefore retains an
# explicit actuator-lag state.
ACTION_CONTRACT_ID = "ardusub_mixer_demand_six_to_five_modal_v2"
ACTUATOR_MODEL_ID = "ardusub_mixer_demand_to_modal_with_actuator_lag"
TEMPORAL_RESAMPLING_ID = "linear_observation_forward_interval_mean_action_v1"
ROTATION_6D_SLICE = slice(0, 6)
VIO_VELOCITY_SLICE = slice(6, 9)
IMU_GYRO_SLICE = slice(9, 12)
YAW_RATE_INDEX = 11
IMU_ACCEL_SLICE = slice(12, 15)
DEPTH_INDEX = 15
MOTION_PROFILE_IDS = {"local_dynamics": 0, "transit_dynamics": 1}
MANUAL_CONTROL_AXES = ("surge", "sway", "heave", "yaw")


def motor_output_to_modal(values: np.ndarray) -> np.ndarray:
    """Convert six BlueROV2 motor demands to five generalized coordinates."""
    motor = np.asarray(values, dtype=np.float32)
    if motor.ndim < 1 or motor.shape[-1] != 6:
        raise ValueError("motor output must end in six channels")
    m1, m2, m3, m4, m5, m6 = np.moveaxis(motor, -1, 0)
    return np.stack(
        (
            (-m1 - m2 + m3 + m4) / 4.0,
            (m1 - m2 + m3 - m4) / 4.0,
            -(m5 + m6) / 2.0,
            (m1 - m2 - m3 + m4) / 4.0,
            (m5 - m6) / 2.0,
        ),
        axis=-1,
    ).astype(np.float32)


def modal_to_motor_output(values: np.ndarray) -> np.ndarray:
    """Reconstruct six motor demands on the standard mixer manifold."""
    modal = np.asarray(values, dtype=np.float32)
    if modal.ndim < 1 or modal.shape[-1] != len(ACTION_LABELS):
        raise ValueError("modal action must end in five channels")
    surge, sway, heave, yaw, roll = np.moveaxis(modal, -1, 0)
    return np.stack(
        (
            yaw - surge + sway,
            -yaw - surge - sway,
            -yaw + surge + sway,
            yaw + surge - sway,
            roll - heave,
            -roll - heave,
        ),
        axis=-1,
    ).astype(np.float32)


def _canonical_quaternion(values: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float32).copy()
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion /= np.maximum(norms, 1e-8)
    if quaternion[0, 3] < 0.0:
        quaternion[0] *= -1.0
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index], quaternion[index - 1]) < 0.0:
            quaternion[index] *= -1.0
    return quaternion


def quaternion_to_rotation6d(values: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to the first two columns of body-to-NED R."""
    quaternion = np.asarray(values, dtype=np.float32)
    if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
        raise ValueError("quaternion array must have shape [samples, 4]")
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8
    )
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    matrix = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ),
        axis=-1,
    )
    return matrix.astype(np.float32)


def gravity_compensated_body_acceleration(
    specific_force_body_frd_m_s2: np.ndarray,
    orientation_body_to_ned_xyzw: np.ndarray,
    *,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> np.ndarray:
    """Convert IMU specific force to dynamic body-frame acceleration.

    ArduSub's ``HIGHRES_IMU`` reports accelerometer specific force.  With the
    vehicle stationary and aligned to NED/FRD this is approximately
    ``[0, 0, -g]``.  The kinematic acceleration used by Stage 2 is therefore
    ``a_body = f_body + R_body_to_ned.T @ [0, 0, g]``.

    The function is vectorized over any common leading shape and intentionally
    uses only the causal attitude estimate supplied with the observation.
    """
    specific_force = np.asarray(specific_force_body_frd_m_s2, dtype=np.float64)
    quaternion = np.asarray(orientation_body_to_ned_xyzw, dtype=np.float64)
    if specific_force.shape[-1:] != (3,) or quaternion.shape[-1:] != (4,):
        raise ValueError("specific force and orientation have invalid widths")
    if specific_force.shape[:-1] != quaternion.shape[:-1]:
        raise ValueError("specific force and orientation leading shapes differ")
    if not np.all(np.isfinite(specific_force)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("specific force and orientation must be finite")
    if not math.isfinite(float(gravity_m_s2)) or gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be positive and finite")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 1e-9):
        raise ValueError("orientation quaternion cannot be zero")
    quaternion = quaternion / norm
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    rotation = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    gravity_ned = np.zeros((*quaternion.shape[:-1], 3), dtype=np.float64)
    gravity_ned[..., 2] = float(gravity_m_s2)
    gravity_body = np.einsum("...ji,...j->...i", rotation, gravity_ned)
    return (specific_force + gravity_body).astype(np.float32)


@dataclass(frozen=True)
class EpisodeSequence:
    job_id: str
    joint_split_cell: str
    partition: str
    time: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    valid: np.ndarray
    target_start_seconds: float
    target_end_seconds: float
    sample_rate_hz: float
    motion_profile: str
    active_axes: tuple[str, ...]
    manual_control: np.ndarray
    deployable_position_ned_m: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def fit(cls, episodes: Sequence[EpisodeSequence]) -> "Standardizer":
        if not episodes:
            raise ValueError("cannot fit a standardizer without train episodes")
        observation_sum = np.zeros(len(OBSERVATION_LABELS), dtype=np.float64)
        observation_square_sum = np.zeros_like(observation_sum)
        action_sum = np.zeros(len(ACTION_LABELS), dtype=np.float64)
        action_square_sum = np.zeros_like(action_sum)
        observation_count = 0
        action_count = 0
        for episode in episodes:
            observation = episode.observation[episode.valid].astype(np.float64)
            action = episode.action[episode.valid].astype(np.float64)
            observation_sum += observation.sum(axis=0)
            observation_square_sum += np.square(observation).sum(axis=0)
            action_sum += action.sum(axis=0)
            action_square_sum += np.square(action).sum(axis=0)
            observation_count += len(observation)
            action_count += len(action)
        observation_mean = observation_sum / observation_count
        action_mean = action_sum / action_count
        observation_variance = observation_square_sum / observation_count - np.square(
            observation_mean
        )
        action_variance = action_square_sum / action_count - np.square(action_mean)
        observation_std = np.sqrt(np.maximum(observation_variance, 1e-8))
        action_std = np.sqrt(np.maximum(action_variance, 1e-8))

        # Depth and pressure are anchored to the current window, not a tank datum.
        observation_mean[DEPTH_INDEX] = 0.0
        observation_std = np.maximum(observation_std, 1e-3)
        action_std = np.maximum(action_std, 1e-3)
        return cls(
            observation_mean.astype(np.float32),
            observation_std.astype(np.float32),
            action_mean.astype(np.float32),
            action_std.astype(np.float32),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "observation_mean": self.observation_mean.tolist(),
            "observation_std": self.observation_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Standardizer":
        return cls(
            np.asarray(values["observation_mean"], dtype=np.float32),
            np.asarray(values["observation_std"], dtype=np.float32),
            np.asarray(values["action_mean"], dtype=np.float32),
            np.asarray(values["action_std"], dtype=np.float32),
        )


def _load_episode(entry: dict[str, Any]) -> EpisodeSequence:
    episode_path = Path(entry["admitted_episode_path"])
    sidecar_path = Path(entry["episode_sidecar_path"])
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    feature_contract = metadata.get("training_contract", {}).get(
        "observation_feature_contract"
    )
    if feature_contract != FEATURE_CONTRACT_ID:
        raise ValueError(
            "episode observation feature contract mismatch: "
            f"expected {FEATURE_CONTRACT_ID}, got {feature_contract!r}: {episode_path}"
        )
    if metadata.get("ground_truth_recording", {}).get("enabled") is not False:
        raise ValueError(f"admitted episode must disable ground-truth recording: {episode_path}")
    depth_sensor = metadata.get("depth_sensor", {})
    controller_depth = metadata.get("controller_depth_sensor", {})
    if not (
        depth_sensor.get("source_id") == PRESSURE_DEPTH_SOURCE_ID
        and depth_sensor.get("derived_from_archive_array") == "pressure_pa"
        and depth_sensor.get("vio_position_z_used") is False
        and depth_sensor.get("ground_truth_position_z_used") is False
        and controller_depth.get("source_id") == PRESSURE_DEPTH_SOURCE_ID
        and controller_depth.get("replaces_vio_position_z_for_control") is True
        and controller_depth.get("ground_truth_position_z_used") is False
    ):
        raise ValueError(f"admitted episode has invalid pressure-depth provenance: {episode_path}")
    with np.load(episode_path, allow_pickle=False) as archive:
        forbidden = {
            name
            for name in archive.files
            if name.startswith("ground_truth_") or name.startswith("thruster_")
        }
        if forbidden:
            raise ValueError(
                f"admitted episode contains privileged arrays: {sorted(forbidden)}"
            )
        time_values = np.asarray(archive["time"], dtype=np.float64)
        deployable_position = np.asarray(
            archive["vio_position_ned_m"], dtype=np.float32
        ).copy()
        quaternion = _canonical_quaternion(
            archive["vio_orientation_body_to_ned_xyzw"]
        )
        raw_acceleration = np.asarray(
            archive["imu_linear_acceleration_body_frd_m_s2"], dtype=np.float32
        )
        ahrs_quaternion = _canonical_quaternion(
            archive["ahrs_orientation_body_to_ned_xyzw"]
        )
        dynamic_acceleration = gravity_compensated_body_acceleration(
            raw_acceleration,
            ahrs_quaternion,
        )
        # Pressure-sensor relative depth is the single depth feature.
        # Raw pressure remains in the archive for sensor/timestamp auditing,
        # but is intentionally not duplicated in the JEPA input.
        observation = np.column_stack(
            (
                quaternion_to_rotation6d(quaternion),
                archive["vio_linear_velocity_body_frd_m_s"],
                archive["imu_angular_velocity_body_frd_rad_s"],
                dynamic_acceleration,
                archive["depth_down_m"],
            )
        ).astype(np.float32)
        deployable_position[:, 2] = np.asarray(
            archive["depth_down_m"], dtype=np.float32
        )
        observation[:, DEPTH_INDEX] -= observation[0, DEPTH_INDEX]
        raw_motor = np.asarray(archive[ACTION_SOURCE_FIELD], dtype=np.float32)
        action = motor_output_to_modal(raw_motor)
        reconstructed_motor = modal_to_motor_output(action)
        if not np.allclose(raw_motor, reconstructed_motor, rtol=0.0, atol=2e-6):
            raise ValueError(
                f"motor output leaves the supported mixer manifold: {episode_path}"
            )
        manual_control = np.asarray(
            archive["manual_control_normalized"]
            if "manual_control_normalized" in archive.files
            else np.zeros((len(time_values), 6), dtype=np.float32),
            dtype=np.float32,
        )
        valid = np.logical_and.reduce(
            tuple(
                np.asarray(archive[name], dtype=bool)
                for name in ("vio_valid", "imu_valid", "depth_valid")
            )
        )
    sample_count = len(time_values)
    if observation.shape != (sample_count, len(OBSERVATION_LABELS)):
        raise ValueError(f"observation shape mismatch: {episode_path}")
    if deployable_position.shape != (sample_count, 3):
        raise ValueError(f"deployable position shape mismatch: {episode_path}")
    if action.shape != (sample_count, len(ACTION_LABELS)):
        raise ValueError(f"action shape mismatch: {episode_path}")
    if manual_control.shape != (sample_count, 6):
        raise ValueError(f"manual control shape mismatch: {episode_path}")
    if valid.shape != (sample_count,):
        raise ValueError(f"validity shape mismatch: {episode_path}")
    if not (
        np.all(np.isfinite(observation))
        and np.all(np.isfinite(action))
        and np.all(np.isfinite(deployable_position))
        and np.all(np.diff(time_values) > 0.0)
    ):
        raise ValueError(f"non-finite or unordered episode: {episode_path}")
    target = metadata.get("episode_timing", {}).get("training_target_window", {})
    start = float(target.get("start_seconds", np.nan))
    end = float(target.get("end_seconds", np.nan))
    if not np.isfinite(start) or not np.isfinite(end) or not 0.0 <= start < end:
        raise ValueError(f"invalid JEPA target interval: {episode_path}")
    motion_profile = (
        metadata.get("command_provenance", {}).get("recipe", {}).get(
            "motion_profile", "local_dynamics"
        )
    )
    if motion_profile not in MOTION_PROFILE_IDS:
        raise ValueError(f"unknown motion profile: {motion_profile}")
    recipe = metadata.get("command_provenance", {}).get("recipe", {})
    active_axes = tuple(str(axis).strip().lower() for axis in recipe.get("active_axes", ()))
    # Older admitted manifests predate axis provenance. Keep them loadable for
    # factual baselines, while new 4-DOF pilots always carry explicit labels.
    if not active_axes:
        active_axes = MANUAL_CONTROL_AXES
    if len(set(active_axes)) != len(active_axes) or any(
        axis not in MANUAL_CONTROL_AXES for axis in active_axes
    ):
        raise ValueError(f"invalid active_axes provenance: {episode_path}")
    return EpisodeSequence(
        job_id=entry["job_id"],
        joint_split_cell=entry["joint_split_cell"],
        partition=entry["partition"],
        time=time_values,
        observation=observation,
        action=action,
        valid=valid,
        target_start_seconds=start,
        target_end_seconds=end,
        sample_rate_hz=float(entry["sample_rate_hz"]),
        motion_profile=motion_profile,
        active_axes=active_axes,
        manual_control=manual_control,
        deployable_position_ned_m=deployable_position,
    )


def load_partition_sequences(
    manifest_path: str | Path,
    partition: str,
) -> list[EpisodeSequence]:
    if partition not in {"train", "validation", "test"}:
        raise ValueError(f"unknown dataset partition: {partition}")
    manifest = load_dataset_manifest(manifest_path)
    entries = [
        entry for entry in manifest["entries"] if entry["partition"] == partition
    ]
    if not entries:
        raise ValueError(f"manifest contains no {partition} episodes")
    return [_load_episode(entry) for entry in entries]


def common_sample_rate(episodes: Sequence[EpisodeSequence]) -> float:
    rates = np.asarray([episode.sample_rate_hz for episode in episodes])
    rate = float(np.median(rates))
    if rate <= 0.0 or np.max(np.abs(rates - rate)) > 0.2:
        raise ValueError("episodes do not share a stable sample rate")
    return rate


def _linear_resample(
    source_time: np.ndarray,
    values: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    """Linearly sample continuous estimator outputs on a uniform model grid."""

    source = np.asarray(source_time, dtype=np.float64)
    data = np.asarray(values)
    query = np.asarray(query_time, dtype=np.float64)
    if source.ndim != 1 or data.ndim != 2 or len(source) != len(data):
        raise ValueError("resampling source arrays are not aligned")
    if query.ndim != 1 or np.any(np.diff(source) <= 0.0):
        raise ValueError("resampling timestamps must be strictly increasing")
    output = np.empty((len(query), data.shape[1]), dtype=np.float64)
    for column in range(data.shape[1]):
        output[:, column] = np.interp(query, source, data[:, column])
    return output.astype(data.dtype, copy=False)


def _project_rotation6d(values: np.ndarray) -> np.ndarray:
    """Project interpolated rotation columns back onto the SO(3) 6D chart."""

    rotation = np.asarray(values, dtype=np.float64)
    if rotation.ndim != 2 or rotation.shape[1] != 6:
        raise ValueError("rotation6D projection requires [samples, 6]")
    first = rotation[:, :3]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-9)
    second = rotation[:, 3:] - np.sum(
        first * rotation[:, 3:], axis=1, keepdims=True
    ) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-9)
    return np.concatenate((first, second), axis=1).astype(values.dtype, copy=False)


def _piecewise_constant_integral(
    source_time: np.ndarray,
    values: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    """Integrate causal zero-order-held commands up to each query time."""

    source = np.asarray(source_time, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    query = np.asarray(query_time, dtype=np.float64)
    if data.ndim != 2 or len(source) != len(data) or len(source) < 2:
        raise ValueError("command integral requires aligned two-dimensional data")
    dt = np.diff(source)
    cumulative = np.concatenate(
        (
            np.zeros((1, data.shape[1]), dtype=np.float64),
            np.cumsum(data[:-1] * dt[:, None], axis=0),
        ),
        axis=0,
    )
    clipped = np.clip(query, source[0], source[-1])
    left = np.searchsorted(source, clipped, side="right") - 1
    left = np.clip(left, 0, len(source) - 1)
    return cumulative[left] + data[left] * (clipped - source[left])[:, None]


def _forward_interval_mean(
    source_time: np.ndarray,
    values: np.ndarray,
    grid_time: np.ndarray,
    step_seconds: float,
) -> np.ndarray:
    """Average commands over ``[t, t + step)`` for each model grid point."""

    start_integral = _piecewise_constant_integral(
        source_time, values, grid_time
    )
    end_integral = _piecewise_constant_integral(
        source_time, values, grid_time + step_seconds
    )
    return ((end_integral - start_integral) / step_seconds).astype(
        np.asarray(values).dtype, copy=False
    )


def resample_episode_sequence(
    episode: EpisodeSequence,
    sample_rate_hz: float,
) -> EpisodeSequence:
    """Resample one admitted episode without leaking future commands.

    Estimator observations are continuous signals and use linear interpolation.
    An action attached to model time ``t`` is the exact mean of the archived
    zero-order-held motor demand over ``[t, t + dt)``. It therefore drives the
    transition from the observation at ``t`` to the one at ``t + dt``.
    """

    rate = float(sample_rate_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("model sample rate must be positive and finite")
    if rate > episode.sample_rate_hz + 0.2:
        raise ValueError("model sample rate cannot exceed the recorder rate")
    source_time = np.asarray(episode.time, dtype=np.float64)
    if source_time.ndim != 1 or len(source_time) < 3:
        raise ValueError("episode timestamps are invalid")
    step = 1.0 / rate
    count = int(math.floor((source_time[-1] - source_time[0]) / step + 1e-9)) + 1
    grid = source_time[0] + np.arange(count, dtype=np.float64) * step
    observation = _linear_resample(source_time, episode.observation, grid)
    observation[:, ROTATION_6D_SLICE] = _project_rotation6d(
        observation[:, ROTATION_6D_SLICE]
    )
    position = _linear_resample(
        source_time, episode.deployable_position_ned_m, grid
    )
    action = _forward_interval_mean(source_time, episode.action, grid, step)
    manual = _forward_interval_mean(source_time, episode.manual_control, grid, step)

    # A model row is valid only if its observation interpolation and complete
    # forward action interval use valid archived rows. This is conservative at
    # the terminal boundary, which is excluded from JEPA windows anyway.
    valid = np.zeros(len(grid), dtype=bool)
    archived_valid = np.asarray(episode.valid, dtype=bool)
    for index, start in enumerate(grid):
        stop = start + step
        if stop > source_time[-1] + 1e-9:
            continue
        left = max(0, int(np.searchsorted(source_time, start, side="right") - 1))
        right = min(
            len(source_time) - 1,
            int(np.searchsorted(source_time, stop, side="left")),
        )
        valid[index] = bool(np.all(archived_valid[left : right + 1]))

    return EpisodeSequence(
        job_id=episode.job_id,
        joint_split_cell=episode.joint_split_cell,
        partition=episode.partition,
        time=grid,
        observation=observation,
        action=action,
        valid=valid,
        target_start_seconds=episode.target_start_seconds,
        target_end_seconds=episode.target_end_seconds,
        sample_rate_hz=rate,
        motion_profile=episode.motion_profile,
        active_axes=episode.active_axes,
        manual_control=manual,
        deployable_position_ned_m=position,
    )


def resample_episode_sequences(
    episodes: Sequence[EpisodeSequence],
    sample_rate_hz: float,
) -> list[EpisodeSequence]:
    return [
        resample_episode_sequence(episode, sample_rate_hz) for episode in episodes
    ]


class MotorJepaWindowDataset(Dataset):
    """Map global indices to causal windows without crossing episodes."""

    def __init__(
        self,
        episodes: Sequence[EpisodeSequence],
        standardizer: Standardizer,
        *,
        history_steps: int,
        horizon_steps: int,
        model_step_stride: int = 5,
        stride: int = 1,
    ) -> None:
        if (
            history_steps < 2
            or horizon_steps < 1
            or model_step_stride < 1
            or stride < 1
        ):
            raise ValueError("invalid JEPA window dimensions")
        self.episodes = tuple(episodes)
        self.standardizer = standardizer
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        self.model_step_stride = int(model_step_stride)
        self.stride = int(stride)
        starts = []
        cumulative = []
        total = 0
        for episode in self.episodes:
            # Each model step spans several recorder ticks. A history token
            # pairs an endpoint observation with the mean action over the
            # immediately preceding interval. Future actions similarly cover
            # the interval leading to each future observation endpoint.
            minimum = self.history_steps * self.model_step_stride
            maximum = len(episode.time) - self.horizon_steps * self.model_step_stride - 1
            candidates = np.arange(minimum, maximum + 1, self.stride, dtype=np.int32)
            if len(candidates):
                first_future = episode.time[candidates + self.model_step_stride]
                last_future = episode.time[
                    candidates + self.horizon_steps * self.model_step_stride
                ]
                interval_valid = np.logical_and(
                    first_future >= episode.target_start_seconds,
                    last_future <= episode.target_end_seconds,
                )
                candidates = candidates[interval_valid]
            if len(candidates):
                keep = np.asarray(
                    [
                        np.all(
                            episode.valid[
                                current
                                - self.history_steps * self.model_step_stride : current
                                + self.horizon_steps * self.model_step_stride
                                + 1
                            ]
                        )
                        for current in candidates
                    ],
                    dtype=bool,
                )
                candidates = candidates[keep]
            starts.append(candidates)
            total += len(candidates)
            cumulative.append(total)
        if total == 0:
            raise ValueError("no valid JEPA windows in selected episodes")
        self._starts = tuple(starts)
        self._cumulative = tuple(cumulative)

    def __len__(self) -> int:
        return self._cumulative[-1]

    def action_sampling_weights(self, *, focused_fraction: float = 0.5) -> np.ndarray:
        """Blend uniform sampling with identifiable factual windows.

        Future action diversity trains counterfactual rollout conditioning,
        while past excitation and observed motion response make the fixed
        history belief identifiable. This changes sampling frequency only; it
        never changes factual actions, targets, or episode provenance.
        """
        if not 0.0 <= focused_fraction <= 1.0:
            raise ValueError("focused sampling fraction must be in [0, 1]")
        if focused_fraction == 0.0:
            return np.ones(len(self), dtype=np.float64)
        scores: list[np.ndarray] = []
        step = self.model_step_stride
        future_offsets = np.arange(self.horizon_steps, dtype=np.int64) * step
        for episode, currents in zip(self.episodes, self._starts, strict=True):
            if len(currents) == 0:
                continue
            action = episode.action.astype(np.float64)
            cumulative = np.concatenate(
                (np.zeros((1, action.shape[1]), dtype=np.float64), action.cumsum(axis=0)),
                axis=0,
            )

            def interval_mean(starts: np.ndarray) -> np.ndarray:
                return (cumulative[starts + step] - cumulative[starts]) / step

            previous = interval_mean(currents - step)[:, None, :]
            future_starts = currents[:, None] + future_offsets[None, :]
            future = interval_mean(future_starts)
            previous = (previous - self.standardizer.action_mean[None, None, :]) / self.standardizer.action_std[None, None, :]
            future = (future - self.standardizer.action_mean[None, None, :]) / self.standardizer.action_std[None, None, :]
            sequence = np.concatenate((previous, future), axis=1)
            future_innovation = np.sqrt(np.mean(np.diff(sequence, axis=1) ** 2, axis=(1, 2)))
            future_magnitude = np.sqrt(np.mean(future**2, axis=(1, 2)))
            yaw_magnitude = np.sqrt(np.mean(future[..., 3] ** 2, axis=1))

            history_offsets = np.arange(self.history_steps, 0, -1, dtype=np.int64) * step
            history_starts = currents[:, None] - history_offsets[None, :]
            history_action = interval_mean(history_starts)
            history_action = (
                history_action - self.standardizer.action_mean[None, None, :]
            ) / self.standardizer.action_std[None, None, :]
            observation_offsets = np.arange(
                self.history_steps - 1, -1, -1, dtype=np.int64
            ) * step
            history_indices = currents[:, None] - observation_offsets[None, :]
            motion_indices = (6, 7, 8, 9, 10, 11, DEPTH_INDEX)
            history_motion = episode.observation[history_indices][..., motion_indices]
            motion_mean = self.standardizer.observation_mean[list(motion_indices)]
            motion_std = self.standardizer.observation_std[list(motion_indices)]
            history_motion = (history_motion - motion_mean) / motion_std
            history_innovation = np.sqrt(
                np.mean(np.diff(history_action, axis=1) ** 2, axis=(1, 2))
            )
            history_magnitude = np.sqrt(np.mean(history_action**2, axis=(1, 2)))
            history_response = np.sqrt(
                np.mean(np.diff(history_motion, axis=1) ** 2, axis=(1, 2))
            )
            future_score = (
                future_innovation + 0.35 * future_magnitude + 0.65 * yaw_magnitude
            )
            history_score = history_innovation + 0.35 * history_magnitude + history_response
            scores.append(0.5 * future_score + 0.5 * history_score)
        focused = np.concatenate(scores)
        if focused.shape != (len(self),) or not np.all(np.isfinite(focused)):
            raise RuntimeError("failed to build finite action sampling weights")
        positive = focused[focused > 0.0]
        if len(positive) == 0:
            return np.ones(len(self), dtype=np.float64)
        focused = np.clip(focused, 0.0, np.quantile(positive, 0.95))
        focused /= max(float(focused.mean()), 1e-12)
        return ((1.0 - focused_fraction) + focused_fraction * focused).astype(np.float64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect_right(self._cumulative, index)
        previous = 0 if episode_index == 0 else self._cumulative[episode_index - 1]
        local_index = index - previous
        current = int(self._starts[episode_index][local_index])
        episode = self.episodes[episode_index]
        step = self.model_step_stride

        def history_window(
            source_episode: EpisodeSequence,
            source_current: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            indices = source_current - np.arange(
                self.history_steps - 1, -1, -1, dtype=np.int64
            ) * step
            observation = source_episode.observation[indices].copy()
            action = np.stack(
                [
                    source_episode.action[start : start + step].mean(axis=0)
                    for start in indices - step
                ],
                axis=0,
            )
            observation[:, [DEPTH_INDEX]] -= source_episode.observation[
                source_current, [DEPTH_INDEX]
            ]
            observation = (
                observation - self.standardizer.observation_mean
            ) / self.standardizer.observation_std
            action = (
                action - self.standardizer.action_mean
            ) / self.standardizer.action_std
            return observation, action

        observation_history, action_history = history_window(episode, current)
        def interval_means(starts: np.ndarray) -> np.ndarray:
            return np.stack(
                [episode.action[start : start + step].mean(axis=0) for start in starts],
                axis=0,
            )

        future_offsets = np.arange(1, self.horizon_steps + 1, dtype=np.int64) * step
        future_observation = episode.observation[current + future_offsets].copy()
        future_action = interval_means(current + future_offsets - step)
        future_position_delta_ned_m = (
            episode.deployable_position_ned_m[current + future_offsets]
            - episode.deployable_position_ned_m[current]
        )

        anchor = episode.observation[current, [DEPTH_INDEX]]
        future_observation[:, [DEPTH_INDEX]] -= anchor
        future_observation = (
            future_observation - self.standardizer.observation_mean
        ) / self.standardizer.observation_std
        future_action = (
            future_action - self.standardizer.action_mean
        ) / self.standardizer.action_std
        return {
            "observation_history": torch.from_numpy(
                np.ascontiguousarray(observation_history, dtype=np.float32)
            ),
            "action_history": torch.from_numpy(
                np.ascontiguousarray(action_history, dtype=np.float32)
            ),
            "future_action": torch.from_numpy(
                np.ascontiguousarray(future_action, dtype=np.float32)
            ),
            "future_observation": torch.from_numpy(
                np.ascontiguousarray(future_observation, dtype=np.float32)
            ),
            "future_position_delta_ned_m": torch.from_numpy(
                np.ascontiguousarray(future_position_delta_ned_m, dtype=np.float32)
            ),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "current_index": torch.tensor(current, dtype=torch.int64),
        }
