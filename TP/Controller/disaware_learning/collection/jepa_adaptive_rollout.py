"""Closed-loop JEPA factual-dynamics adaptive-control rollout."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from sensor_msgs.msg import Imu
import rclpy
import torch

from disaware_learning.collection.manual_rollout import ManualRollout, quaternion_rpy
from disaware_learning.control.contract import intent_to_manual_control
from disaware_learning.control.body_twist_pid import (
    summarize_state_estimation_audit,
    summarize_trajectory_tracking,
)
from disaware_learning.control.jepa_adaptive_control import (
    JepaAdaptiveConfig,
    JepaFactualDynamicsAdaptiveController,
    load_jepa_adaptive_config,
)
from disaware_learning.control.rc import clamp_manual_control
from disaware_learning.data.episode_validation import inspect_episode
from disaware_learning.data.episode_io import load_metadata, write_metadata
from disaware_learning.learning.motor_jepa_data import (
    gravity_compensated_body_acceleration,
    quaternion_to_rotation6d,
)


class JepaAdaptiveRollout(ManualRollout):
    """Run one reference/domain pair with factual JEPA compensation."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("stage2_checkpoint", "")
        self.declare_parameter("jepa_adaptive_config", "")
        self.declare_parameter("jepa_adaptive_device", "auto")
        self.declare_parameter("adaptive_trace_path", "")
        checkpoint = str(self.get_parameter("stage2_checkpoint").value).strip()
        if not checkpoint:
            raise ValueError("stage2_checkpoint is required")
        config_path = str(self.get_parameter("jepa_adaptive_config").value).strip()
        config = load_jepa_adaptive_config(config_path) if config_path else JepaAdaptiveConfig()
        limit = float(self.action_limit)
        config_values = asdict(config)
        config_values["command_limit"] = (limit, limit, limit, limit)
        config = JepaAdaptiveConfig(**config_values)
        requested_device = str(self.get_parameter("jepa_adaptive_device").value).strip().lower()
        device = (
            "cuda"
            if requested_device in ("", "auto") and torch.cuda.is_available()
            else ("cpu" if requested_device in ("", "auto") else requested_device)
        )
        self.jepa_adaptive = JepaFactualDynamicsAdaptiveController.from_checkpoint(
            checkpoint, config=config, device=device
        )
        self.model_stride = max(
            1,
            int(round(self.command_rate / self.jepa_adaptive.sample_rate_hz)),
        )
        self.effective_model_rate = self.command_rate / self.model_stride
        if abs(self.effective_model_rate - self.jepa_adaptive.sample_rate_hz) > 0.15:
            raise ValueError(
                "collector rate is not compatible with JEPA model rate: "
                f"collector={self.command_rate}, model={self.jepa_adaptive.sample_rate_hz}"
            )
        self.stage2_checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.jepa_device = device
        self.jepa_config = config
        self.imu_rate: np.ndarray | None = None
        self.imu_accel: np.ndarray | None = None
        self.imu_received_at = -math.inf
        self.create_subscription(
            Imu, f"/{self.vehicle_name}/ardusub/imu", self._adaptive_imu_callback, 20
        )
        self._observation_history: deque[np.ndarray] = deque(
            maxlen=self.jepa_adaptive.history_steps
        )
        self._action_history: deque[np.ndarray] = deque(
            maxlen=self.jepa_adaptive.history_steps
        )
        self._depth_anchor: float | None = None
        self._pending_motor_samples: list[np.ndarray] = []
        self._cached_predicted_acceleration: np.ndarray | None = None
        self.adaptive_trace_path = str(
            self.get_parameter("adaptive_trace_path").value
        ).strip()
        self.adaptive_trace: dict[str, list[Any]] = {}

    def _adaptive_imu_callback(self, message: Imu) -> None:
        rate = np.asarray(
            [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z],
            dtype=np.float64,
        )
        accel = np.asarray(
            [message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z],
            dtype=np.float64,
        )
        if np.all(np.isfinite(rate)) and np.all(np.isfinite(accel)):
            self.imu_rate = rate
            self.imu_accel = accel
            self.imu_received_at = time.monotonic()

    def _wait_for_observation(self) -> None:
        super()._wait_for_observation()
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and self.imu_accel is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.imu_accel is None or self.imu_rate is None:
            raise RuntimeError("ArduSub IMU observation is unavailable")

    def _current_observation(self) -> np.ndarray:
        state = self._last_vio_state
        if state is None or self.imu_rate is None or self.imu_accel is None:
            raise RuntimeError("causal VIO/IMU state is not initialized")
        if time.monotonic() - self.imu_received_at > 0.25:
            raise RuntimeError("ArduSub IMU observation became stale")
        if self.latest_depth is None or self._depth_anchor is None:
            raise RuntimeError("ArduSub depth observation is not initialized")
        rotation6d = quaternion_to_rotation6d(
            state.orientation_body_to_ned_xyzw.reshape(1, 4)
        )[0]
        attitude_quaternion = (
            self.latest_quaternion
            if self.latest_quaternion is not None
            else state.orientation_body_to_ned_xyzw
        )
        dynamic_acceleration = gravity_compensated_body_acceleration(
            self.imu_accel.reshape(1, 3),
            np.asarray(attitude_quaternion, dtype=np.float64).reshape(1, 4),
        )[0]
        observation = np.concatenate(
            (
                rotation6d,
                state.linear_velocity_body_frd_m_s,
                self.imu_rate,
                dynamic_acceleration,
                [float(self.latest_depth - self._depth_anchor)],
            )
        ).astype(np.float32)
        if observation.shape != (16,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("constructed JEPA observation is invalid")
        return observation

    def _initialize_adaptive(self) -> None:
        current = self._current_observation()
        motors = (
            np.asarray(self.applied_motors, dtype=np.float32)
            if self.applied_motors is not None
            else np.zeros(6, dtype=np.float32)
        )
        if motors.shape != (6,) or not np.all(np.isfinite(motors)):
            motors = np.zeros(6, dtype=np.float32)
        self._observation_history.clear()
        self._action_history.clear()
        for _ in range(self.jepa_adaptive.history_steps):
            self._observation_history.append(current.copy())
            self._action_history.append(motors.copy())
        self.jepa_adaptive.reset()
        self._pending_motor_samples.clear()
        self._cached_predicted_acceleration = None
        self.jepa_adaptive.warmup(
            np.stack(self._observation_history), np.stack(self._action_history)
        )
        self.adaptive_trace = {
            "time_seconds": [],
            "external_current_time_seconds": [],
            "reference_body_twist": [],
            "reference_position_ned_m": [],
            "reference_yaw_rad": [],
            "actual_position_ned_m": [],
            "actual_yaw_rad": [],
            "estimated_position_ned_m": [],
            "estimated_yaw_rad": [],
            "external_current_velocity_ned_m_s": [],
            "estimated_body_twist": [],
            "previous_intent": [],
            "selected_intent": [],
            "nominal_intent": [],
            "model_compensation_intent": [],
            "predicted_factual_acceleration": [],
            "filtered_model_residual_acceleration": [],
            "tracking_error": [],
            "adaptive_acceleration": [],
            "used_model": [],
            "model_prediction_updated": [],
            "timing_seconds": [],
            "reason": [],
        }

    def _save_adaptive_trace(self) -> None:
        if not self.adaptive_trace.get("time_seconds"):
            return
        target = self.adaptive_trace_path
        if not target and self.report_path:
            target = str(
                Path(self.report_path).expanduser().resolve().with_suffix(
                    ".adaptive_trace.npz"
                )
            )
        if not target:
            return
        path = Path(target).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{
                name: np.asarray(values)
                for name, values in self.adaptive_trace.items()
                if name != "reason"
            },
            reason=np.asarray(self.adaptive_trace["reason"], dtype=str),
        )
        self.adaptive_trace_path = str(path)

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        super()._annotate_rollout_provenance(episode_path)
        metadata = load_metadata(episode_path)
        controller = metadata.setdefault("controller_provenance", {})
        controller.update(
            {
                "controller_id": "jepa_factual_dynamics_adaptive_control_v1",
                "state_source": self.vio_source_id,
                "stage2_checkpoint": self.stage2_checkpoint,
                "jepa_adaptive_config": self.jepa_config.as_dict(),
                "jepa_adaptive_device": self.jepa_device,
                "adaptive_trace_path": self.adaptive_trace_path or None,
                "model_input_boundary": {
                    "observation": "causal VINS stereo VIO + ArduSub IMU + ArduSub depth",
                    "action_history": "past measured six-motor normalized outputs",
                    "future_action_query": "held latest measured six-motor output only",
                    "counterfactual_candidate_search": False,
                    "ground_truth_feedback": False,
                },
            }
        )
        write_metadata(episode_path, metadata)

    def _reference(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(self.command_arrays["body_twist_reference"], dtype=np.float64)
        next_index = min(index + 1, len(values) - 1)
        return values[index], values[next_index]

    def run(self) -> dict[str, Any]:
        self._connect_ardusub()
        recorder_started = False
        episode_path = ""
        started = time.monotonic()
        failure: str | None = None
        motor_samples: list[np.ndarray] = []
        try:
            self._activate_physics_domain()
            if self.simulation_reset:
                self._prepare_vehicle()
            else:
                self._arm()
                self._run_for(np.zeros(6), 1.0)
            self._wait_for_observation()
            self._wait_for_reset_convergence()
            if self.pre_record_settle_seconds:
                self._run_for(np.zeros(6), self.pre_record_settle_seconds)
            self._initialize_pid_anchor()
            self._depth_anchor = float(self.latest_depth)
            self._initialize_adaptive()
            if self.manage_recorder:
                self._start_recorder()
                recorder_started = True
            self._begin_external_current_episode()
            for index in range(len(self.command_arrays["body_twist_reference"])):
                tick_started = time.monotonic()
                elapsed_seconds = self._formal_episode_time_seconds(index)
                current_velocity = self._publish_external_current(elapsed_seconds)
                state = self._deployable_control_state()
                current_obs = self._current_observation()
                latest_motor = (
                    np.asarray(self.applied_motors, dtype=np.float32)
                    if self.applied_motors is not None
                    else self._action_history[-1].copy()
                )
                # Match MotorJepaWindowDataset exactly: the action token at a
                # current observation is the mean of the preceding macro
                # interval, not the motor sample at the current endpoint.
                model_update = index == 0 or index % self.model_stride == 0
                if model_update and index > 0:
                    if len(self._pending_motor_samples) != self.model_stride:
                        raise RuntimeError(
                            "factual action interval does not contain one full model step"
                        )
                    self._observation_history.append(current_obs.copy())
                    self._action_history.append(
                        np.mean(
                            np.stack(self._pending_motor_samples), axis=0
                        ).astype(np.float32)
                    )
                    self._pending_motor_samples.clear()
                self._pending_motor_samples.append(latest_motor.copy())
                reference, next_reference = self._reference(index)
                trajectory_reference = self._trajectory_reference(index)
                previous_intent = self.current_command[[4, 5, 2, 3]].astype(np.float32)
                roll, _pitch, _yaw = quaternion_rpy(
                    state.orientation_body_to_ned_xyzw
                )
                roll_safety_command = (
                    -self.pid_controller.config.roll_kp * roll
                    - self.pid_controller.config.roll_rate_kd
                    * state.angular_velocity_body_frd_rad_s[0]
                )
                result = self.jepa_adaptive.step(
                    np.stack(self._observation_history),
                    np.stack(self._action_history),
                    reference,
                    next_reference,
                    previous_intent,
                    self.period,
                    current_observation=current_obs,
                    predicted_factual_acceleration=(
                        None if model_update else self._cached_predicted_acceleration
                    ),
                    auxiliary_roll_command=roll_safety_command,
                )
                if model_update:
                    self._cached_predicted_acceleration = diagnostics_acceleration = (
                        result.diagnostics["predicted_factual_acceleration"].copy()
                    )
                selected_manual = intent_to_manual_control(result.command)
                projected_roll_safety = float(
                    result.diagnostics["projected_auxiliary_roll_command"]
                )
                selected_manual[1] = (
                    projected_roll_safety
                )
                selected_manual = clamp_manual_control(selected_manual, self.action_limit)
                # JepaAdaptiveController already applies the per-axis command
                # slew.  A second all-axis slew here would add an unmodelled
                # delay. Keep only the independent roll safety limiter.
                selected_manual[1] = self.current_command[1] + np.clip(
                    selected_manual[1] - self.current_command[1],
                    -self.action_slew_rate * self.period,
                    self.action_slew_rate * self.period,
                )
                self.current_command = selected_manual
                self._send_command(self.current_command)
                diagnostics = result.diagnostics
                for name, value in {
                    "time_seconds": time.monotonic() - started,
                    "external_current_time_seconds": elapsed_seconds,
                    "reference_body_twist": reference.copy(),
                    "reference_position_ned_m": trajectory_reference.position_preview_ned_m.copy(),
                    "reference_yaw_rad": trajectory_reference.yaw_preview_rad,
                    "actual_position_ned_m": _truth.position_ned_m.copy(),
                    "actual_yaw_rad": quaternion_rpy(
                        _truth.orientation_body_to_ned_xyzw
                    )[2],
                    "estimated_position_ned_m": state.position_ned_m.copy(),
                    "estimated_yaw_rad": quaternion_rpy(
                        state.orientation_body_to_ned_xyzw
                    )[2],
                    "external_current_velocity_ned_m_s": current_velocity.copy(),
                    "estimated_body_twist": np.concatenate(
                        (state.linear_velocity_body_frd_m_s, [state.angular_velocity_body_frd_rad_s[2]])
                    ),
                    "previous_intent": previous_intent.copy(),
                    "selected_intent": result.command.copy(),
                    "nominal_intent": result.nominal_command.copy(),
                    "model_compensation_intent": result.model_compensation_command.copy(),
                    "predicted_factual_acceleration": diagnostics["predicted_factual_acceleration"].copy(),
                    "filtered_model_residual_acceleration": diagnostics[
                        "filtered_model_residual_acceleration"
                    ].copy(),
                    "tracking_error": diagnostics["tracking_error"].copy(),
                    "adaptive_acceleration": diagnostics["adaptive_acceleration"].copy(),
                    "used_model": result.used_model,
                    "model_prediction_updated": diagnostics["model_prediction_updated"],
                    "timing_seconds": diagnostics["timing_seconds"],
                    "reason": result.reason,
                }.items():
                    self.adaptive_trace[name].append(value)
                rclpy.spin_once(self, timeout_sec=min(0.01, self.period))
                for _ in range(12):
                    rclpy.spin_once(self, timeout_sec=0.0)
                self._verify_execution_heartbeat()
                self._safety_gate()
                if self.applied_motors is not None:
                    motor_samples.append(self.applied_motors.copy())
                remaining = self.period - (time.monotonic() - tick_started)
                if remaining > 0.0:
                    time.sleep(remaining)
        except Exception as error:
            failure = str(error)
            self.get_logger().error(f"JEPA adaptive rollout failed: {failure}")
        finally:
            self._end_external_current_episode()
            try:
                self._save_adaptive_trace()
            except OSError as error:
                failure = failure or f"could not save adaptive trace: {error}"
            if recorder_started:
                try:
                    episode_path = self._stop_recorder()
                    self._annotate_rollout_provenance(episode_path)
                except RuntimeError as error:
                    failure = failure or str(error)
            self._neutral_and_disarm(release=True)

        motors = np.stack(motor_samples) if motor_samples else np.empty((0, 6))
        saturation_fraction = (
            float(np.mean(np.any(np.abs(motors) >= 0.999, axis=1)))
            if motors.size
            else 1.0
        )
        episode_report = (
            inspect_episode(episode_path)
            if episode_path
            else {"pass": False, "error": "episode was not saved"}
        )
        timing = np.asarray(self.adaptive_trace.get("timing_seconds", [math.inf]))
        commands = np.asarray(self.adaptive_trace.get("selected_intent", []))
        response_metrics: dict[str, Any] = {}
        trajectory_metrics: dict[str, Any] = {}
        state_estimation_metrics: dict[str, Any] = {}
        if episode_path and len(commands):
            with np.load(episode_path, allow_pickle=False) as archive:
                actual = np.column_stack(
                    (
                        archive["vio_linear_velocity_body_frd_m_s"],
                        archive["vio_angular_velocity_body_frd_rad_s"][:, 2],
                    )
                )
                elapsed = np.asarray(archive["time"], dtype=np.float64)
            trace_time = np.asarray(self.adaptive_trace["time_seconds"], dtype=np.float64)
            trace_time -= trace_time[0]
            reference = np.asarray(self.adaptive_trace["reference_body_twist"])
            aligned_reference = np.column_stack(
                [np.interp(elapsed, trace_time, reference[:, axis]) for axis in range(4)]
            )
            response_metrics = {
                "reference_rms": np.sqrt(np.mean(aligned_reference**2, axis=0)).tolist(),
                "vio_response_rms": np.sqrt(np.mean(actual**2, axis=0)).tolist(),
                "tracking_rmse": np.sqrt(np.mean((actual - aligned_reference) ** 2, axis=0)).tolist(),
                "command_rms": np.sqrt(np.mean(commands**2, axis=0)).tolist(),
                "command_rate_p95": np.percentile(
                    np.abs(np.diff(commands, axis=0) / self.period), 95, axis=0
                ).tolist(),
            }
            reference_position = np.asarray(
                self.adaptive_trace["reference_position_ned_m"], dtype=np.float64
            )
            reference_yaw = np.asarray(
                self.adaptive_trace["reference_yaw_rad"], dtype=np.float64
            )
            actual_position = np.asarray(
                self.adaptive_trace["actual_position_ned_m"], dtype=np.float64
            )
            estimated_position = np.asarray(
                self.adaptive_trace["estimated_position_ned_m"], dtype=np.float64
            )
            actual_yaw = np.asarray(
                self.adaptive_trace["actual_yaw_rad"], dtype=np.float64
            )
            estimated_yaw = np.asarray(
                self.adaptive_trace["estimated_yaw_rad"], dtype=np.float64
            )
            trajectory_metrics = {
                "ground_truth": summarize_trajectory_tracking(
                    reference_position,
                    actual_position,
                    reference_yaw,
                    actual_yaw,
                ),
            }
            state_estimation_metrics = summarize_state_estimation_audit(
                actual_position,
                estimated_position,
                actual_yaw,
                estimated_yaw,
            )
        report = {
            "protocol_id": "jepa_factual_dynamics_adaptive_control_closed_loop_v1",
            "execution_pass": bool(
                failure is None
                and motors.size
                and saturation_fraction <= self.envelope["motor_saturation_ratio"]
                and episode_report.get("pass") is True
            ),
            "failure": failure,
            "episode_path": episode_path,
            "episode_contract": episode_report,
            "stage2_checkpoint": self.stage2_checkpoint,
            "jepa_adaptive_device": self.jepa_device,
            "controller_provenance": {
                "controller_id": "jepa_factual_dynamics_adaptive_control_v1",
                # ``used_model`` means a valid factual estimate was used on
                # this control tick, including cached estimates between JEPA
                # updates.  Keep the actual inference cadence separate so
                # reports do not confuse 25 Hz control with 5 Hz inference.
                "model_used_fraction": float(
                    np.mean(np.asarray(self.adaptive_trace.get("used_model", [False]), dtype=bool))
                ),
                "model_prediction_update_fraction": float(
                    np.mean(
                        np.asarray(
                            self.adaptive_trace.get(
                                "model_prediction_updated", [False]
                            ),
                            dtype=bool,
                        )
                    )
                ),
                "model_prediction_update_rate_hz": float(
                    np.mean(
                        np.asarray(
                            self.adaptive_trace.get(
                                "model_prediction_updated", [False]
                            ),
                            dtype=bool,
                        )
                    )
                    * self.command_rate
                ),
                "latency_p50_seconds": float(np.percentile(timing, 50)),
                "latency_p95_seconds": float(np.percentile(timing, 95)),
                "latency_max_seconds": float(np.max(timing)),
                "adaptive_trace_path": self.adaptive_trace_path or None,
            },
            "motor_saturation_fraction": saturation_fraction,
            "observed_response_metrics": response_metrics,
            "trajectory_metrics": trajectory_metrics,
            "state_estimation_metrics": state_estimation_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        if self.report_path:
            target = Path(self.report_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = JepaAdaptiveRollout()
    try:
        report = node.run()
        node.get_logger().info(
            f"JEPA ADAPTIVE ROLLOUT COMPLETE: execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
