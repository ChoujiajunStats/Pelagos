"""Direct-action JEPA-MPPI rollout with a causal VINS state boundary.

The four trajectory-control axes are selected entirely in the learned model
space. A fixed roll PD remains outside MPPI as an attitude safety channel.
Only deployable observations and past measured motor outputs enter the model.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from sensor_msgs.msg import Imu
import rclpy
import torch

from disaware_learning.collection.manual_rollout import ManualRollout
from disaware_learning.control.body_twist_pid import (
    BodyTwistReference,
    BodyTwistVelocityRatePid,
    quaternion_rpy,
    summarize_state_estimation_audit,
    summarize_trajectory_tracking,
    zero_twist_settling_pid_config,
)
from disaware_learning.control.contract import intent_to_manual_control
from disaware_learning.control.jepa_mppi import (
    IMPLEMENTATION_ID as JEPA_MPPI_IMPLEMENTATION_ID,
    JepaMppiConfig,
    ReferenceConditionedJepaMppi,
    ReferenceTrajectory,
    load_jepa_mppi_config,
)
from disaware_learning.control.rc import clamp_manual_control, slew_towards
from disaware_learning.data.episode_io import load_metadata, write_metadata
from disaware_learning.data.episode_validation import inspect_episode
from disaware_learning.learning.motor_jepa_data import (
    gravity_compensated_body_acceleration,
    quaternion_to_rotation6d,
)
from disaware_learning.estimation.vio_state import VioState


class JepaMppiRollout(ManualRollout):
    """Run one reference/domain pair through the absolute-action MPPI loop."""

    # Subclasses can reuse the Stonefish, recorder, and causal-state boundary
    # with a different JEPA optimizer.  The parameter names stay stable so an
    # existing launch file can swap the executable without changing the data
    # contract.
    CONTROLLER_ID = JEPA_MPPI_IMPLEMENTATION_ID
    CONTROLLER_TYPE = "direct_action_jepa_mppi"
    PROTOCOL_ID = "direct_action_jepa_mppi_closed_loop_v22_applied_async_boundary"
    ASYNC_OPTIMIZATION = True
    # Fill the deployable history with controlled-settling transitions and
    # their actual motor outputs before reference time starts.
    PRIME_CAUSAL_HISTORY = True
    CONFIG_PARAMETER = "jepa_mppi_config"
    DEVICE_PARAMETER = "jepa_mppi_device"
    TRACE_PARAMETER = "mppi_trace_path"
    CONFIG_METADATA_KEY = "jepa_mppi_config"
    DEVICE_METADATA_KEY = "jepa_mppi_device"
    TRACE_METADATA_KEY = "mppi_trace_path"
    TRACE_SUFFIX = ".mppi_trace.npz"

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("stage2_checkpoint", "")
        self.declare_parameter(self.CONFIG_PARAMETER, "")
        self.declare_parameter(self.DEVICE_PARAMETER, "auto")
        self.declare_parameter(self.TRACE_PARAMETER, "")
        checkpoint = str(self.get_parameter("stage2_checkpoint").value).strip()
        if not checkpoint:
            raise ValueError("stage2_checkpoint is required")
        config_path = str(self.get_parameter(self.CONFIG_PARAMETER).value).strip()
        config = self._load_controller_config(config_path)
        # ArduSub's MANUAL boundary is currently +/- action_limit on all six
        # channels. Keep the controller's four-axis limits consistent with it.
        limit = float(self.action_limit)
        config = replace(
            config,
            manual_limit=tuple(
                min(float(value), limit) for value in config.manual_limit
            ),
        )
        requested_device = (
            str(self.get_parameter(self.DEVICE_PARAMETER).value).strip().lower()
        )
        device = (
            "cuda"
            if requested_device in ("", "auto") and torch.cuda.is_available()
            else ("cpu" if requested_device in ("", "auto") else requested_device)
        )
        self.jepa_mppi = self._build_controller(
            checkpoint,
            config,
            device,
        )
        controller_rate = self.jepa_mppi.sample_rate_hz
        recorder_to_model = self.command_rate / controller_rate
        if recorder_to_model < 1.0 or not math.isclose(
            recorder_to_model, round(recorder_to_model), rel_tol=1e-5, abs_tol=1e-3
        ):
            raise ValueError(
                "JEPA controller rate must divide collector command rate: "
                f"{controller_rate} vs {self.command_rate}"
            )
        self.model_stride = int(round(recorder_to_model))
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        self.stage2_checkpoint = str(checkpoint_path)
        resolved_config_path = (
            Path(config_path).expanduser().resolve() if config_path else None
        )
        self.jepa_config_path = (
            str(resolved_config_path) if resolved_config_path is not None else ""
        )
        self.jepa_device = device
        self.jepa_config = config
        self.imu_rate: np.ndarray | None = None
        self.imu_accel: np.ndarray | None = None
        self.imu_received_at = -math.inf
        self.create_subscription(
            Imu,
            f"/{self.vehicle_name}/ardusub/imu",
            self._mppi_imu_callback,
            20,
        )
        history_steps = self.jepa_mppi.history_steps
        self._observation_history: deque[np.ndarray] = deque(maxlen=history_steps)
        self._action_history: deque[np.ndarray] = deque(maxlen=history_steps)
        self._pending_action_samples: list[np.ndarray] = []
        # The dataset pairs an observation at model tick t with the action
        # interval immediately preceding it.  Seed the first interval with
        # the actuator sample already applied before the preroll starts, then
        # advance this counter independently of planner completion latency.
        self._causal_history_tick_count = 0
        self._depth_anchor: float | None = None
        self.mppi_trace_path = str(
            self.get_parameter(self.TRACE_PARAMETER).value
        ).strip()
        self.mppi_trace: dict[str, list[Any]] = {}
        self._held_mppi_command: np.ndarray | None = None
        self._last_mppi_update_index = -(10**9)
        self._optimization_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="jepa-planner")
            if self.ASYNC_OPTIMIZATION
            else None
        )
        self._pending_optimization: Future | None = None
        self._pending_optimization_submitted_at: float | None = None
        self._pending_optimization_state_received_at: float | None = None
        self._pending_optimization_reference_index: int | None = None
        # Wall-clock time of the most recent planner submission.  This is
        # separate from measured-token completion: a CUDA solve can finish
        # before the next admissible 10 Hz planning instant.
        self._last_planner_submission_at: float | None = None
        self._history_preroll_seconds = 0.0
        self._history_preroll_model_samples = 0
        self._settling_pid = BodyTwistVelocityRatePid(
            zero_twist_settling_pid_config()
        )
        self._causal_preroll_audit: dict[str, Any] = {
            "policy": "fixed_duration_zero_twist_pressure_depth_hold_manual",
            "feedback_controller_used": True,
            "feedback_state": (
                "VINS position/orientation/body velocity + ArduSub pressure depth/IMU gyro"
            ),
            "ground_truth_feedback": False,
            "settling_controller_config": self._settling_pid.config.as_dict(),
        }
        self._initial_plan_result: Any | None = None
        self._initial_plan_solve_seconds = math.nan
        self._initial_plan_state_age_seconds = math.nan

    def _build_controller(self, checkpoint: str, config: Any, device: str):
        return ReferenceConditionedJepaMppi.from_checkpoint(
            checkpoint,
            config=config,
            device=device,
        )

    def _load_controller_config(self, config_path: str):
        return load_jepa_mppi_config(config_path) if config_path else JepaMppiConfig()

    def _mppi_imu_callback(self, message: Imu) -> None:
        rate = np.asarray(
            [
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        accel = np.asarray(
            [
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ],
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
            refresh_deadline = time.monotonic() + 0.20
            while (
                rclpy.ok()
                and time.monotonic() - self.imu_received_at > 0.25
                and time.monotonic() < refresh_deadline
            ):
                rclpy.spin_once(self, timeout_sec=0.01)
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

    def _roll_safety_command(self) -> float:
        state = self._last_vio_state
        if state is None or self.imu_rate is None:
            raise RuntimeError("roll safety state is not initialized")
        roll = float(quaternion_rpy(state.orientation_body_to_ned_xyzw)[0])
        config = self.pid_controller.config
        value = -config.roll_kp * roll - config.roll_rate_kd * float(self.imu_rate[0])
        return float(np.clip(value, -self.action_limit, self.action_limit))

    def _reference_horizon(self, index: int) -> ReferenceTrajectory:
        horizon = self.jepa_mppi.model_horizon_steps
        indices = np.minimum(
            index + np.arange(1, horizon + 1) * self.model_stride,
            len(self.command_arrays["body_twist_reference"]) - 1,
        )
        references = [self._trajectory_reference(int(item)) for item in indices]
        times = (
            np.arange(1, horizon + 1, dtype=np.float64) / self.jepa_mppi.sample_rate_hz
        )
        return ReferenceTrajectory(
            time_seconds=times,
            position_ned_m=np.stack(
                [reference.position_preview_ned_m for reference in references]
            ).astype(np.float32),
            linear_velocity_body_m_s=np.stack(
                [reference.linear_velocity_body_frd_m_s for reference in references]
            ).astype(np.float32),
            yaw_rad=np.asarray(
                [reference.yaw_preview_rad for reference in references],
                dtype=np.float32,
            ),
            yaw_rate_rad_s=np.asarray(
                [reference.yaw_rate_rad_s for reference in references], dtype=np.float32
            ),
        )

    def _submit_async_optimization(
        self,
        *,
        index: int,
        roll_command: float,
        current_position_ned_m: np.ndarray | None = None,
    ) -> np.ndarray:
        """Submit from the actuator command already applied in this tick."""
        if self._optimization_executor is None:
            raise RuntimeError("asynchronous optimizer is unavailable")
        if self._pending_optimization is not None:
            raise RuntimeError("cannot replace a pending optimization")
        previous_intent = self.current_command[[4, 5, 2, 3]].astype(np.float32)
        arguments = (
            np.stack(self._observation_history).copy(),
            np.stack(self._action_history).copy(),
            self._reference_horizon(index),
            previous_intent.copy(),
        )
        if current_position_ned_m is None:
            if self._last_vio_state is None:
                raise RuntimeError("cannot submit optimization without a deployable position")
            current_position_ned_m = self._last_vio_state.position_ned_m
        position = np.asarray(current_position_ned_m, dtype=np.float32)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("current deployable position must contain three finite values")
        keywords = {
            "current_position_ned_m": position.copy(),
            "roll_command": roll_command,
        }
        submitted_at = time.monotonic()
        self._pending_optimization = self._optimization_executor.submit(
            self.jepa_mppi.optimize,
            *arguments,
            **keywords,
        )
        self._pending_optimization_submitted_at = submitted_at
        self._pending_optimization_state_received_at = self._last_vio_received_at
        self._pending_optimization_reference_index = index
        self._last_planner_submission_at = submitted_at
        return previous_intent

    def _initialize_mppi(self) -> None:
        current = self._current_observation()
        deployable_state = self._deployable_control_state()
        motors = (
            np.asarray(self.applied_motors, dtype=np.float32)
            if self.applied_motors is not None
            else np.zeros(6, dtype=np.float32)
        )
        if motors.shape != (6,) or not np.all(np.isfinite(motors)):
            motors = np.zeros(6, dtype=np.float32)
        self._observation_history.clear()
        self._action_history.clear()
        self._pending_action_samples.clear()
        self._causal_history_tick_count = 0
        self.jepa_mppi.reset(np.zeros(4, dtype=np.float32))
        self._held_mppi_command = None
        self._last_mppi_update_index = -(10**9)
        self._pending_optimization_submitted_at = None
        self._pending_optimization_state_received_at = None
        self._pending_optimization_reference_index = None
        self._last_planner_submission_at = None
        # Warm GPU kernels with a shape-correct synthetic history, but do not
        # feed that history to the controller during the actual rollout.
        warm_observation = np.repeat(
            current[None], self.jepa_mppi.history_steps, axis=0
        )
        warm_action = np.repeat(motors[None], self.jepa_mppi.history_steps, axis=0)
        self.jepa_mppi.warmup(
            warm_observation,
            warm_action,
            np.zeros(4, dtype=np.float32),
            self._reference_horizon(0),
            current_position_ned_m=deployable_state.position_ned_m,
            roll_command=self._roll_safety_command(),
        )
        self.mppi_trace = {
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
            "nominal_intent": [],
            "selected_intent": [],
            "executed_intent": [],
            "accepted": [],
            "nominal_cost": [],
            "selected_cost": [],
            "predicted_improvement": [],
            "prediction_valid": [],
            "predicted_body_velocity_frd_m_s": [],
            "predicted_angular_velocity_body_frd_rad_s": [],
            "predicted_position_delta_ned_m": [],
            "timing_seconds": [],
            "core_optimization_seconds": [],
            "postprocess_seconds": [],
            "total_optimize_seconds": [],
            "planner_submitted": [],
            "planner_completed": [],
            "planner_solve_seconds": [],
            "planner_result_age_seconds": [],
            "planner_state_age_seconds": [],
            "planner_submitted_reference_index": [],
            "planner_submission_previous_intent": [],
            "planner_completed_reference_index": [],
            "planner_stale_rejected": [],
            "reason": [],
            # Compact JSON keeps the complete controller decision chain
            # replayable without putting large candidate tensors in every
            # runtime result. Controllers without this contract write ``{}``.
            "debug_chain_json": [],
        }

    def _append_causal_history_sample(self, observation: np.ndarray) -> None:
        if self.applied_motors is not None:
            latest_motor = np.asarray(self.applied_motors, dtype=np.float32)
        elif self._action_history:
            latest_motor = self._action_history[-1].copy()
        else:
            latest_motor = np.zeros(6, dtype=np.float32)
        # This callback runs before the next command is sent. ``latest_motor``
        # is therefore the demand that produced the just-arrived observation
        # and belongs to the interval ending at this tick. Include it before
        # closing the model-rate interval so consecutive tokens neither repeat
        # a boundary sample nor omit the final applied action.
        self._pending_action_samples.append(latest_motor.copy())
        self._causal_history_tick_count += 1
        if len(self._pending_action_samples) > self.model_stride:
            raise RuntimeError(
                "causal action history interval exceeded the model stride: "
                f"{len(self._pending_action_samples)} > {self.model_stride}"
            )
        if len(self._pending_action_samples) < self.model_stride:
            return
        self._observation_history.append(observation.copy())
        self._action_history.append(
            np.mean(np.stack(self._pending_action_samples), axis=0).astype(
                np.float32
            )
        )
        self._pending_action_samples.clear()

    @staticmethod
    def _planning_due(
        *,
        macro_boundary: bool,
        planner_completed: bool,
        index: int,
    ) -> bool:
        """Keep the historical successor-submission phase contract."""
        return bool(macro_boundary or (planner_completed and index > 0))

    @staticmethod
    def _submission_interval_elapsed(
        last_submission_at: float | None,
        now: float,
        minimum_interval: float,
    ) -> bool:
        """Return whether a new model evaluation may be submitted.

        Completion/application is deliberately governed by the normal planner
        state machine; this predicate gates only a successor submission.  A
        non-positive interval retains the historical unthrottled behavior.
        """
        if minimum_interval <= 0.0 or last_submission_at is None:
            return True
        elapsed = now - last_submission_at
        return elapsed >= minimum_interval or math.isclose(
            elapsed, minimum_interval, rel_tol=1e-12, abs_tol=1e-12
        )

    def _fresh_preroll_state(self) -> VioState:
        """Refresh VIO callbacks before startup feedback uses the state.

        Camera/VINS callbacks can arrive just after the fixed-rate collector
        tick.  Waiting here keeps the deployable state contract intact without
        silently reusing an old estimate or falling back to simulation truth.
        """

        deadline = time.monotonic() + max(0.20, 2.0 * self.maximum_vio_age)
        while rclpy.ok():
            if (
                self._last_vio_state is not None
                and time.monotonic() - self._last_vio_received_at
                <= self.maximum_vio_age
            ):
                return self._last_vio_state
            if time.monotonic() >= deadline:
                raise RuntimeError("canonical VIO state became stale during preroll")
            rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
            for _ in range(4):
                rclpy.spin_once(self, timeout_sec=0.0)
        raise RuntimeError("ROS stopped during controlled causal history preroll")

    def _prime_causal_history(self) -> None:
        """Use the required real history window to settle on deployable state."""

        if not self.PRIME_CAUSAL_HISTORY:
            return
        target_samples = self.jepa_mppi.history_steps
        target_ticks = target_samples * self.model_stride
        started = time.monotonic()
        state = self._fresh_preroll_state()
        initial_velocity = state.linear_velocity_body_frd_m_s.copy()
        if self.latest_depth is None or self.imu_rate is None:
            raise RuntimeError("pressure depth/IMU unavailable for controlled preroll")
        initial_depth = float(self.latest_depth)
        initial_yaw = float(quaternion_rpy(state.orientation_body_to_ned_xyzw)[2])
        settling_position = state.position_ned_m.copy()
        settling_position[2] = initial_depth
        settling_reference = BodyTwistReference(
            position_preview_ned_m=settling_position,
            yaw_preview_rad=initial_yaw,
            linear_velocity_body_frd_m_s=np.zeros(3, dtype=np.float64),
            yaw_rate_rad_s=0.0,
        )
        self._settling_pid.reset()
        command_trace: list[list[float]] = []
        measured_motor_trace: list[list[float]] = []
        for _ in range(target_ticks):
            if not rclpy.ok():
                raise RuntimeError("ROS stopped during causal history preroll")
            tick_started = time.monotonic()
            state = self._fresh_preroll_state()
            observation = self._current_observation()
            # Pair the current observation with the measured motor action that
            # produced it before computing and sending the next MANUAL command.
            measured_motor = (
                np.asarray(self.applied_motors, dtype=np.float32)
                if self.applied_motors is not None
                else np.zeros(6, dtype=np.float32)
            )
            measured_motor_trace.append(measured_motor.tolist())
            self._append_causal_history_sample(observation)
            assert self.latest_depth is not None and self.imu_rate is not None
            pressure_depth_position = state.position_ned_m.copy()
            pressure_depth_position[2] = float(self.latest_depth)
            settling_state = VioState(
                position_ned_m=pressure_depth_position,
                orientation_body_to_ned_xyzw=(
                    state.orientation_body_to_ned_xyzw
                ),
                linear_velocity_body_frd_m_s=(
                    state.linear_velocity_body_frd_m_s
                ),
                angular_velocity_body_frd_rad_s=self.imu_rate,
                timestamp_seconds=state.timestamp_seconds,
                source=f"{state.source}+ardusub_pressure_imu",
            )
            requested = clamp_manual_control(
                self._settling_pid.step(
                    settling_reference,
                    settling_state,
                    self.period,
                ).manual_control_normalized,
                self.action_limit,
            )
            self.current_command = slew_towards(
                self.current_command,
                requested,
                self.action_slew_rate,
                self.period,
            )
            command_trace.append(self.current_command.tolist())
            self._send_command(self.current_command)
            rclpy.spin_once(self, timeout_sec=min(0.01, self.period))
            for _ in range(6):
                rclpy.spin_once(self, timeout_sec=0.0)
            self._verify_execution_heartbeat()
            self._safety_gate()
            remaining = self.period - (time.monotonic() - tick_started)
            if remaining > 0.0:
                time.sleep(remaining)

        if (
            len(self._observation_history) != target_samples
            or len(self._action_history) != target_samples
        ):
            raise RuntimeError("causal JEPA history preroll did not reach target length")
        # The formal trajectory begins at the settling terminal state. Rebase
        # the stored depth feature as well so history and online observations
        # share the new pressure-depth origin without a discontinuity.
        state = self._fresh_preroll_state()
        if self.latest_depth is None:
            raise RuntimeError("pressure depth unavailable after controlled preroll")
        terminal_depth = float(self.latest_depth)
        previous_depth_anchor = float(self._depth_anchor)
        depth_rebase = previous_depth_anchor - terminal_depth
        for observation in self._observation_history:
            observation[-1] += depth_rebase
        self._depth_anchor = terminal_depth
        self.pid_anchor_position_ned_m = state.position_ned_m.copy()
        self.pid_anchor_position_ned_m[2] = terminal_depth
        terminal_rpy = quaternion_rpy(state.orientation_body_to_ned_xyzw)
        self.pid_anchor_yaw_rad = float(terminal_rpy[2])
        terminal_intent = self.current_command[[4, 5, 2, 3]].astype(np.float32)
        self.jepa_mppi.reset(terminal_intent)
        self._held_mppi_command = terminal_intent.copy()
        self._history_preroll_seconds = time.monotonic() - started
        self._history_preroll_model_samples = len(self._observation_history)
        terminal_speed = float(np.linalg.norm(state.linear_velocity_body_frd_m_s))
        terminal_yaw_error = float(
            math.atan2(
                math.sin(float(terminal_rpy[2]) - initial_yaw),
                math.cos(float(terminal_rpy[2]) - initial_yaw),
            )
        )
        terminal_depth_error = terminal_depth - initial_depth
        speed_threshold = 0.03
        depth_threshold = 0.03
        yaw_threshold = 0.05
        self._causal_preroll_audit.update(
            {
                "initial_body_velocity_frd_m_s": initial_velocity.tolist(),
                "terminal_body_velocity_frd_m_s": (
                    state.linear_velocity_body_frd_m_s.tolist()
                ),
                "terminal_linear_speed_m_s": terminal_speed,
                "initial_pressure_depth_down_m": initial_depth,
                "terminal_pressure_depth_down_m": terminal_depth,
                "terminal_pressure_depth_error_m": terminal_depth_error,
                "initial_yaw_rad": initial_yaw,
                "terminal_yaw_rad": float(terminal_rpy[2]),
                "terminal_yaw_error_rad": terminal_yaw_error,
                "terminal_roll_rad": float(terminal_rpy[0]),
                "terminal_command": self.current_command.tolist(),
                "terminal_4dof_intent": terminal_intent.tolist(),
                "manual_command_tick_history": command_trace,
                "measured_motor_tick_history": measured_motor_trace,
                "model_action_history": [
                    action.tolist() for action in self._action_history
                ],
                "model_action_history_nonzero_fraction": float(
                    np.mean(np.abs(np.stack(self._action_history)) > 1e-4)
                ),
                "model_action_history_peak": float(
                    np.max(np.abs(np.stack(self._action_history)))
                ),
                "thresholds": {
                    "linear_speed_m_s": speed_threshold,
                    "absolute_pressure_depth_error_m": depth_threshold,
                    "absolute_yaw_error_rad": yaw_threshold,
                },
                "threshold_results": {
                    "linear_speed_passed": terminal_speed <= speed_threshold,
                    "pressure_depth_passed": (
                        abs(terminal_depth_error) <= depth_threshold
                    ),
                    "yaw_passed": abs(terminal_yaw_error) <= yaw_threshold,
                },
                "settling_thresholds_passed": bool(
                    terminal_speed <= speed_threshold
                    and abs(terminal_depth_error) <= depth_threshold
                    and abs(terminal_yaw_error) <= yaw_threshold
                ),
                "elapsed_seconds": self._history_preroll_seconds,
            }
        )

    def _compute_initial_plan(self) -> None:
        """Solve the first model plan before the reference clock starts."""

        self._fresh_preroll_state()
        state = self._deployable_control_state()
        previous_intent = self.current_command[[4, 5, 2, 3]].astype(np.float32)
        submitted_at = time.monotonic()
        self._last_planner_submission_at = submitted_at
        state_received_at = self._last_vio_received_at
        asynchronous_config = self.jepa_mppi.config
        self.jepa_mppi.config = replace(
            asynchronous_config,
            planner_latency_compensation_seconds=0.0,
            deadline_seconds=0.0,
        )
        try:
            self._initial_plan_result = self.jepa_mppi.optimize(
                np.stack(self._observation_history).copy(),
                np.stack(self._action_history).copy(),
                self._reference_horizon(0),
                previous_intent.copy(),
                current_position_ned_m=state.position_ned_m.copy(),
                roll_command=self._roll_safety_command(),
            )
        finally:
            self.jepa_mppi.config = asynchronous_config
        completed_at = time.monotonic()
        self._initial_plan_solve_seconds = completed_at - submitted_at
        self._initial_plan_state_age_seconds = completed_at - state_received_at

    @staticmethod
    def _debug_chain_json(diagnostics: dict[str, Any]) -> str:
        groups = (
            "input_contract",
            "history",
            "context",
            "candidate_generation",
            "model_rollout",
            "cost",
            "verification",
            "timing_breakdown_seconds",
        )

        def serializable(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(key): serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        compact: dict[str, Any] = {
            name: serializable(diagnostics[name])
            for name in groups
            if name in diagnostics
        }
        # The direct-action controller exposes compact diagnostics at the top
        # level. Retain the fields needed to distinguish a poor proposal from
        # a poor learned rollout; the previous nested-only filter serialized
        # these decisions as an empty ``{}`` trace.
        decision_fields = (
            "candidate_sampling_centres",
            "wide_noise_fraction",
            "wide_noise_scale",
            "wide_noise_candidate_count",
            "wide_noise_supported_fraction",
            "local_noise_supported_fraction",
            "wide_candidate_weight_mass",
            "best_candidate_sampling_centre",
            "best_candidate_is_wide",
            "selected_plan_distance_to_seed_normalized",
            "reference_seed_first_command",
            "feedback_seed_first_command",
            "terminal_stop_weight",
            "previous_command",
            "executed_command",
            "nominal_cost",
            "nominal_cost_components",
            "selected_cost",
            "best_candidate_cost",
            "predicted_improvement",
            "selected_cost_components",
            "supported_candidate_count",
            "effective_sample_size",
            "adaptive_temperature",
            "weighted_plan_verified",
            "verification_skipped_reason",
            "selected_plan_source",
            "planner_latency_compensation_seconds",
            "core_optimization_seconds",
            "postprocess_seconds",
            "total_optimize_seconds",
            "timing_seconds",
        )
        compact.update(
            {
                name: serializable(diagnostics[name])
                for name in decision_fields
                if name in diagnostics
            }
        )
        return json.dumps(compact, sort_keys=True, separators=(",", ":"))

    def _save_mppi_trace(self) -> None:
        if not self.mppi_trace.get("time_seconds"):
            return
        target = self.mppi_trace_path
        if not target and self.report_path:
            target = str(
                Path(self.report_path)
                .expanduser()
                .resolve()
                .with_suffix(self.TRACE_SUFFIX)
            )
        if not target:
            return
        path = Path(target).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, **{name: np.asarray(value) for name, value in self.mppi_trace.items()}
        )
        self.mppi_trace_path = str(path)

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        super()._annotate_rollout_provenance(episode_path)
        metadata = load_metadata(episode_path)
        controller = metadata.setdefault("controller_provenance", {})
        controller.update(
            {
                "controller_id": self.CONTROLLER_ID,
                "controller_type": self.CONTROLLER_TYPE,
                "state_source": self.vio_source_id,
                "stage2_checkpoint": self.stage2_checkpoint,
                self.CONFIG_METADATA_KEY: self.jepa_config.as_dict(),
                self.DEVICE_METADATA_KEY: self.jepa_device,
                "trajectory_pid_used": False,
                "nominal_plan_source": "shifted_previous_model_plan",
                "candidate_sampling_centres": (
                    [
                        "shifted_previous_model_plan",
                        "reference_action_seed",
                        "response_feedback_seed",
                    ]
                    if self.jepa_config.reference_seed_sample_fraction > 0.0
                    else ["shifted_previous_model_plan"]
                ),
                "roll_safety_controller": "fixed_vio_imu_roll_pd",
                self.TRACE_METADATA_KEY: self.mppi_trace_path or None,
                "model_input_boundary": {
                    "observation": "causal VINS stereo VIO + ArduSub IMU + ArduSub depth",
                    "action_history": "past measured six-motor normalized outputs",
                    "future_reference": "known planner/reference body twist + NED position preview + yaw preview",
                    "ground_truth_feedback": False,
                },
                "causal_history_preroll": {
                    "enabled": bool(self.PRIME_CAUSAL_HISTORY),
                    "model_samples": self._history_preroll_model_samples,
                    "elapsed_seconds": self._history_preroll_seconds,
                    "synthetic_samples": 0,
                    **self._causal_preroll_audit,
                },
                "initial_plan_synchronous": True,
                "async_successor_submission_boundary": (
                    "actual command sent before successor submission"
                ),
                "offline_evaluation_trace": {
                    "contains_ground_truth": True,
                    "feedback_access_allowed": False,
                    "purpose": "post-rollout trajectory scoring only",
                },
            }
        )
        write_metadata(episode_path, metadata)

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
            # Removing the passive pre-record wait also removes its incidental
            # callback pumping. Explicitly cross the reset-to-controller
            # boundary on a fresh deployable VIO state.
            self._fresh_preroll_state()
            self._initialize_pid_anchor()
            self._depth_anchor = float(self.latest_depth)
            self._initialize_mppi()
            # GPU warmup can outlive the VIO freshness contract while ROS
            # callbacks are blocked. Require a strictly newer post-warmup VINS
            # estimate, then confirm the independent truth audit is also fresh.
            # Truth remains evaluation-only and is never passed to the model.
            refresh_requested_at = time.monotonic()
            pre_refresh_timestamp = (
                self._last_vio_state.timestamp_seconds
                if self._last_vio_state is not None
                else None
            )
            self._spin_until_fresh_vio(
                received_after=refresh_requested_at,
                timestamp_after=pre_refresh_timestamp,
                timeout_seconds=1.5,
                failure_message="canonical VIO did not refresh after GPU warmup",
            )
            imu_refresh_deadline = time.monotonic() + 1.5
            while (
                rclpy.ok()
                and time.monotonic() - self.imu_received_at > 0.10
                and time.monotonic() < imu_refresh_deadline
            ):
                rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
            if time.monotonic() - self.imu_received_at > 0.10:
                raise RuntimeError("ArduSub IMU did not refresh after GPU warmup")
            refresh_deadline = time.monotonic() + 1.0
            while (
                rclpy.ok()
                and time.monotonic() - self.validation_odometry_received_at > 0.10
                and time.monotonic() < refresh_deadline
            ):
                rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
            self._deployable_control_state()
            rclpy.spin_once(self, timeout_sec=min(0.02, self.period))
            self._prime_causal_history()
            self._compute_initial_plan()
            if self.manage_recorder:
                self._start_recorder()
                recorder_started = True
            reference_started = time.monotonic()
            self._begin_external_current_episode()
            count = len(self.command_arrays["body_twist_reference"])
            for index in range(count):
                tick_started = time.monotonic()
                elapsed_seconds = self._formal_episode_time_seconds(index)
                current_velocity = self._publish_external_current(elapsed_seconds)
                state = self._deployable_control_state()
                truth = self._simulation_audit_state()
                roll_command = self._roll_safety_command()
                current_obs = self._current_observation()
                previous_intent = self.current_command[[4, 5, 2, 3]].astype(np.float32)
                self._append_causal_history_sample(current_obs)
                history_steps = self.jepa_mppi.history_steps
                macro_boundary = (index + 1) % self.model_stride == 0
                completed_result = None
                planner_submitted = False
                planner_completed = False
                planner_stale_rejected = False
                planner_solve_seconds = math.nan
                planner_result_age_seconds = math.nan
                planner_state_age_seconds = math.nan
                planner_submitted_reference_index = -1
                planner_submission_previous_intent = np.full(
                    4, np.nan, dtype=np.float32
                )
                planner_completed_reference_index = -1
                hold_reason: str | None = None
                if index == 0 and self._initial_plan_result is not None:
                    completed_result = self._initial_plan_result
                    self._initial_plan_result = None
                    planner_submitted = True
                    planner_completed = True
                    planner_solve_seconds = self._initial_plan_solve_seconds
                    planner_result_age_seconds = self._initial_plan_solve_seconds
                    planner_state_age_seconds = self._initial_plan_state_age_seconds
                    planner_submitted_reference_index = 0
                    planner_completed_reference_index = 0
                if (
                    completed_result is None
                    and
                    self._pending_optimization is not None
                    and self._pending_optimization.done()
                ):
                    completed_result = self._pending_optimization.result()
                    completed_at = time.monotonic()
                    planner_completed = True
                    planner_solve_seconds = float(
                        completed_result.diagnostics["timing_seconds"]
                    )
                    if self._pending_optimization_submitted_at is not None:
                        planner_result_age_seconds = (
                            completed_at - self._pending_optimization_submitted_at
                        )
                    if self._pending_optimization_state_received_at is not None:
                        planner_state_age_seconds = (
                            completed_at
                            - self._pending_optimization_state_received_at
                        )
                    if self._pending_optimization_reference_index is not None:
                        planner_completed_reference_index = (
                            self._pending_optimization_reference_index
                        )
                    self._pending_optimization = None
                    self._pending_optimization_submitted_at = None
                    self._pending_optimization_state_received_at = None
                    self._pending_optimization_reference_index = None
                    maximum_plan_age = float(
                        getattr(self.jepa_config, "maximum_plan_age_seconds", 0.30)
                    )
                    if (
                        maximum_plan_age > 0.0
                        and planner_state_age_seconds > maximum_plan_age
                    ):
                        planner_stale_rejected = True
                        completed_result = None
                        hold_reason = "stale_plan_rejected"
                        # A stale result invalidates only the action warm start.
                        self.jepa_mppi.reset_plan(
                            self._held_mppi_command
                            if self._held_mppi_command is not None
                            else previous_intent
                        )

                history_ready = len(self._observation_history) >= history_steps
                # CUDA completion is not a new observation token. Keep
                # applying a completed result immediately, but throttle only
                # the next optimizer launch to the configured model cadence.
                planner_minimum_interval = float(
                    getattr(
                        self.jepa_config,
                        "planner_minimum_interval_seconds",
                        0.12,
                    )
                )
                cadence_due = self._submission_interval_elapsed(
                    getattr(self, "_last_planner_submission_at", None),
                    time.monotonic(),
                    planner_minimum_interval,
                )
                planning_due = self._planning_due(
                    macro_boundary=macro_boundary,
                    planner_completed=planner_completed,
                    index=index,
                ) or (planner_minimum_interval > 0.0 and cadence_due)
                submission_due = planning_due and cadence_due
                if (
                    submission_due
                    and history_ready
                    and self._optimization_executor is None
                ):
                    reference_horizon = self._reference_horizon(index)
                    arguments = (
                        np.stack(self._observation_history).copy(),
                        np.stack(self._action_history).copy(),
                        reference_horizon,
                        previous_intent.copy(),
                    )
                    keywords = {
                        "current_position_ned_m": state.position_ned_m.copy(),
                        "roll_command": roll_command,
                    }
                    submitted_at = time.monotonic()
                    self._last_planner_submission_at = submitted_at
                    completed_result = self.jepa_mppi.optimize(
                        *arguments,
                        **keywords,
                    )
                    planner_submitted = True
                    planner_completed = True
                    planner_solve_seconds = float(
                        completed_result.diagnostics["timing_seconds"]
                    )
                    planner_result_age_seconds = planner_solve_seconds
                    planner_state_age_seconds = (
                        time.monotonic() - self._last_vio_received_at
                    )
                    planner_submitted_reference_index = index
                    planner_submission_previous_intent = previous_intent.copy()
                    planner_completed_reference_index = index

                if completed_result is None:
                    # Do not expose repeated/synthetic history to Stage 2.
                    # The asynchronous planner holds this command between
                    # model-rate updates while optimization runs in the background.
                    result_command = (
                        self._held_mppi_command.copy()
                        if self._held_mppi_command is not None
                        else previous_intent.copy()
                    )
                    nominal_intent = result_command.copy()
                    selected_manual = intent_to_manual_control(result_command)
                    accepted = False
                    reason = (
                        hold_reason
                        if hold_reason is not None
                        else (
                            "hold_previous_plan"
                            if self._held_mppi_command is not None
                            else (
                                "planner_pending" if history_ready else "history_warmup"
                            )
                        )
                    )
                    nominal_cost = math.nan
                    selected_cost = math.nan
                    predicted_improvement = 0.0
                    timing_seconds = 0.0
                    core_optimization_seconds = 0.0
                    postprocess_seconds = 0.0
                    total_optimize_seconds = 0.0
                    prediction_valid = False
                    debug_chain_json = "{}"
                    predicted_body_velocity = np.full(
                        (self.jepa_mppi.rollout_horizon_steps, 3),
                        np.nan,
                        dtype=np.float32,
                    )
                    predicted_angular_velocity = np.full(
                        (self.jepa_mppi.rollout_horizon_steps, 3),
                        np.nan,
                        dtype=np.float32,
                    )
                    predicted_position_delta = np.full(
                        (self.jepa_mppi.rollout_horizon_steps, 3),
                        np.nan,
                        dtype=np.float32,
                    )
                else:
                    self._held_mppi_command = completed_result.command.copy()
                    self._last_mppi_update_index = index
                    result_command = completed_result.command
                    nominal_intent = completed_result.nominal_command[0].copy()
                    selected_manual = intent_to_manual_control(result_command)
                    accepted = completed_result.accepted
                    reason = completed_result.reason
                    nominal_cost = completed_result.diagnostics["nominal_cost"]
                    selected_cost = completed_result.diagnostics["selected_cost"]
                    predicted_improvement = completed_result.diagnostics[
                        "predicted_improvement"
                    ]
                    timing_seconds = completed_result.diagnostics["timing_seconds"]
                    core_optimization_seconds = completed_result.diagnostics[
                        "core_optimization_seconds"
                    ]
                    postprocess_seconds = completed_result.diagnostics[
                        "postprocess_seconds"
                    ]
                    total_optimize_seconds = completed_result.diagnostics[
                        "total_optimize_seconds"
                    ]
                    debug_chain_json = self._debug_chain_json(
                        completed_result.diagnostics
                    )
                    predicted_physical = np.asarray(
                        completed_result.predicted_physical_observation,
                        dtype=np.float32,
                    )
                    if predicted_physical.ndim != 2 or predicted_physical.shape[1] < 12:
                        raise RuntimeError(
                            "Stage 2 prediction must expose velocity and angular velocity"
                        )
                    prediction_valid = bool(np.isfinite(predicted_physical).all())
                    predicted_body_velocity = predicted_physical[:, 6:9].copy()
                    predicted_angular_velocity = predicted_physical[:, 9:12].copy()
                    if completed_result.predicted_position_delta_ned_m is None:
                        predicted_position_delta = np.full(
                            (predicted_physical.shape[0], 3),
                            np.nan,
                            dtype=np.float32,
                        )
                    else:
                        predicted_position_delta = np.asarray(
                            completed_result.predicted_position_delta_ned_m,
                            dtype=np.float32,
                        ).copy()
                        prediction_valid = bool(
                            prediction_valid and np.isfinite(predicted_position_delta).all()
                        )
                # Roll is the only fixed feedback channel and is not one of the
                # four trajectory actions optimized by MPPI.
                selected_manual[1] = roll_command
                self.current_command = slew_towards(
                    self.current_command,
                    clamp_manual_control(selected_manual, self.action_limit),
                    self.action_slew_rate,
                    self.period,
                )
                self._send_command(self.current_command)
                executed_intent = self.current_command[[4, 5, 2, 3]].copy()
                if (
                    self._optimization_executor is not None
                    and submission_due
                    and history_ready
                    and self._pending_optimization is None
                ):
                    # Apply a completed plan before launching its successor. The
                    # command above is the causal actuator boundary in force while
                    # the asynchronous solve runs. Submitting before this update
                    # reused the old command and duplicated consecutive first knots.
                    planner_submission_previous_intent = (
                        self._submit_async_optimization(
                            index=index,
                            roll_command=roll_command,
                            current_position_ned_m=state.position_ned_m,
                        )
                    )
                    planner_submitted = True
                    planner_submitted_reference_index = index
                self.mppi_trace["time_seconds"].append(
                    time.monotonic() - reference_started
                )
                self.mppi_trace["reference_body_twist"].append(
                    np.asarray(
                        self.command_arrays["body_twist_reference"][index]
                    ).copy()
                )
                trajectory_reference = self._trajectory_reference(index)
                self.mppi_trace["reference_position_ned_m"].append(
                    trajectory_reference.position_preview_ned_m.copy()
                )
                self.mppi_trace["reference_yaw_rad"].append(
                    trajectory_reference.yaw_preview_rad
                )
                self.mppi_trace["actual_position_ned_m"].append(
                    truth.position_ned_m.copy()
                )
                self.mppi_trace["actual_yaw_rad"].append(
                    quaternion_rpy(truth.orientation_body_to_ned_xyzw)[2]
                )
                self.mppi_trace["estimated_position_ned_m"].append(
                    state.position_ned_m.copy()
                )
                self.mppi_trace["estimated_yaw_rad"].append(
                    quaternion_rpy(state.orientation_body_to_ned_xyzw)[2]
                )
                self.mppi_trace["external_current_velocity_ned_m_s"].append(
                    current_velocity.copy()
                )
                self.mppi_trace["external_current_time_seconds"].append(
                    elapsed_seconds
                )
                self.mppi_trace["nominal_intent"].append(nominal_intent.copy())
                self.mppi_trace["selected_intent"].append(result_command.copy())
                self.mppi_trace["executed_intent"].append(executed_intent)
                self.mppi_trace["accepted"].append(accepted)
                self.mppi_trace["nominal_cost"].append(nominal_cost)
                self.mppi_trace["selected_cost"].append(selected_cost)
                self.mppi_trace["predicted_improvement"].append(predicted_improvement)
                self.mppi_trace["prediction_valid"].append(prediction_valid)
                self.mppi_trace["predicted_body_velocity_frd_m_s"].append(
                    predicted_body_velocity
                )
                self.mppi_trace[
                    "predicted_angular_velocity_body_frd_rad_s"
                ].append(predicted_angular_velocity)
                self.mppi_trace["predicted_position_delta_ned_m"].append(
                    predicted_position_delta
                )
                self.mppi_trace["timing_seconds"].append(timing_seconds)
                self.mppi_trace["core_optimization_seconds"].append(
                    core_optimization_seconds
                )
                self.mppi_trace["postprocess_seconds"].append(postprocess_seconds)
                self.mppi_trace["total_optimize_seconds"].append(
                    total_optimize_seconds
                )
                self.mppi_trace["planner_submitted"].append(planner_submitted)
                self.mppi_trace["planner_completed"].append(planner_completed)
                self.mppi_trace["planner_solve_seconds"].append(
                    planner_solve_seconds
                )
                self.mppi_trace["planner_result_age_seconds"].append(
                    planner_result_age_seconds
                )
                self.mppi_trace["planner_state_age_seconds"].append(
                    planner_state_age_seconds
                )
                self.mppi_trace["planner_submitted_reference_index"].append(
                    planner_submitted_reference_index
                )
                self.mppi_trace["planner_submission_previous_intent"].append(
                    planner_submission_previous_intent
                )
                self.mppi_trace["planner_completed_reference_index"].append(
                    planner_completed_reference_index
                )
                self.mppi_trace["planner_stale_rejected"].append(
                    planner_stale_rejected
                )
                self.mppi_trace["reason"].append(reason)
                self.mppi_trace["debug_chain_json"].append(debug_chain_json)
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
            self.get_logger().error(f"JEPA-MPPI rollout failed: {failure}")
        finally:
            self._end_external_current_episode()
            if self._optimization_executor is not None:
                self._optimization_executor.shutdown(wait=True, cancel_futures=True)
            try:
                self._save_mppi_trace()
            except OSError as error:
                failure = failure or f"could not save MPPI trace: {error}"
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
            else {
                "pass": False,
                "error": "episode was not saved",
            }
        )
        response_metrics: dict[str, Any] = {}
        trajectory_metrics: dict[str, Any] = {}
        state_estimation_metrics: dict[str, Any] = {}
        if episode_path and self.mppi_trace.get("time_seconds"):
            with np.load(episode_path, allow_pickle=False) as archive:
                actual = np.column_stack(
                    (
                        archive["vio_linear_velocity_body_frd_m_s"],
                        archive["vio_angular_velocity_body_frd_rad_s"][:, 2],
                    )
                )
                elapsed = np.asarray(archive["time"], dtype=np.float64)
            trace_time = np.asarray(self.mppi_trace["time_seconds"], dtype=np.float64)
            trace_time -= trace_time[0]
            reference = np.asarray(
                self.mppi_trace["reference_body_twist"], dtype=np.float64
            )
            aligned_reference = np.column_stack(
                [
                    np.interp(elapsed, trace_time, reference[:, axis])
                    for axis in range(4)
                ]
            )
            response_metrics = {
                "reference_rms": np.sqrt(
                    np.mean(aligned_reference**2, axis=0)
                ).tolist(),
                "vio_response_rms": np.sqrt(np.mean(actual**2, axis=0)).tolist(),
                "vio_response_peak": np.max(np.abs(actual), axis=0).tolist(),
                "yaw_rms_gain": float(
                    np.sqrt(np.mean(actual[:, 3] ** 2))
                    / max(np.sqrt(np.mean(aligned_reference[:, 3] ** 2)), 1e-9)
                ),
            }
            reference_position = np.asarray(
                self.mppi_trace["reference_position_ned_m"], dtype=np.float64
            )
            reference_yaw = np.asarray(
                self.mppi_trace["reference_yaw_rad"], dtype=np.float64
            )
            actual_position = np.asarray(
                self.mppi_trace["actual_position_ned_m"], dtype=np.float64
            )
            estimated_position = np.asarray(
                self.mppi_trace["estimated_position_ned_m"], dtype=np.float64
            )
            actual_yaw = np.asarray(
                self.mppi_trace["actual_yaw_rad"], dtype=np.float64
            )
            estimated_yaw = np.asarray(
                self.mppi_trace["estimated_yaw_rad"], dtype=np.float64
            )
            trajectory_metrics = {
                "ground_truth": summarize_trajectory_tracking(
                    reference_position,
                    actual_position - actual_position[0] + reference_position[0],
                    reference_yaw,
                    np.arctan2(
                        np.sin(
                            actual_yaw
                            - float(actual_yaw[0])
                            + reference_yaw[0]
                        ),
                        np.cos(
                            actual_yaw
                            - float(actual_yaw[0])
                            + reference_yaw[0]
                        ),
                    ),
                ),
            }
            state_estimation_metrics = summarize_state_estimation_audit(
                actual_position,
                estimated_position,
                actual_yaw,
                estimated_yaw,
            )
        timing_values = np.asarray(
            self.mppi_trace.get("timing_seconds", []), dtype=np.float64
        )
        core_timing_values = np.asarray(
            self.mppi_trace.get("core_optimization_seconds", []), dtype=np.float64
        )
        postprocess_timing_values = np.asarray(
            self.mppi_trace.get("postprocess_seconds", []), dtype=np.float64
        )
        update_mask = np.isfinite(timing_values) & (timing_values > 0.0)
        accepted_values = np.asarray(self.mppi_trace.get("accepted", []), dtype=bool)
        predicted_values = np.asarray(
            self.mppi_trace.get("predicted_improvement", []), dtype=np.float64
        )
        finite_improvement_mask = update_mask & np.isfinite(predicted_values)
        all_timing_values = (
            timing_values if timing_values.size else np.asarray([math.inf])
        )
        all_accepted_values = (
            accepted_values if accepted_values.size else np.asarray([False])
        )
        planner_result_age = np.asarray(
            self.mppi_trace.get("planner_result_age_seconds", []), dtype=np.float64
        )
        planner_state_age = np.asarray(
            self.mppi_trace.get("planner_state_age_seconds", []), dtype=np.float64
        )
        finite_result_age = np.isfinite(planner_result_age)
        finite_state_age = np.isfinite(planner_state_age)
        planner_submitted_values = np.asarray(
            self.mppi_trace.get("planner_submitted", []), dtype=bool
        )
        planner_completed_values = np.asarray(
            self.mppi_trace.get("planner_completed", []), dtype=bool
        )
        planner_stale_values = np.asarray(
            self.mppi_trace.get("planner_stale_rejected", []), dtype=bool
        )
        reason_values = np.asarray(
            self.mppi_trace.get("reason", []), dtype=str
        )
        reason_names, reason_counts = np.unique(reason_values, return_counts=True)
        decision_reason_counts = {
            str(name): int(count)
            for name, count in zip(reason_names, reason_counts, strict=True)
        }
        executed_intent = np.asarray(
            self.mppi_trace.get("executed_intent", []), dtype=np.float64
        )
        manual_saturation_fraction = (
            float(
                np.mean(
                    np.any(
                        np.abs(executed_intent) >= self.action_limit - 1e-6,
                        axis=1,
                    )
                )
            )
            if executed_intent.ndim == 2 and executed_intent.shape[0] > 0
            else 1.0
        )
        ground_truth_trajectory = trajectory_metrics.get("ground_truth", {})
        ground_truth_error = ground_truth_trajectory.get(
            "time_aligned_position_error", {}
        )
        report = {
            "protocol_id": self.PROTOCOL_ID,
            "execution_mode": self.mode,
            "execution_pass": bool(
                failure is None
                and motors.size
                and saturation_fraction <= self.envelope["motor_saturation_ratio"]
                and episode_report.get("pass") is True
            ),
            "failure": failure,
            "command_path": str(self.command_path),
            "data_split": self.command_provenance["data_split"],
            "command_recipe": self.command_provenance["recipe"],
            "training_bank_protocol_id": self.command_provenance[
                "training_bank_protocol_id"
            ],
            "physics_domain": self.domain_provenance,
            "physics_domain_runtime": self.physics_domain_runtime,
            "episode_reset_convergence": self.reset_convergence_gate,
            "vins_reset_lifecycle": self.vins_reset_lifecycle,
            "episode_path": episode_path,
            "episode_contract": episode_report,
            "episode_contract_pass": bool(episode_report.get("pass")),
            "episode_contract_failed_checks": [
                name
                for name, passed in episode_report.get("checks", {}).items()
                if not passed
            ],
            "stage2_checkpoint": self.stage2_checkpoint,
            "jepa_mppi_config_path": self.jepa_config_path,
            self.DEVICE_METADATA_KEY: self.jepa_device,
            "controller_provenance": {
                "controller_id": self.CONTROLLER_ID,
                "controller_type": self.CONTROLLER_TYPE,
                "trajectory_pid_used": bool(
                    getattr(self, "TRAJECTORY_PID_USED", False)
                ),
                "nominal_plan_source": str(
                    getattr(
                        self,
                        "NOMINAL_PLAN_SOURCE",
                        "shifted_previous_model_plan",
                    )
                ),
                "roll_safety_controller": "fixed_vio_imu_roll_pd",
                "accepted_fraction": float(np.mean(all_accepted_values)),
                "accepted_fraction_on_updates": float(
                    np.mean(accepted_values[update_mask])
                    if np.any(update_mask)
                    else 0.0
                ),
                "optimization_update_fraction": (
                    float(np.mean(update_mask)) if len(update_mask) else 0.0
                ),
                "proposal_hold_fraction": (
                    float(1.0 - np.mean(update_mask)) if len(update_mask) else 1.0
                ),
                "planner_async": self._optimization_executor is not None,
                "async_successor_submission_boundary": (
                    "actual command sent before successor submission"
                ),
                "planner_submission_count": int(np.sum(planner_submitted_values)),
                "planner_completion_count": int(np.sum(planner_completed_values)),
                "planner_stale_rejection_count": int(np.sum(planner_stale_values)),
                "planner_stale_rejection_fraction_on_completions": float(
                    np.sum(planner_stale_values)
                    / max(int(np.sum(planner_completed_values)), 1)
                ),
                "planner_result_age_p50_seconds": float(
                    np.percentile(planner_result_age[finite_result_age], 50)
                    if np.any(finite_result_age)
                    else math.inf
                ),
                "planner_result_age_p95_seconds": float(
                    np.percentile(planner_result_age[finite_result_age], 95)
                    if np.any(finite_result_age)
                    else math.inf
                ),
                "planner_state_age_p50_seconds": float(
                    np.percentile(planner_state_age[finite_state_age], 50)
                    if np.any(finite_state_age)
                    else math.inf
                ),
                "planner_state_age_p95_seconds": float(
                    np.percentile(planner_state_age[finite_state_age], 95)
                    if np.any(finite_state_age)
                    else math.inf
                ),
                "mean_predicted_improvement": float(
                    np.mean(predicted_values[finite_improvement_mask])
                    if np.any(finite_improvement_mask)
                    else 0.0
                ),
                "decision_reason_counts": decision_reason_counts,
                "deadline_failure_fraction_on_updates": float(
                    decision_reason_counts.get("deadline exceeded", 0)
                    / max(int(np.sum(update_mask)), 1)
                ),
                "prediction_valid_fraction_on_updates": float(
                    np.mean(
                        np.asarray(
                            self.mppi_trace.get("prediction_valid", []),
                            dtype=bool,
                        )[update_mask]
                    )
                    if np.any(update_mask)
                    else 0.0
                ),
                "debug_trace_path": self.mppi_trace_path or None,
                "latency_p50_seconds": float(np.percentile(all_timing_values, 50)),
                "latency_p95_seconds": float(np.percentile(all_timing_values, 95)),
                "latency_p50_update_seconds": float(
                    np.percentile(timing_values[update_mask], 50)
                    if np.any(update_mask)
                    else math.inf
                ),
                "latency_p95_update_seconds": float(
                    np.percentile(timing_values[update_mask], 95)
                    if np.any(update_mask)
                    else math.inf
                ),
                "core_optimization_p95_update_seconds": float(
                    np.percentile(core_timing_values[update_mask], 95)
                    if np.any(update_mask)
                    else math.inf
                ),
                "postprocess_p95_update_seconds": float(
                    np.percentile(postprocess_timing_values[update_mask], 95)
                    if np.any(update_mask)
                    else math.inf
                ),
            },
            "motor_sample_count": int(motors.shape[0]),
            "motor_saturation_fraction": saturation_fraction,
            "manual_saturation_fraction": manual_saturation_fraction,
            "maximum_episode_motor_output": float(
                episode_report.get("maximum_motor_output", 1.0)
            ),
            "maximum_accepted_motor_output": self.maximum_accepted_motor_output,
            "position_rmse_ned_m": ground_truth_error.get("rmse_ned_m"),
            "position_rmse_3d_m": ground_truth_error.get("position_3d_rms_m"),
            "simulation_truth_audit": episode_report.get("simulation_truth_audit"),
            "observed_response_metrics": response_metrics,
            "trajectory_metrics": trajectory_metrics,
            "state_estimation_metrics": state_estimation_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        if self.report_path:
            target = Path(self.report_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return report


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = JepaMppiRollout()
    try:
        report = node.run()
        node.get_logger().info(
            f"JEPA-MPPI CLOSED LOOP COMPLETE: execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
