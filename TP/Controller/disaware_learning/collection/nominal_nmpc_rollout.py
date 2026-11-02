"""Run one Stonefish slow-multisine episode with the nominal NMPC."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import rclpy

from disaware_learning.collection.manual_rollout import ManualRollout
from disaware_learning.control.body_twist_pid import (
    BodyTwistReference,
    BodyTwistVelocityRatePid,
    quaternion_rpy,
    wrap_angle,
    zero_twist_settling_pid_config,
)
from disaware_learning.control.contract import intent_to_manual_control
from disaware_learning.control.nominal_nmpc import (
    IMPLEMENTATION_ID,
    NmpcReference,
    NmpcResult,
    NominalNmpc,
    NominalNmpcConfig,
    load_nominal_nmpc_config,
)
from disaware_learning.control.rc import clamp_manual_control, slew_towards
from disaware_learning.data.episode_io import load_metadata, write_metadata
from disaware_learning.estimation.vio_state import VioState


ROLLOUT_PROTOCOL_ID = "stonefish_nominal_nmpc_reference_controlled_preroll_v2"


class NominalNmpcRollout(ManualRollout):
    """Reuse the hard-reset recorder while replacing PID control with NMPC."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("nmpc_config_path", "")
        self.declare_parameter("nmpc_update_steps", 8)
        self.declare_parameter("controlled_preroll_minimum_seconds", 2.50)
        self.declare_parameter("controlled_preroll_stable_seconds", 1.00)
        self.declare_parameter("controlled_preroll_timeout_seconds", 8.00)
        self.declare_parameter("controlled_preroll_max_speed_m_s", 0.03)
        # ArduSub pressure depth is quantized at centimetre scale. The formal
        # trajectory is rebased at the terminal state, so absolute depth error
        # is retained as an audit rather than used as a convergence gate.
        self.declare_parameter("controlled_preroll_max_depth_error_m", 0.05)
        self.declare_parameter("controlled_preroll_max_yaw_error_rad", 0.05)
        self.nmpc_update_steps = int(self.get_parameter("nmpc_update_steps").value)
        if self.nmpc_update_steps < 1:
            raise ValueError("nmpc_update_steps must be positive")
        config_path = str(self.get_parameter("nmpc_config_path").value).strip()
        config = (
            load_nominal_nmpc_config(config_path)
            if config_path
            else NominalNmpcConfig(
                sample_time_seconds=self.period,
                horizon_steps=4 * self.nmpc_update_steps,
                control_hold_steps=self.nmpc_update_steps,
            )
        )
        if not math.isclose(config.sample_time_seconds, self.period, rel_tol=0.02):
            raise ValueError(
                "NMPC sample time must match the 25 Hz collector period: "
                f"{config.sample_time_seconds} vs {self.period}"
            )
        if config.control_hold_steps != self.nmpc_update_steps:
            raise ValueError(
                "NMPC control_hold_steps must match nmpc_update_steps: "
                f"{config.control_hold_steps} vs {self.nmpc_update_steps}"
            )
        self.nmpc = NominalNmpc(config)
        self._settling_pid = BodyTwistVelocityRatePid(
            zero_twist_settling_pid_config()
        )
        self._controlled_preroll_minimum = self._positive_parameter(
            "controlled_preroll_minimum_seconds"
        )
        self._controlled_preroll_stable = self._positive_parameter(
            "controlled_preroll_stable_seconds"
        )
        self._controlled_preroll_timeout = self._positive_parameter(
            "controlled_preroll_timeout_seconds"
        )
        self._controlled_preroll_max_speed = self._positive_parameter(
            "controlled_preroll_max_speed_m_s"
        )
        self._controlled_preroll_max_depth_error = self._positive_parameter(
            "controlled_preroll_max_depth_error_m"
        )
        self._controlled_preroll_max_yaw_error = self._positive_parameter(
            "controlled_preroll_max_yaw_error_rad"
        )
        if (
            self._controlled_preroll_timeout
            < self._controlled_preroll_minimum
            + self._controlled_preroll_stable
        ):
            raise ValueError(
                "controlled preroll timeout must cover minimum and stable periods"
            )
        # Nominal NMPC now uses the same active zero-twist initialization as
        # JEPA-MPPI. A passive zero-command wait only lets current and buoyancy
        # establish drift before the common RMSE clock starts.
        self.pre_record_settle_seconds = 0.0
        self._controlled_preroll_audit: dict[str, Any] = {
            "policy": "converged_zero_twist_pressure_depth_hold_manual",
            "feedback_controller_used": True,
            "feedback_state": (
                "VINS position/orientation/body velocity + ArduSub pressure depth/gyro"
            ),
            "ground_truth_feedback": False,
            "settling_controller_config": self._settling_pid.config.as_dict(),
        }
        self._held_nmpc_intent = np.zeros(4, dtype=np.float64)
        self._last_nmpc_submission = -10**9
        self._solver_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nominal-nmpc",
        )
        self._solver_future: Future[NmpcResult] | None = None

    def _apply_command_slew(self, requested: np.ndarray) -> np.ndarray:
        """Apply NMPC's per-axis slew envelope at the 25 Hz send rate.

        The base collector has one scalar slew rate for all six MANUAL
        channels.  Reusing it here bypasses the NMPC yaw envelope and can
        turn a feasible plan into a sharp yaw command step.
        """
        target = clamp_manual_control(requested, self.action_limit)
        current = clamp_manual_control(self.current_command, self.action_limit)
        rates = np.full(6, self.action_slew_rate, dtype=np.float64)
        configured = np.asarray(
            self.nmpc.config.command_slew_per_second, dtype=np.float64
        )
        # MANUAL order: pitch, roll, throttle(heave), yaw, forward(surge),
        # lateral(sway). Pitch is unactuated and remains zero.
        rates[2] = configured[2]
        rates[3] = configured[3]
        rates[4] = configured[0]
        rates[5] = configured[1]
        max_step = rates * self.period
        return current + np.clip(target - current, -max_step, max_step)

    def _fresh_preroll_state(self) -> VioState:
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
        raise RuntimeError("ROS stopped during controlled PID preroll")

    def _controlled_pid_preroll(self) -> np.ndarray:
        state = self._fresh_preroll_state()
        if self.latest_depth is None or self.latest_angular_velocity is None:
            raise RuntimeError("pressure depth/gyro unavailable for controlled preroll")
        initial_velocity = state.linear_velocity_body_frd_m_s.copy()
        initial_depth = float(self.latest_depth)
        initial_yaw = float(quaternion_rpy(state.orientation_body_to_ned_xyzw)[2])
        settling_position = state.position_ned_m.copy()
        settling_position[2] = initial_depth
        reference = BodyTwistReference(
            position_preview_ned_m=settling_position,
            yaw_preview_rad=initial_yaw,
            linear_velocity_body_frd_m_s=np.zeros(3, dtype=np.float64),
            yaw_rate_rad_s=0.0,
        )
        self._settling_pid.reset()
        started = time.monotonic()
        stable_since: float | None = None
        command_trace: list[list[float]] = []
        latest_metrics: dict[str, float | bool] = {}
        while rclpy.ok():
            tick_started = time.monotonic()
            state = self._fresh_preroll_state()
            if self.latest_depth is None or self.latest_angular_velocity is None:
                raise RuntimeError(
                    "pressure depth/gyro became unavailable during controlled preroll"
                )
            pressure_position = state.position_ned_m.copy()
            pressure_position[2] = float(self.latest_depth)
            settling_state = VioState(
                position_ned_m=pressure_position,
                orientation_body_to_ned_xyzw=state.orientation_body_to_ned_xyzw,
                linear_velocity_body_frd_m_s=state.linear_velocity_body_frd_m_s,
                angular_velocity_body_frd_rad_s=(
                    self.latest_angular_velocity.copy()
                ),
                timestamp_seconds=state.timestamp_seconds,
                source=f"{state.source}+ardusub_pressure_gyro",
            )
            requested = clamp_manual_control(
                self._settling_pid.step(
                    reference,
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

            elapsed = time.monotonic() - started
            speed = float(np.linalg.norm(state.linear_velocity_body_frd_m_s))
            depth_error = float(self.latest_depth) - initial_depth
            yaw = float(quaternion_rpy(state.orientation_body_to_ned_xyzw)[2])
            yaw_error = wrap_angle(yaw - initial_yaw)
            stable = bool(
                speed <= self._controlled_preroll_max_speed
                and abs(yaw_error) <= self._controlled_preroll_max_yaw_error
            )
            stable_since = (
                (stable_since if stable_since is not None else time.monotonic())
                if stable
                else None
            )
            stable_duration = (
                0.0
                if stable_since is None
                else time.monotonic() - stable_since
            )
            latest_metrics = {
                "terminal_linear_speed_m_s": speed,
                "terminal_pressure_depth_error_m": depth_error,
                "pressure_depth_audit_passed": bool(
                    abs(depth_error) <= self._controlled_preroll_max_depth_error
                ),
                "terminal_yaw_error_rad": yaw_error,
                "stable_continuous_seconds": stable_duration,
                "thresholds_passed": stable,
            }
            if (
                elapsed >= self._controlled_preroll_minimum
                and stable_duration >= self._controlled_preroll_stable
            ):
                break
            if elapsed >= self._controlled_preroll_timeout:
                self._controlled_preroll_audit.update(
                    {
                        **latest_metrics,
                        "passed": False,
                        "elapsed_seconds": elapsed,
                        "manual_command_tick_history": command_trace,
                    }
                )
                raise RuntimeError(
                    "controlled PID preroll did not converge before timeout: "
                    f"{latest_metrics}"
                )
            remaining = self.period - (time.monotonic() - tick_started)
            if remaining > 0.0:
                time.sleep(remaining)

        terminal_state = self._fresh_preroll_state()
        terminal_intent = self.current_command[[4, 5, 2, 3]].astype(np.float64)
        self._controlled_preroll_audit.update(
            {
                **latest_metrics,
                "passed": True,
                "initial_body_velocity_frd_m_s": initial_velocity.tolist(),
                "terminal_body_velocity_frd_m_s": (
                    terminal_state.linear_velocity_body_frd_m_s.tolist()
                ),
                "initial_pressure_depth_down_m": initial_depth,
                "terminal_pressure_depth_down_m": float(self.latest_depth),
                "initial_yaw_rad": initial_yaw,
                "terminal_yaw_rad": float(
                    quaternion_rpy(terminal_state.orientation_body_to_ned_xyzw)[2]
                ),
                "terminal_command": self.current_command.tolist(),
                "terminal_4dof_intent": terminal_intent.tolist(),
                "manual_command_tick_history": command_trace,
                "elapsed_seconds": time.monotonic() - started,
                "minimum_seconds": self._controlled_preroll_minimum,
                "required_stable_seconds": self._controlled_preroll_stable,
                "timeout_seconds": self._controlled_preroll_timeout,
                "thresholds": {
                    "linear_speed_m_s": self._controlled_preroll_max_speed,
                    "absolute_pressure_depth_error_m_audit_only": (
                        self._controlled_preroll_max_depth_error
                    ),
                    "absolute_yaw_error_rad": self._controlled_preroll_max_yaw_error,
                },
            }
        )
        return terminal_intent

    def _initialize_nmpc_handoff(self, terminal_intent: np.ndarray) -> None:
        intent = np.asarray(terminal_intent, dtype=np.float64)
        if intent.shape != (4,) or not np.all(np.isfinite(intent)):
            raise ValueError("controlled preroll intent must contain four finite axes")
        self.nmpc.reset(intent)
        self._held_nmpc_intent = intent.copy()

    def _initialize_pid_anchor(self) -> None:
        terminal_intent = self._controlled_pid_preroll()
        # Anchor the formal trajectory only after the common initialization
        # controller has converged. Recorder time and trajectory RMSE start
        # after this boundary.
        super()._initialize_pid_anchor()
        self._initialize_nmpc_handoff(terminal_intent)
        self._last_nmpc_submission = -10**9
        self.pid_trace.update(
            {
                "nmpc_update": [],
                "nmpc_solve_submitted": [],
                "nmpc_solve_pending": [],
                "nmpc_solver_time_seconds": [],
                "nmpc_solver_success": [],
                "nmpc_accepted": [],
                "nmpc_reason": [],
                "nmpc_selected_cost": [],
                "nmpc_maximum_raw_motor": [],
                "nmpc_command_envelope_saturated": [],
                "nmpc_motor_soft_envelope_saturated": [],
                "nmpc_motor_hard_envelope_saturated": [],
                "requested_manual_saturated": [],
            }
        )

    def _nmpc_reference(self, index: int) -> NmpcReference:
        count = len(self.command_arrays["body_twist_reference"])
        indices = np.minimum(
            np.arange(index, index + self.nmpc.config.horizon_steps + 1),
            count - 1,
        )
        references = [self._trajectory_reference(int(item)) for item in indices]
        return NmpcReference.from_arrays(
            np.stack([value.position_preview_ned_m for value in references]),
            np.stack([value.linear_velocity_body_frd_m_s for value in references]),
            np.asarray([value.yaw_preview_rad for value in references]),
            np.asarray([value.yaw_rate_rad_s for value in references]),
            source="planner_reference_bank",
        )

    def _pid_command(self, index: int, elapsed_seconds: float) -> np.ndarray:
        state = self._deployable_control_state()
        current_reference = self._trajectory_reference(index)
        result_applied = False
        solve_submitted = False
        if self._solver_future is not None and self._solver_future.done():
            result = self._solver_future.result()
            self._solver_future = None
            self._held_nmpc_intent = result.command.copy()
            result_applied = True
            solver_time = float(result.diagnostics["solve_time_seconds"])
            solver_success = bool(result.diagnostics["solver_success"])
            accepted = bool(result.accepted)
            reason = result.reason
            selected_cost = float(result.diagnostics["selected_cost"])
            maximum_raw_motor = float(result.diagnostics["maximum_raw_motor"])
        else:
            solver_time = 0.0
            solver_success = True
            accepted = False
            reason = "hold_previous_nmpc"
            selected_cost = math.nan
            maximum_raw_motor = math.nan

        submission_due = (
            index - self._last_nmpc_submission >= self.nmpc_update_steps
        )
        if submission_due and self._solver_future is None:
            reference = self._nmpc_reference(index)
            previous_command = self.current_command[[4, 5, 2, 3]].copy()
            self._solver_future = self._solver_executor.submit(
                self.nmpc.optimize,
                state,
                reference,
                previous_command=previous_command,
            )
            self._last_nmpc_submission = index
            solve_submitted = True
            if not result_applied:
                reason = "solve_submitted_hold_previous"

        # Keep the command sent to ArduSub inside the same six-thruster
        # horizontal mixer envelope used by NMPC. The outer collector slew
        # limiter can otherwise create a small combination overshoot.
        mixer_peak = float(np.max(np.abs(self.nmpc._mixer_raw(self._held_nmpc_intent))))
        if mixer_peak > self.nmpc.config.motor_hard_limit:
            self._held_nmpc_intent *= self.nmpc.config.motor_hard_limit / mixer_peak
        manual = intent_to_manual_control(self._held_nmpc_intent)
        roll, pitch, yaw = quaternion_rpy(state.orientation_body_to_ned_xyzw)
        manual[1] = np.clip(
            -self.pid_controller.config.roll_kp * roll
            - self.pid_controller.config.roll_rate_kd
            * state.angular_velocity_body_frd_rad_s[0],
            -self.action_limit,
            self.action_limit,
        )
        manual = clamp_manual_control(manual, self.action_limit)
        envelope_metrics = self.nmpc.envelope_saturation_metrics(
            self._held_nmpc_intent
        )
        truth = self._simulation_audit_state()
        # Unlike solver cost, this remains meaningful on held-command ticks.
        maximum_raw_motor = float(envelope_metrics["maximum_raw_motor"][0])
        velocity_error = (
            current_reference.linear_velocity_body_frd_m_s
            - state.linear_velocity_body_frd_m_s
        )
        yaw_rate_error = (
            current_reference.yaw_rate_rad_s
            - state.angular_velocity_body_frd_rad_s[2]
        )
        zero3 = np.zeros(3, dtype=np.float64)
        values: dict[str, Any] = {
            "time_seconds": elapsed_seconds,
            "position_preview_ned_m": current_reference.position_preview_ned_m.copy(),
            "reference_velocity_body_frd_m_s": current_reference.linear_velocity_body_frd_m_s.copy(),
            "reference_acceleration_body_frd_m_s2": current_reference.linear_acceleration_body_frd_m_s2.copy(),
            "yaw_preview_rad": current_reference.yaw_preview_rad,
            "reference_yaw_rate_rad_s": current_reference.yaw_rate_rad_s,
            # Truth is retained only in the post-run audit trace. It is never
            # passed to NominalNmpc.optimize().
            "actual_position_ned_m": truth.position_ned_m.copy(),
            "actual_orientation_body_to_ned_xyzw": truth.orientation_body_to_ned_xyzw.copy(),
            "actual_velocity_body_frd_m_s": truth.linear_velocity_body_frd_m_s.copy(),
            "actual_yaw_rad": quaternion_rpy(truth.orientation_body_to_ned_xyzw)[2],
            "actual_yaw_rate_rad_s": truth.angular_velocity_body_frd_rad_s[2],
            "estimated_position_ned_m": state.position_ned_m.copy(),
            "estimated_orientation_body_to_ned_xyzw": state.orientation_body_to_ned_xyzw.copy(),
            "estimated_velocity_body_frd_m_s": state.linear_velocity_body_frd_m_s.copy(),
            "estimated_angular_velocity_body_frd_rad_s": state.angular_velocity_body_frd_rad_s.copy(),
            "estimated_yaw_rad": yaw,
            "vio_measurement_latency_seconds": max(
                0.0, truth.timestamp_seconds - state.timestamp_seconds
            ),
            "requested_4dof_intent": self._held_nmpc_intent.copy(),
            "requested_manual_control_normalized": manual.copy(),
            "velocity_error_body_frd_m_s": velocity_error,
            "yaw_rate_error_rad_s": yaw_rate_error,
            "linear_feedforward_term": zero3.copy(),
            "linear_acceleration_feedforward_term": zero3.copy(),
            "linear_proportional_term": zero3.copy(),
            "linear_integral_term": zero3.copy(),
            "yaw_feedforward_term": 0.0,
            "yaw_proportional_term": 0.0,
            "yaw_integral_term": 0.0,
            "roll_rad": roll,
            "pitch_rad": pitch,
            "manual_saturated": False,
            "requested_manual_saturated": bool(
                np.any(np.abs(manual) >= self.action_limit - 1e-9)
            ),
            "state_source": state.source,
            "nmpc_update": result_applied,
            "nmpc_solve_submitted": solve_submitted,
            "nmpc_solve_pending": self._solver_future is not None,
            "nmpc_solver_time_seconds": solver_time,
            "nmpc_solver_success": solver_success,
            "nmpc_accepted": accepted,
            "nmpc_reason": reason,
            "nmpc_selected_cost": selected_cost,
            "nmpc_maximum_raw_motor": maximum_raw_motor,
            "nmpc_command_envelope_saturated": (
                envelope_metrics["command_envelope_saturation_fraction"] > 0.0
            ).copy(),
            "nmpc_motor_soft_envelope_saturated": (
                envelope_metrics["motor_soft_envelope_saturation_fraction"] > 0.0
            ).copy(),
            "nmpc_motor_hard_envelope_saturated": (
                envelope_metrics["motor_hard_envelope_saturation_fraction"] > 0.0
            ).copy(),
        }
        for name, value in values.items():
            self.pid_trace[name].append(value)
        return manual

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        super()._annotate_rollout_provenance(episode_path)
        metadata = load_metadata(episode_path)
        controller = metadata.setdefault("controller_provenance", {})
        controller.update(
            {
                "controller_id": IMPLEMENTATION_ID,
                "state_source": self.vio_source_id,
                "nmpc_config": self.nmpc.config.as_dict(),
                "nmpc_update_steps": self.nmpc_update_steps,
                "model_input_boundary": {
                    "runtime_state": "causal VioState only",
                    "future_reference": "planner reference bank",
                    "ground_truth_feedback": False,
                    "physics_domain_feedback": False,
                },
                "position_outer_loop_enabled": True,
                "yaw_outer_loop_enabled": True,
                "preview_geometry_used_for_feedback": True,
                "controlled_pid_preroll": self._controlled_preroll_audit,
            }
        )
        controller.pop("pid_config", None)
        write_metadata(episode_path, metadata)

    def run(self) -> dict[str, Any]:
        try:
            report = super().run()
        finally:
            if self._solver_future is not None:
                self._solver_future.cancel()
            self._solver_executor.shutdown(wait=True, cancel_futures=True)
        updates = np.asarray(self.pid_trace.get("nmpc_update", []), dtype=bool)
        timing = np.asarray(
            self.pid_trace.get("nmpc_solver_time_seconds", []), dtype=np.float64
        )
        report["protocol_id"] = ROLLOUT_PROTOCOL_ID
        report["controller_provenance"].update(
            {
                "controller_id": IMPLEMENTATION_ID,
                "nmpc_config": self.nmpc.config.as_dict(),
                "nmpc_update_steps": self.nmpc_update_steps,
                "model_input_boundary": {
                    "runtime_state": "causal VioState only",
                    "future_reference": "planner reference bank",
                    "ground_truth_feedback": False,
                    "physics_domain_feedback": False,
                },
                "position_outer_loop_enabled": True,
                "yaw_outer_loop_enabled": True,
                "preview_geometry_used_for_feedback": True,
                "controlled_pid_preroll": self._controlled_preroll_audit,
                "solver_timing_ms": {
                    "p50": float(np.percentile(timing[updates], 50)) * 1000.0
                    if np.any(updates)
                    else math.inf,
                    "p95": float(np.percentile(timing[updates], 95)) * 1000.0
                    if np.any(updates)
                    else math.inf,
                    "max": float(np.max(timing[updates])) * 1000.0
                    if np.any(updates)
                    else math.inf,
                },
            }
        )
        command_envelope = np.asarray(
            self.pid_trace.get("nmpc_command_envelope_saturated", []), dtype=bool
        )
        motor_soft_envelope = np.asarray(
            self.pid_trace.get("nmpc_motor_soft_envelope_saturated", []), dtype=bool
        )
        motor_hard_envelope = np.asarray(
            self.pid_trace.get("nmpc_motor_hard_envelope_saturated", []), dtype=bool
        )
        if command_envelope.ndim == 2 and command_envelope.shape[1] == 4:
            report["nmpc_envelope_saturation_fraction_per_axis"] = {
                axis: float(value)
                for axis, value in zip(
                    ("surge", "sway", "heave_up", "yaw"),
                    np.mean(command_envelope, axis=0),
                )
            }
        if motor_soft_envelope.ndim == 2 and motor_soft_envelope.shape[1] == 6:
            report["nmpc_motor_soft_envelope_saturation_fraction_per_motor"] = (
                np.mean(motor_soft_envelope, axis=0).tolist()
            )
        if motor_hard_envelope.ndim == 2 and motor_hard_envelope.shape[1] == 6:
            report["nmpc_motor_hard_envelope_saturation_fraction_per_motor"] = (
                np.mean(motor_hard_envelope, axis=0).tolist()
            )
        tracking = (
            report.get("controller_provenance", {})
            .get("tracking_summary", {})
            .get("trajectory_metrics", {})
            .get("time_aligned_position_error", {})
        )
        if tracking:
            # Keep the scalar definition explicit: RMSE of the 3-D vector
            # error, not the arithmetic mean of per-axis RMSE values.
            report["position_rmse_3d_m"] = float(tracking["position_3d_rms_m"])
            report["position_rmse_ned_m"] = list(tracking["rmse_ned_m"])
        report["controller_provenance"].pop("pid_config", None)
        if self.report_path:
            Path(self.report_path).expanduser().resolve().write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return report


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = NominalNmpcRollout()
    try:
        report = node.run()
        node.get_logger().info(
            "NOMINAL NMPC STONEFISH ROLLOUT COMPLETE: "
            f"execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
