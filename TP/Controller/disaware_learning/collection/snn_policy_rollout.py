"""Bounded Stonefish rollout for the stateful multi-rate SNN policy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import rclpy

from disaware_learning.collection.jepa_mppi_rollout import JepaMppiRollout
from disaware_learning.control.jepa_mppi import ReferenceTrajectory
from disaware_learning.control.snn_policy import (
    ANCHORED_IMPLEMENTATION_ID,
    StatefulSnnPolicyController,
    discrete_model_tick_due,
)
from disaware_learning.data.episode_io import load_metadata, write_metadata


class SnnPolicyRollout(JepaMppiRollout):
    CONTROLLER_ID = ANCHORED_IMPLEMENTATION_ID
    CONTROLLER_TYPE = "multirate_anchored_snn_correction_policy"
    PROTOCOL_ID = "multirate_anchored_snn_closed_loop_canary_v2"
    ASYNC_OPTIMIZATION = False
    TRACE_SUFFIX = ".snn_trace.npz"
    TRAJECTORY_PID_USED = True
    NOMINAL_PLAN_SOURCE = "fixed_anchored_position_velocity_feedback"

    def _load_controller_config(self, config_path: str):
        config = super()._load_controller_config(config_path)
        return replace(
            config,
            # SNN inference is synchronous and comfortably faster than one
            # control tick. Schedule it from the discrete 50 Hz/25 Hz rate
            # ratio; a wall-clock 40 ms gate can miss a boundary by a few
            # microseconds and degrade deterministically to every third tick.
            planner_minimum_interval_seconds=0.0,
            maximum_plan_age_seconds=0.08,
        )

    @staticmethod
    def _planning_due(
        *,
        macro_boundary: bool,
        planner_completed: bool,
        index: int,
    ) -> bool:
        del planner_completed, index
        return discrete_model_tick_due(macro_boundary=macro_boundary)

    def _build_controller(self, checkpoint: str, config, device: str):
        return StatefulSnnPolicyController.from_checkpoint(
            checkpoint, config=config, device=device
        )

    def _reference_horizon(self, index: int) -> ReferenceTrajectory:
        horizon = self.jepa_mppi.model_horizon_steps
        reference_rate_hz = self.jepa_mppi.teacher_rate_hz
        stride = max(1, int(round(self.command_rate / reference_rate_hz)))
        indices = np.minimum(
            index + np.arange(1, horizon + 1) * stride,
            len(self.command_arrays["body_twist_reference"]) - 1,
        )
        references = [self._trajectory_reference(int(item)) for item in indices]
        return ReferenceTrajectory(
            time_seconds=np.arange(1, horizon + 1, dtype=np.float64)
            / reference_rate_hz,
            position_ned_m=np.stack(
                [reference.position_preview_ned_m for reference in references]
            ).astype(np.float32),
            linear_velocity_body_m_s=np.stack(
                [reference.linear_velocity_body_frd_m_s for reference in references]
            ).astype(np.float32),
            yaw_rad=np.asarray(
                [reference.yaw_preview_rad for reference in references], dtype=np.float32
            ),
            yaw_rate_rad_s=np.asarray(
                [reference.yaw_rate_rad_s for reference in references], dtype=np.float32
            ),
        )

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        super()._annotate_rollout_provenance(episode_path)
        metadata = load_metadata(episode_path)
        controller = metadata.setdefault("controller_provenance", {})
        controller.update(
            {
                "controller_id": self.CONTROLLER_ID,
                "controller_type": self.CONTROLLER_TYPE,
                "snn_checkpoint": self.stage2_checkpoint,
                "sensor_state_rate_hz": self.jepa_mppi.sample_rate_hz,
                "reference_preview_rate_hz": self.jepa_mppi.teacher_rate_hz,
                "stateful_across_control_updates": True,
                "future_reference_serialized_through_sensor_state": False,
                "fixed_safety_projection": "axis_limits_plus_six_motor_0p90_envelope",
                "correction_execution_fraction": 1.0,
                "correction_limit": self.jepa_mppi.correction_limit.tolist(),
                "correction_gain": self.jepa_mppi.correction_gain.tolist(),
                "nominal_plan_source": "fixed_anchored_position_velocity_feedback",
                "feedback_config": self.jepa_mppi.feedback_config.as_dict(),
                "trajectory_pid_used": True,
            }
        )
        write_metadata(episode_path, metadata)


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = SnnPolicyRollout()
    try:
        report = node.run()
        node.get_logger().info(
            f"SNN POLICY CLOSED LOOP COMPLETE: execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
