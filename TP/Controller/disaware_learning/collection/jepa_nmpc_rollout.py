"""Stonefish rollout adapter for differentiable JEPA-NMPC."""

from __future__ import annotations

from disaware_learning.data.episode_io import load_metadata, write_metadata
import rclpy

from disaware_learning.collection.jepa_mppi_rollout import JepaMppiRollout
from disaware_learning.control.jepa_nmpc import (
    IMPLEMENTATION_ID,
    JepaNmpcConfig,
    JepaNmpc,
    load_jepa_nmpc_config,
)


class JepaNmpcRollout(JepaMppiRollout):
    """Reuse the causal VIO, recorder, and domain boundary for JEPA-NMPC."""

    CONTROLLER_ID = IMPLEMENTATION_ID
    CONTROLLER_TYPE = "differentiable_jepa_nmpc"
    PROTOCOL_ID = "differentiable_jepa_nmpc_closed_loop_v1"
    ASYNC_OPTIMIZATION = True
    CONFIG_PARAMETER = "jepa_nmpc_config"
    DEVICE_PARAMETER = "jepa_nmpc_device"
    TRACE_PARAMETER = "nmpc_trace_path"
    CONFIG_METADATA_KEY = "jepa_nmpc_config"
    DEVICE_METADATA_KEY = "jepa_nmpc_device"
    TRACE_METADATA_KEY = "nmpc_trace_path"
    TRACE_SUFFIX = ".nmpc_trace.npz"

    def _load_controller_config(self, config_path: str):
        return load_jepa_nmpc_config(config_path) if config_path else JepaNmpcConfig()

    def _build_controller(self, checkpoint: str, config, device: str):
        return JepaNmpc.from_checkpoint(checkpoint, config=config, device=device)

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        super()._annotate_rollout_provenance(episode_path)
        metadata = load_metadata(episode_path)
        controller = metadata.setdefault("controller_provenance", {})
        controller.pop("candidate_sampling_centres", None)
        controller.update(
            {
                "optimizer": "deterministic_slsqp_exact_model_gradient",
                "learned_residual_enabled": (self.jepa_config.include_learned_residual),
                "nominal_plan_source": "shifted_previous_model_plan",
                "optimization_seed_source": "projected_reference_action_seed",
                "position_outer_loop_enabled": True,
                "preview_geometry_used_for_feedback": True,
                "yaw_outer_loop_enabled": True,
            }
        )
        write_metadata(episode_path, metadata)


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = JepaNmpcRollout()
    try:
        report = node.run()
        node.get_logger().info(
            "JEPA-NMPC CLOSED LOOP COMPLETE: "
            f"execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
