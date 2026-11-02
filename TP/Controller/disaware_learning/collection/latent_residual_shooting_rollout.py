"""Stonefish adapter for the deterministic Stage-2 residual shooting MPC."""

from __future__ import annotations

import rclpy
from disaware_learning.data.episode_io import load_metadata, write_metadata

from disaware_learning.collection.jepa_mppi_rollout import JepaMppiRollout
from disaware_learning.control.latent_residual_shooting_mpc import (
    CONTROLLER_TYPE,
    IMPLEMENTATION_ID,
    LatentResidualShootingConfig,
    LatentResidualShootingMpc,
    load_latent_residual_shooting_config,
)


class LatentResidualShootingRollout(JepaMppiRollout):
    """Reuse the causal VINS/IMU/depth and recorder boundary.

    The existing ``jepa_mppi_rollout.launch.py`` can select this executable via
    ``controller_executable:=latent_residual_shooting_rollout``.  Keeping the
    parameter name ``jepa_mppi_config`` makes the launch swap backward-simple;
    the file itself must contain ``implementation_id`` for this controller.
    """

    CONTROLLER_ID = IMPLEMENTATION_ID
    CONTROLLER_TYPE = CONTROLLER_TYPE
    PROTOCOL_ID = "latent_residual_shooting_mpc_closed_loop_v2_preview"
    CONFIG_PARAMETER = "jepa_mppi_config"
    DEVICE_PARAMETER = "jepa_mppi_device"
    TRACE_PARAMETER = "mppi_trace_path"
    CONFIG_METADATA_KEY = "latent_residual_shooting_config"
    DEVICE_METADATA_KEY = "latent_residual_shooting_device"
    TRACE_METADATA_KEY = "latent_residual_shooting_trace_path"
    TRACE_SUFFIX = ".latent_residual_shooting_trace.npz"
    # The paired 32-candidate GPU rollout is comfortably below the 5 Hz model
    # tick but longer than one 25 Hz actuator tick. Run it off the command loop
    # and hold the last verified command until the result is ready.
    ASYNC_OPTIMIZATION = True
    PRIME_CAUSAL_HISTORY = True

    def _load_controller_config(self, config_path: str):
        return (
            load_latent_residual_shooting_config(config_path)
            if config_path
            else LatentResidualShootingConfig()
        )

    def _build_controller(self, checkpoint: str, config, device: str):
        return LatentResidualShootingMpc.from_checkpoint(
            checkpoint,
            config=config,
            device=device,
        )

    def _reference_horizon(self, index: int):
        preview_ticks = int(
            round(
                self.jepa_config.planner_application_preview_seconds
                * self.command_rate
            )
        )
        maximum_index = len(self.command_arrays["body_twist_reference"]) - 1
        return super()._reference_horizon(min(index + preview_ticks, maximum_index))

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        super()._annotate_rollout_provenance(episode_path)
        metadata = load_metadata(episode_path)
        controller = metadata.setdefault("controller_provenance", {})
        controller.update(
            {
                "optimizer": "deterministic_batched_residual_shooting",
                "candidate_zero_is_exact_nominal_fallback": True,
                "paired_learned_nominal_rollout": "periodic_shadow_crosscheck",
                "candidate_sampling_centres": [
                    "shifted_previous_model_plan",
                    "reference_action_seed",
                    "per_axis_future_reference_lead_seeds",
                    "nominal_reference_blends",
                    "sobol_fast_hold_residuals",
                ],
                "planner_application_preview_seconds": (
                    self.jepa_config.planner_application_preview_seconds
                ),
                "ground_truth_feedback": False,
                "debug_chain": (
                    "input_contract/history/context/candidates/rollout/cost/"
                    "verification/timing"
                ),
            }
        )
        write_metadata(episode_path, metadata)


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = LatentResidualShootingRollout()
    try:
        report = node.run()
        node.get_logger().info(
            "LATENT RESIDUAL SHOOTING CLOSED LOOP COMPLETE: "
            f"execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
