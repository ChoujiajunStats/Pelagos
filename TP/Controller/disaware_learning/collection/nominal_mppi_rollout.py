"""Fixed-nominal MPPI rollout on the JEPA-MPPI execution boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rclpy

from disaware_learning.collection.jepa_mppi_rollout import JepaMppiRollout
from disaware_learning.control.nominal_mppi import (
    DYNAMICS_MODEL_ID,
    IMPLEMENTATION_ID,
    ReferenceConditionedNominalMppi,
)
from disaware_learning.data.episode_io import load_metadata, write_metadata


class NominalMppiRollout(JepaMppiRollout):
    """Run MPPI with fixed calibrated dynamics and no learned model path."""

    CONTROLLER_ID = IMPLEMENTATION_ID
    CONTROLLER_TYPE = "nominal_model_mppi"
    PROTOCOL_ID = "nominal_underwater_mppi_closed_loop_v1"

    @staticmethod
    def _model_provenance() -> dict[str, Any]:
        return {
            "learned_model_used": False,
            "learned_weights_loaded": False,
            "checkpoint_model_state_dict_used": False,
            "stage2_checkpoint_role": "metadata_and_fixed_nominal_parameters_only",
            "dynamics_model": DYNAMICS_MODEL_ID,
            "candidate_search_and_cost": "identical_to_jepa_mppi",
        }

    def _build_controller(self, checkpoint: str, config: Any, device: str):
        return ReferenceConditionedNominalMppi.from_checkpoint_metadata(
            checkpoint,
            config=config,
            device=device,
        )

    def _annotate_rollout_provenance(self, episode_path: str) -> None:
        super()._annotate_rollout_provenance(episode_path)
        metadata = load_metadata(episode_path)
        metadata.setdefault("controller_provenance", {}).update(
            self._model_provenance()
        )
        write_metadata(episode_path, metadata)

    def run(self) -> dict[str, Any]:
        report = super().run()
        report.setdefault("controller_provenance", {}).update(
            self._model_provenance()
        )
        if self.report_path:
            target = Path(self.report_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return report


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = NominalMppiRollout()
    try:
        report = node.run()
        node.get_logger().info(
            "NOMINAL-MPPI CLOSED LOOP COMPLETE: "
            f"execution_pass={report['execution_pass']}"
        )
        return 0 if report["execution_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
