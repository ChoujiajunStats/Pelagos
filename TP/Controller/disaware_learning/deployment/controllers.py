"""Controller adapters used by the real-vehicle deployment loop.

Both adapters expose the same four-axis command result while preserving the
native controller result in ``raw_result``.  Transport, VIO, and reference
generation stay outside this module so a controller cannot silently invent
state or bypass the trained observation contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from disaware_learning.control.jepa_mppi import (
    JepaMppiConfig,
    JepaMppiResult,
    ReferenceConditionedJepaMppi,
    ReferenceTrajectory,
)
from disaware_learning.control.nominal_nmpc import (
    NominalNmpc,
    NominalNmpcConfig,
    NmpcReference,
    NmpcResult,
    load_nominal_nmpc_config,
)
from disaware_learning.estimation.vio_state import VioState


ControllerName = Literal["nmpc", "latent"]
COMMAND_WIDTH = 4
COMMAND_ORDER = ("surge", "sway", "heave_up", "yaw")


def _command(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (COMMAND_WIDTH,) or not np.all(np.isfinite(result)):
        raise ValueError("controller command must be four finite axes")
    return result.copy()


@dataclass(frozen=True)
class DeploymentCommand:
    """Common output contract for either real-vehicle controller."""

    controller: ControllerName
    command: np.ndarray
    accepted: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    raw_result: NmpcResult | JepaMppiResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _command(self.command))
        if self.controller not in ("nmpc", "latent"):
            raise ValueError(f"unsupported controller: {self.controller}")


class NmpcDeploymentController:
    """Adapter for the nominal six-thruster NMPC model."""

    name: ControllerName = "nmpc"

    def __init__(
        self,
        config: NominalNmpcConfig | None = None,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise ValueError("provide config or config_path, not both")
        if config_path is not None:
            config = load_nominal_nmpc_config(config_path)
        self.controller = NominalNmpc(config or NominalNmpcConfig())

    def reset(self, command: object | None = None) -> None:
        self.controller.reset(None if command is None else _command(command))

    def update(
        self,
        state: VioState,
        reference: NmpcReference,
        *,
        previous_command: object | None = None,
    ) -> DeploymentCommand:
        result = self.controller.optimize(
            state,
            reference,
            previous_command=(
                None
                if previous_command is None
                else _command(previous_command).astype(np.float64)
            ),
        )
        return DeploymentCommand(
            controller=self.name,
            command=result.command,
            accepted=result.accepted,
            reason=result.reason,
            diagnostics=result.diagnostics,
            raw_result=result,
        )


class LatentDeploymentController:
    """Adapter for the frozen-Stage1, latent-response JEPA-MPPI model."""

    name: ControllerName = "latent"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config: JepaMppiConfig = JepaMppiConfig(),
        device: str = "cpu",
    ) -> None:
        self.controller = ReferenceConditionedJepaMppi.from_checkpoint(
            checkpoint,
            config=config,
            device=device,
        )

    @property
    def history_steps(self) -> int:
        return self.controller.history_steps

    def reset(self, command: object | None = None) -> None:
        self.controller.reset(None if command is None else _command(command))

    def update(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        reference: ReferenceTrajectory,
        *,
        previous_command: object,
        current_position_ned_m: object | None = None,
        roll_command: float = 0.0,
    ) -> DeploymentCommand:
        result = self.controller.optimize(
            np.asarray(observation_history, dtype=np.float32),
            np.asarray(action_history, dtype=np.float32),
            reference,
            _command(previous_command),
            current_position_ned_m=(
                None
                if current_position_ned_m is None
                else np.asarray(current_position_ned_m, dtype=np.float32)
            ),
            roll_command=roll_command,
        )
        return DeploymentCommand(
            controller=self.name,
            command=result.command,
            accepted=result.accepted,
            reason=result.reason,
            diagnostics=result.diagnostics,
            raw_result=result,
        )


def build_controller(
    name: ControllerName,
    *,
    nmpc_config: NominalNmpcConfig | None = None,
    nmpc_config_path: str | Path | None = None,
    latent_checkpoint: str | Path | None = None,
    latent_config: JepaMppiConfig = JepaMppiConfig(),
    device: str = "cpu",
) -> NmpcDeploymentController | LatentDeploymentController:
    """Construct one of the two deployment controllers explicitly."""
    if name == "nmpc":
        return NmpcDeploymentController(
            nmpc_config,
            config_path=nmpc_config_path,
        )
    if name == "latent":
        if latent_checkpoint is None:
            raise ValueError("latent_checkpoint is required for the latent controller")
        return LatentDeploymentController(
            latent_checkpoint,
            config=latent_config,
            device=device,
        )
    raise ValueError(f"unsupported controller: {name}")
