"""Causal observation/action history for the latent deployment controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def _finite(values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return array.copy()


@dataclass(frozen=True)
class DeploymentHistory:
    """Fixed-width causal history with measured/executed six-motor actions.

    Each token is ``(observation_t, action_t)``.  ``action_t`` is the motor
    command that was executed during the transition into the next observation;
    callers should append a new observation with the action that produced it.
    """

    observation: np.ndarray
    action: np.ndarray


class CausalHistoryBuffer:
    """Maintain the exact window needed by Stage 2/JEPA-MPPI."""

    def __init__(self, history_steps: int = 26) -> None:
        if history_steps < 2:
            raise ValueError("history_steps must be at least two")
        self.history_steps = int(history_steps)
        self._observations: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self._actions: deque[np.ndarray] = deque(maxlen=self.history_steps)

    @property
    def ready(self) -> bool:
        return len(self._observations) == self.history_steps

    def __len__(self) -> int:
        return len(self._observations)

    def reset(self) -> None:
        self._observations.clear()
        self._actions.clear()

    def seed(self, observation: object, motor_action: object) -> None:
        """Fill the causal warmup window with the known neutral/initial action."""
        obs = _finite(observation, (16,), "observation")
        action = _finite(motor_action, (6,), "motor_action")
        self.reset()
        for _ in range(self.history_steps):
            self._observations.append(obs.copy())
            self._actions.append(action.copy())

    def append_transition(self, observation: object, executed_motor_action: object) -> None:
        """Append the observation after one executed six-motor transition."""
        obs = _finite(observation, (16,), "observation")
        action = _finite(executed_motor_action, (6,), "executed_motor_action")
        if not self._observations:
            raise RuntimeError("seed history before appending a transition")
        self._observations.append(obs)
        self._actions.append(action)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError(
                f"history is not ready: {len(self)}/{self.history_steps} observations"
            )
        return (
            np.stack(tuple(self._observations)).astype(np.float32),
            np.stack(tuple(self._actions)).astype(np.float32),
        )
