from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import secrets
import time
from typing import Callable

from .config import SafetyConfig
from .protocol import ControlCommand


class SafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetySnapshot:
    has_control: bool
    owner: str | None
    armed: bool
    arm_pending: bool
    last_sequence: int
    last_command_age_seconds: float | None
    fault: str | None


class SafetyGate:
    def __init__(
        self,
        config: SafetyConfig = SafetyConfig(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._token: str | None = None
        self._owner: str | None = None
        self._armed = False
        self._arm_started_at: float | None = None
        self._last_update_at: float | None = None
        self._last_sequence = -1
        self._command = ControlCommand.neutral()
        self._fault: str | None = None

    def acquire(self, owner: str) -> str:
        if self._token is not None and self._owner != owner:
            raise SafetyError("control is already held by another client")
        if self._token is None:
            self._token = secrets.token_urlsafe(24)
            self._owner = owner
            self._reset_output("control acquired; arm required")
        return self._token

    def _require_token(self, token: str) -> None:
        if not self._token or not secrets.compare_digest(token, self._token):
            raise SafetyError("invalid or expired control token")

    def begin_arm(self, token: str) -> None:
        self._require_token(token)
        if self._last_update_at is None or self._clock() - self._last_update_at > self.config.command_timeout_seconds:
            raise SafetyError("fresh joystick data is required before arming")
        axes = (self._command.surge, self._command.yaw, self._command.sway, self._command.heave)
        if any(abs(value) > self.config.neutral_deadzone for value in axes):
            raise SafetyError("all motion axes must be neutral before arming")
        self._arm_started_at = self._clock()
        self._fault = None

    def commit_arm(self, token: str) -> None:
        self._require_token(token)
        now = self._clock()
        if self._arm_started_at is None:
            raise SafetyError("arm hold was not started")
        if now - self._arm_started_at < self.config.arm_hold_seconds:
            self._arm_started_at = None
            raise SafetyError("arm hold was too short")
        if self._last_update_at is None or now - self._last_update_at > self.config.command_timeout_seconds:
            self._arm_started_at = None
            raise SafetyError("joystick data expired during arm hold")
        axes = (self._command.surge, self._command.yaw, self._command.sway, self._command.heave)
        if any(abs(value) > self.config.neutral_deadzone for value in axes):
            self._arm_started_at = None
            raise SafetyError("an axis moved during arm hold")
        self._armed = True
        self._arm_started_at = None
        self._fault = None

    def cancel_arm(self, token: str) -> None:
        self._require_token(token)
        self._arm_started_at = None

    def update(self, token: str, sequence: int, command: ControlCommand) -> None:
        self._require_token(token)
        if not isinstance(sequence, int) or sequence <= self._last_sequence:
            raise SafetyError("control sequence is stale")
        self._last_sequence = sequence
        self._last_update_at = self._clock()
        self._command = replace(command, armed=self._armed)

    def disarm(self, token: str | None = None, reason: str = "operator disarm") -> None:
        if token is not None:
            self._require_token(token)
        self._reset_output(reason)

    def release(self, token: str | None = None, reason: str = "control client disconnected") -> None:
        if token is not None:
            self._require_token(token)
        self._reset_output(reason)
        self._token = None
        self._owner = None
        self._last_sequence = -1

    def fault(self, reason: str) -> None:
        self._reset_output(reason)

    def _reset_output(self, reason: str) -> None:
        self._armed = False
        self._arm_started_at = None
        self._command = ControlCommand.neutral()
        self._last_update_at = None
        self._fault = reason

    def output(self) -> ControlCommand:
        if self._armed:
            if self._last_update_at is None or self._clock() - self._last_update_at > self.config.command_timeout_seconds:
                self._reset_output("control timeout")
        if not self._armed:
            return ControlCommand.neutral()
        return replace(self._command, armed=True)

    def owns(self, token: str | None) -> bool:
        return bool(token and self._token and secrets.compare_digest(token, self._token))

    def snapshot(self) -> dict:
        now = self._clock()
        age = None if self._last_update_at is None else max(0.0, now - self._last_update_at)
        snapshot = SafetySnapshot(
            has_control=self._token is not None,
            owner=self._owner,
            armed=self._armed,
            arm_pending=self._arm_started_at is not None,
            last_sequence=self._last_sequence,
            last_command_age_seconds=age,
            fault=self._fault,
        )
        return asdict(snapshot)
