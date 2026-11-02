"""Read-only MAVLink audit state for the ArduSub execution contract."""

from __future__ import annotations

import math
import os
import time

from disaware_learning.control.contract import ARDUSUB_NOMINAL_PARAMETERS

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402


class ArduSubRuntimeContract:
    """Track heartbeat and parameter readback without changing vehicle state."""

    def __init__(self, endpoint: str, source_system: int = 252) -> None:
        self.link = mavutil.mavlink_connection(
            endpoint,
            source_system=int(source_system),
            source_component=193,
            dialect="ardupilotmega",
        )
        self.target_system: int | None = None
        self.target_component: int | None = None
        self.parameters: dict[str, float] = {}
        self.last_parameter_request = -math.inf
        self.last_heartbeat_received = -math.inf
        self.last_mode = "UNKNOWN"
        self.last_armed = False
        self.last_system_status: int | None = None
        self.checks = 0
        self.failure: str | None = None

    def poll(self) -> None:
        for _ in range(500):
            message = self.link.recv_match(blocking=False)
            if message is None:
                break
            message_type = message.get_type()
            if message_type == "BAD_DATA":
                continue
            if message_type == "HEARTBEAT":
                self._handle_heartbeat(message)
            elif message_type == "PARAM_VALUE":
                self._handle_parameter(message)

        now = time.monotonic()
        if (
            self.target_system is not None
            and len(self.parameters) < len(ARDUSUB_NOMINAL_PARAMETERS)
            and now - self.last_parameter_request >= 5.0
        ):
            self.link.mav.param_request_list_send(
                self.target_system,
                self.target_component or 0,
            )
            self.last_parameter_request = now

    def _handle_heartbeat(self, message) -> None:
        if int(message.autopilot) == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
            return
        source_system = int(message.get_srcSystem())
        source_component = int(message.get_srcComponent())
        if self.target_system is None:
            self.target_system = source_system
            self.target_component = source_component
            self.link.target_system = source_system
            self.link.target_component = source_component
        if source_system != self.target_system:
            return
        self.last_heartbeat_received = time.monotonic()
        self.last_mode = mavutil.mode_string_v10(message)
        self.last_armed = bool(
            message.base_mode
            & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        self.last_system_status = int(message.system_status)

    def _handle_parameter(self, message) -> None:
        if (
            self.target_system is None
            or int(message.get_srcSystem()) != self.target_system
        ):
            return
        name = message.param_id
        if isinstance(name, bytes):
            name = name.decode("ascii", errors="ignore")
        name = str(name).rstrip("\x00")
        if name in ARDUSUB_NOMINAL_PARAMETERS:
            self.parameters[name] = float(message.param_value)

    def execution_gate(
        self,
        *,
        expected_mode: str,
        maximum_heartbeat_age: float = 2.5,
    ) -> tuple[bool, str | None]:
        self.checks += 1
        missing = sorted(
            set(ARDUSUB_NOMINAL_PARAMETERS) - set(self.parameters)
        )
        if missing:
            return False, "ArduSub parameter readback is incomplete"
        mismatches = {
            name: value
            for name, value in self.parameters.items()
            if not math.isclose(
                value,
                float(ARDUSUB_NOMINAL_PARAMETERS[name]),
                abs_tol=1e-3,
            )
        }
        if mismatches:
            return False, f"ArduSub parameter mismatch: {mismatches}"
        if time.monotonic() - self.last_heartbeat_received > maximum_heartbeat_age:
            return False, "ArduSub heartbeat is stale"
        if not self.last_armed:
            return False, "ArduSub is not armed"
        if self.last_mode != expected_mode:
            return False, (
                f"ArduSub mode is {self.last_mode}, expected {expected_mode}"
            )
        if self.last_system_status != mavutil.mavlink.MAV_STATE_ACTIVE:
            return False, (
                "ArduSub system status is not ACTIVE: "
                f"{self.last_system_status}"
            )
        return True, None

    def mark_failure(self, reason: str) -> None:
        if self.failure is None:
            self.failure = reason

    def begin_episode(self) -> None:
        self.checks = 0
        self.failure = None

    def metadata(self, maximum_heartbeat_age: float = 2.5) -> dict:
        return {
            "parameter_snapshot": dict(self.parameters),
            "parameter_snapshot_source": "MAVLink PARAM_VALUE readback",
            "execution_mode": self.last_mode,
            "execution_mode_verified_from_heartbeat": bool(self.checks),
            "continuous_heartbeat_gate": {
                "checks": self.checks,
                "last_mode": self.last_mode,
                "last_armed": self.last_armed,
                "last_system_status": self.last_system_status,
                "maximum_heartbeat_age_seconds": maximum_heartbeat_age,
                "passed": self.failure is None and self.checks > 0,
                "failure": self.failure,
            },
        }

    def close(self) -> None:
        try:
            self.link.close()
        except Exception:
            pass
