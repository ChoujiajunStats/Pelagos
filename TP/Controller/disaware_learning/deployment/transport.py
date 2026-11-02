"""Minimal TCP transport for the external ROV protocol.

This module intentionally does not decode sensor units. It provides framing
and the fail-closed command boundary needed by the deployment loop; the VIO
adapter can consume validated raw sensor frames after unit calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import time

from disaware_learning.deployment.protocol import FRAME_SIZE, pack_heartbeat, pack_motor_command


SENSOR_FRAME_SIZE = 32
SENSOR_HEADER = 0xAA
SENSOR_TAIL = 0xFF


@dataclass(frozen=True)
class RovTcpConfig:
    host: str = "192.168.1.7"
    port: int = 8887
    connect_timeout_seconds: float = 5.0
    io_timeout_seconds: float = 0.25
    heartbeat_period_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("ROV TCP host is required")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("ROV TCP port is invalid")
        if self.connect_timeout_seconds <= 0.0 or self.io_timeout_seconds <= 0.0:
            raise ValueError("TCP timeouts must be positive")
        if self.heartbeat_period_seconds <= 0.0 or self.heartbeat_period_seconds > 5.0:
            raise ValueError("heartbeat period must be in (0, 5] seconds")


class RovTcpTransport:
    """Synchronous, fail-closed transport for one ROV TCP connection."""

    def __init__(self, config: RovTcpConfig = RovTcpConfig()) -> None:
        self.config = config
        self._socket: socket.socket | None = None
        self._receive_buffer = bytearray()
        self._last_heartbeat_at = 0.0

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self.connected:
            return
        connection = socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.connect_timeout_seconds,
        )
        connection.settimeout(self.config.io_timeout_seconds)
        self._socket = connection
        self._receive_buffer.clear()
        self._last_heartbeat_at = 0.0
        self.send_heartbeat(force=True)

    def close(self) -> None:
        connection, self._socket = self._socket, None
        self._receive_buffer.clear()
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def __enter__(self) -> "RovTcpTransport":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.emergency_stop()
        self.close()

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("ROV TCP transport is not connected")
        return self._socket

    def send_heartbeat(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_heartbeat_at < self.config.heartbeat_period_seconds:
            return False
        self._require_socket().sendall(pack_heartbeat())
        self._last_heartbeat_at = now
        return True

    def send_motor_frame(self, frame: bytes) -> None:
        if not isinstance(frame, bytes) or len(frame) != FRAME_SIZE:
            raise ValueError("motor frame must be exactly 24 bytes")
        if frame[0] != 0xFF or frame[-1] != 0xBB:
            raise ValueError("motor frame has invalid header or tail")
        self._require_socket().sendall(frame)
        self.send_heartbeat()

    def send_neutral(self) -> None:
        self.send_motor_frame(pack_motor_command([1500.0] * 6, arm=0))

    def emergency_stop(self) -> None:
        """Best-effort neutral/disarm output; safe while disconnected."""
        if not self.connected:
            return
        try:
            self.send_neutral()
        except (OSError, RuntimeError):
            self.close()

    def _extract_sensor_frame(self) -> bytes | None:
        while True:
            try:
                start = self._receive_buffer.index(SENSOR_HEADER)
            except ValueError:
                self._receive_buffer.clear()
                return None
            if start:
                del self._receive_buffer[:start]
            if len(self._receive_buffer) < SENSOR_FRAME_SIZE:
                return None
            candidate = bytes(self._receive_buffer[:SENSOR_FRAME_SIZE])
            if candidate[-1] != SENSOR_TAIL:
                del self._receive_buffer[0]
                continue
            del self._receive_buffer[:SENSOR_FRAME_SIZE]
            return candidate

    def receive_sensor_frame(self, *, timeout_seconds: float | None = None) -> bytes | None:
        """Receive the next complete 32-byte sensor frame, or ``None`` on timeout."""
        connection = self._require_socket()
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, timeout_seconds)
        )
        while True:
            frame = self._extract_sensor_frame()
            if frame is not None:
                return frame
            if deadline is not None and time.monotonic() >= deadline:
                return None
            try:
                chunk = connection.recv(4096)
            except socket.timeout:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue
            if not chunk:
                self.close()
                raise ConnectionError("ROV TCP peer closed the connection")
            self._receive_buffer.extend(chunk)
