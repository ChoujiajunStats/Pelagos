from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .protocol import MAIN_TELEMETRY_SIZE, STATUS_TELEMETRY_SIZE, plausible_status_frame


class FrameKind(str, Enum):
    MAIN = "main"
    STATUS = "status"


@dataclass(frozen=True)
class ParsedFrame:
    kind: FrameKind
    raw: bytes


class MixedTelemetryFramer:
    """Recover 29-byte and 32-byte telemetry frames from one TCP byte stream."""

    def __init__(self, max_buffer_size: int = 65536) -> None:
        self._buffer = bytearray()
        self.max_buffer_size = max_buffer_size
        self.discarded_bytes = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> list[ParsedFrame]:
        if not isinstance(data, bytes):
            raise TypeError("framer input must be bytes")
        self._buffer.extend(data)
        if len(self._buffer) > self.max_buffer_size:
            overflow = len(self._buffer) - self.max_buffer_size
            del self._buffer[:overflow]
            self.discarded_bytes += overflow

        frames: list[ParsedFrame] = []
        while True:
            try:
                start = self._buffer.index(0xAA)
            except ValueError:
                self.discarded_bytes += len(self._buffer)
                self._buffer.clear()
                break
            if start:
                del self._buffer[:start]
                self.discarded_bytes += start
            if len(self._buffer) < STATUS_TELEMETRY_SIZE:
                break

            status_candidate = self._buffer[:STATUS_TELEMETRY_SIZE]
            if plausible_status_frame(status_candidate):
                frames.append(ParsedFrame(FrameKind.STATUS, bytes(status_candidate)))
                del self._buffer[:STATUS_TELEMETRY_SIZE]
                continue

            if len(self._buffer) < MAIN_TELEMETRY_SIZE:
                break
            if self._buffer[MAIN_TELEMETRY_SIZE - 1] == 0xFF:
                frames.append(ParsedFrame(FrameKind.MAIN, bytes(self._buffer[:MAIN_TELEMETRY_SIZE])))
                del self._buffer[:MAIN_TELEMETRY_SIZE]
                continue

            del self._buffer[0]
            self.discarded_bytes += 1
        return frames

