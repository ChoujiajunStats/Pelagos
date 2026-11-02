from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import logging
import time
from typing import Callable

from .config import RovConfig
from .framing import MixedTelemetryFramer, ParsedFrame
from .protocol import ControlCommand, pack_channel_command, pack_heartbeat, pack_motor_command


LOG = logging.getLogger(__name__)


@dataclass
class TransportState:
    connected: bool = False
    connecting: bool = False
    last_error: str | None = None
    connected_at_monotonic: float | None = None
    received_frames: int = 0
    discarded_bytes: int = 0
    sent_command_frames: int = 0
    sent_heartbeats: int = 0


class RovClient:
    def __init__(
        self,
        config: RovConfig,
        on_frame: Callable[[ParsedFrame], None],
        command_provider: Callable[[], ControlCommand],
        on_connection_change: Callable[[bool, str | None], None] | None = None,
    ) -> None:
        self.config = config
        self.on_frame = on_frame
        self.command_provider = command_provider
        self.on_connection_change = on_connection_change
        self.state = TransportState()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="hmu-rov-client")

    async def stop(self) -> None:
        self._stop.set()
        if self.config.mode == "control":
            await self.send_neutral_now()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._close_writer()

    async def send_neutral_now(self) -> None:
        if self.config.mode != "control" or self._writer is None:
            return
        try:
            async with self._write_lock:
                self._writer.write(self._pack_command(ControlCommand.neutral()))
                await self._writer.drain()
                self.state.sent_command_frames += 1
        except (ConnectionError, OSError):
            pass

    async def _write(self, payload: bytes, *, heartbeat: bool = False) -> None:
        if self._writer is None:
            raise ConnectionError("ROV is not connected")
        async with self._write_lock:
            self._writer.write(payload)
            await self._writer.drain()
        if heartbeat:
            self.state.sent_heartbeats += 1
        else:
            self.state.sent_command_frames += 1

    def _pack_command(self, command: ControlCommand) -> bytes:
        if self.config.command_protocol == "motor":
            return pack_motor_command(command, self.config.motor_channels, self.config.motor_signs)
        return pack_channel_command(command)

    def snapshot(self) -> dict:
        result = asdict(self.state)
        connected_at = self.state.connected_at_monotonic
        result["connected_seconds"] = (
            None if connected_at is None or not self.state.connected else max(0.0, time.monotonic() - connected_at)
        )
        result["mode"] = self.config.mode
        result["command_protocol"] = self.config.command_protocol
        result["host"] = self.config.host
        result["port"] = self.config.port
        return result

    def _connection_changed(self, connected: bool, error: str | None = None) -> None:
        changed = self.state.connected != connected
        self.state.connected = connected
        self.state.connecting = False
        self.state.last_error = error
        self.state.connected_at_monotonic = time.monotonic() if connected else None
        if changed or error:
            LOG.info("ROV connection connected=%s error=%s", connected, error)
            if self.on_connection_change is not None:
                self.on_connection_change(connected, error)

    async def _close_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _run(self) -> None:
        while not self._stop.is_set():
            framer = MixedTelemetryFramer()
            try:
                self.state.connecting = True
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.config.host, self.config.port),
                    timeout=self.config.connect_timeout_seconds,
                )
                self._writer = writer
                self._connection_changed(True)
                await self._connected_loop(reader, framer)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, ConnectionError, OSError) as error:
                self._connection_changed(False, str(error) or type(error).__name__)
            except Exception as error:
                LOG.exception("Unexpected ROV transport error")
                self._connection_changed(False, f"{type(error).__name__}: {error}")
            finally:
                self.state.discarded_bytes += framer.discarded_bytes
                await self._close_writer()
                if self.state.connected:
                    self._connection_changed(False, "connection closed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.reconnect_delay_seconds)
            except TimeoutError:
                pass

    async def _connected_loop(self, reader: asyncio.StreamReader, framer: MixedTelemetryFramer) -> None:
        next_heartbeat = 0.0
        next_command = 0.0
        command_period = 1.0 / self.config.command_rate_hz
        while not self._stop.is_set():
            now = time.monotonic()
            if self.config.mode == "control":
                if now >= next_heartbeat:
                    await self._write(pack_heartbeat(), heartbeat=True)
                    next_heartbeat = now + self.config.heartbeat_period_seconds
                if now >= next_command:
                    await self._write(self._pack_command(self.command_provider()))
                    next_command = now + command_period

            read_timeout = 0.05 if self.config.mode == "control" else 0.5
            try:
                data = await asyncio.wait_for(reader.read(4096), timeout=read_timeout)
            except TimeoutError:
                continue
            if not data:
                raise ConnectionError("ROV closed the TCP connection")
            for frame in framer.feed(data):
                self.state.received_frames += 1
                self.on_frame(frame)
