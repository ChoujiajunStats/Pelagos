from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import logging
import math
import time

from .protocol import FRAME_SIZE


LOG = logging.getLogger(__name__)


def _bcd4(value: int) -> tuple[int, int]:
    value = max(0, min(9999, value))
    digits = f"{value:04d}"
    return int(digits[:2], 16), int(digits[2:], 16)


def make_main_frame(elapsed: float = 0.0) -> bytes:
    frame = bytearray(32)
    frame[0] = 0xAA
    heading = round((elapsed * 3) % 360)
    frame[1:3] = heading.to_bytes(2, "big")
    pitch = round(4 * math.sin(elapsed / 3))
    roll = round(7 * math.sin(elapsed / 4))
    frame[3:5] = bytes((0x10 if pitch < 0 else 0x01, abs(pitch)))
    frame[5:7] = bytes((0x10 if roll < 0 else 0x01, abs(roll)))
    depth_cm = round(125 + 15 * math.sin(elapsed / 5))
    frame[7:10] = depth_cm.to_bytes(3, "big")
    frame[10] = 18
    frame[11] = 1
    for offset, value in zip((12, 15, 18), (15, 5, 1000)):
        high, low = _bcd4(value)
        frame[offset : offset + 3] = bytes((0x01, high, low))
    for offset, value in zip((21, 24, 27), (25, 10, 5)):
        high, low = _bcd4(value)
        frame[offset : offset + 3] = bytes((0x00, high, low))
    frame[30] = 87
    frame[31] = 0xFF
    return bytes(frame)


def make_status_frame(elapsed: float = 0.0, control: bytes | None = None) -> bytes:
    frame = bytearray(29)
    frame[0] = 0xAA
    if control is not None and len(control) == FRAME_SIZE and control[-1] in (0xAA, 0xBB):
        if control[-1] == 0xAA:
            frame[1] = control[18]
            frame[2] = control[10]
            frame[3] = control[11]
            frame[23] = control[9]
            frame[24] = control[12]
            frame[25] = control[13]
            frame[26] = control[14]
            frame[27] = control[15]
        else:
            frame[1] = control[21]
            frame[23] = control[17]
            frame[25] = control[18]
            frame[26] = control[19]
            frame[27] = control[20]
            frame[24] = 1
    else:
        frame[24] = 1
    frame[4:10] = bytes((20, 2, 8, 30, 3, 9))
    frame[20:23] = round(elapsed).to_bytes(3, "big")
    frame[28] = 0xEE
    return bytes(frame)


@dataclass
class SimulatorStats:
    clients: int = 0
    received_heartbeats: int = 0
    received_commands: int = 0


class MockRovServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 18887) -> None:
        self.host = host
        self.port = port
        self.stats = SimulatorStats()
        self._server: asyncio.Server | None = None
        self._started_at = time.monotonic()
        self.last_control: bytes | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        socket = self._server.sockets[0]
        self.port = int(socket.getsockname()[1])
        LOG.info("ROV simulator listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.stats.clients += 1
        sender = asyncio.create_task(self._send_telemetry(writer))
        try:
            while True:
                frame = await reader.readexactly(FRAME_SIZE)
                if frame[0] != 0xFF:
                    continue
                if frame[-1] == 0x11 and frame[1:23] == bytes(22):
                    self.stats.received_heartbeats += 1
                elif frame[-1] in (0xAA, 0xBB):
                    self.stats.received_commands += 1
                    self.last_control = frame
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            self.stats.clients -= 1

    async def _send_telemetry(self, writer: asyncio.StreamWriter) -> None:
        status_at = 0.0
        while True:
            elapsed = time.monotonic() - self._started_at
            writer.write(make_main_frame(elapsed))
            if elapsed >= status_at:
                writer.write(make_status_frame(elapsed, self.last_control))
                status_at = elapsed + 1.0
            await writer.drain()
            await asyncio.sleep(0.1)


async def _async_main(host: str, port: int) -> None:
    simulator = MockRovServer(host, port)
    await simulator.start()
    await simulator.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Haimuyun ROV protocol simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18887)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_async_main(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
