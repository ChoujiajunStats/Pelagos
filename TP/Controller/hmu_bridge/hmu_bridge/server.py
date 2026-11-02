from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.resources import files
import json
import logging
from pathlib import Path
import uuid
from typing import Any

from aiohttp import WSMsgType, web

from . import __version__
from .config import AppConfig, load_config
from .framing import FrameKind, ParsedFrame
from .protocol import ControlCommand, decode_main_telemetry, decode_status_telemetry
from .recording import TelemetryRecorder
from .safety import SafetyError, SafetyGate
from .transport import RovClient


LOG = logging.getLogger(__name__)
RUNTIME_KEY: web.AppKey["BridgeRuntime"] = web.AppKey("runtime", object)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assignment(name: str, value: Any) -> str:
    if isinstance(value, bool):
        encoded = "true" if value else "false"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        encoded = str(value)
    else:
        encoded = json.dumps(str(value), ensure_ascii=True)
    return f"{name}={encoded}"


class BridgeRuntime:
    def __init__(self, config: AppConfig, start_rov: bool = True) -> None:
        self.config = config
        self.start_rov = start_rov
        self.safety = SafetyGate(config.safety)
        self.recorder = TelemetryRecorder(config.storage.data_dir, config.storage.record_raw_frames)
        self.main_telemetry: dict[str, Any] | None = None
        self.vehicle_status: dict[str, Any] | None = None
        self.last_frame_at: str | None = None
        self.last_status_at: str | None = None
        self.telemetry_clients: set[web.WebSocketResponse] = set()
        self.control_clients: set[web.WebSocketResponse] = set()
        self._broadcast_lock = asyncio.Lock()
        self.rov = RovClient(
            config.rov,
            on_frame=self._on_frame,
            command_provider=self.safety.output,
            on_connection_change=self._on_connection_change,
        )

    async def start(self) -> None:
        self.config.storage.data_dir.mkdir(parents=True, exist_ok=True)
        if self.start_rov:
            self.rov.start()

    async def stop(self) -> None:
        await self.rov.stop()
        for ws in tuple(self.telemetry_clients | self.control_clients):
            await ws.close(code=1001, message=b"bridge stopping")
        self.recorder.close()

    def _on_frame(self, parsed: ParsedFrame) -> None:
        try:
            if parsed.kind == FrameKind.MAIN:
                decoded = decode_main_telemetry(parsed.raw).to_dict()
                self.main_telemetry = decoded
            else:
                decoded = decode_status_telemetry(parsed.raw).to_dict()
                self.vehicle_status = decoded
                self.last_status_at = _utc_now()
            self.last_frame_at = _utc_now()
            self.recorder.record(parsed.kind.value, parsed.raw, decoded)
            asyncio.create_task(self.broadcast())
        except ValueError as error:
            LOG.warning("Dropped invalid %s telemetry: %s", parsed.kind.value, error)

    def _on_connection_change(self, connected: bool, error: str | None) -> None:
        if not connected:
            self.vehicle_status = None
            self.last_status_at = None
            self.safety.fault("ROV disconnected")
        asyncio.create_task(self.broadcast())

    def state(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "time": _utc_now(),
            "rov": self.rov.snapshot(),
            "control": self.safety.snapshot(),
            "main": self.main_telemetry,
            "vehicle_status": self.vehicle_status,
            "last_frame_at": self.last_frame_at,
            "last_status_at": self.last_status_at,
            "recording_path": str(self.recorder.path) if self.recorder.path else None,
        }

    def datalake_values(self) -> dict[str, Any]:
        main = self.main_telemetry or {}
        status = self.vehicle_status or {}
        control = self.safety.snapshot()
        values: dict[str, Any] = {
            "HMU ROV Connected": self.rov.state.connected,
            "HMU Bridge Mode": self.config.rov.mode,
            "HMU Control Armed": control["armed"],
            "HMU Control Fault": control["fault"] or "none",
        }
        mapping = {
            "heading_deg": "HMU Heading Deg",
            "pitch_deg": "HMU Pitch Deg",
            "roll_deg": "HMU Roll Deg",
            "depth_m": "HMU Depth M",
            "water_temperature_c": "HMU Water Temperature C",
            "leak_detected": "HMU Leak Detected",
            "battery_percent": "HMU Battery Percent",
            "acceleration_x_g": "HMU Acceleration X G",
            "acceleration_y_g": "HMU Acceleration Y G",
            "acceleration_z_g": "HMU Acceleration Z G",
            "angular_rate_x_dps": "HMU Angular Rate X DPS",
            "angular_rate_y_dps": "HMU Angular Rate Y DPS",
            "angular_rate_z_dps": "HMU Angular Rate Z DPS",
        }
        for key, name in mapping.items():
            if main.get(key) is not None:
                values[name] = main[key]
        status_mapping = {
            "armed": "HMU Vehicle Armed",
            "heading_hold": "HMU Heading Hold",
            "depth_hold": "HMU Depth Hold",
            "runtime_seconds": "HMU Runtime Seconds",
            "light": "HMU Light Percent",
            "gear": "HMU Gear",
            "gripper_open": "HMU Gripper Open",
        }
        for key, name in status_mapping.items():
            if key in status:
                values[name] = status[key]
        return values

    async def broadcast(self) -> None:
        async with self._broadcast_lock:
            assignments = [_assignment(name, value) for name, value in self.datalake_values().items()]
            dead_telemetry: list[web.WebSocketResponse] = []
            for ws in tuple(self.telemetry_clients):
                try:
                    for assignment in assignments:
                        await ws.send_str(assignment)
                except (ConnectionError, RuntimeError):
                    dead_telemetry.append(ws)
            for ws in dead_telemetry:
                self.telemetry_clients.discard(ws)

            payload = {"type": "state", "state": self.state()}
            dead_control: list[web.WebSocketResponse] = []
            for ws in tuple(self.control_clients):
                try:
                    await ws.send_json(payload)
                except (ConnectionError, RuntimeError):
                    dead_control.append(ws)
            for ws in dead_control:
                self.control_clients.discard(ws)

    async def neutral_now(self) -> None:
        await self.rov.send_neutral_now()
        await self.broadcast()


@web.middleware
async def cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def _runtime(request: web.Request) -> BridgeRuntime:
    return request.app[RUNTIME_KEY]


async def health(request: web.Request) -> web.Response:
    runtime = _runtime(request)
    status = "ok" if runtime.rov.state.connected else "degraded"
    return web.json_response({"status": status, **runtime.state()})


async def api_state(request: web.Request) -> web.Response:
    return web.json_response(_runtime(request).state())


def _control_command(data: dict[str, Any]) -> ControlCommand:
    axes = data.get("axes", {})
    return ControlCommand(
        surge=axes.get("surge", 0),
        yaw=axes.get("yaw", 0),
        sway=axes.get("sway", 0),
        heave=axes.get("heave", 0),
        light=data.get("light", 0),
        heading_hold=bool(data.get("heading_hold", False)),
        depth_hold=bool(data.get("depth_hold", False)),
        gear=data.get("gear", 1),
        gimbal_1=data.get("gimbal_1", 0),
        gimbal_2=data.get("gimbal_2", 0),
        gripper=data.get("gripper", 0),
    )


async def _perform_action(runtime: BridgeRuntime, owner: str, data: dict[str, Any]) -> dict[str, Any]:
    action = data.get("type") or data.get("action")
    token = str(data.get("token", ""))
    if action == "takeover":
        if runtime.config.rov.mode != "control":
            raise SafetyError("bridge is in observe mode; control output is locked")
        return {"type": "takeover", "token": runtime.safety.acquire(owner)}
    if action == "control":
        runtime.safety.update(token, data.get("sequence"), _control_command(data))
    elif action == "arm_begin":
        runtime.safety.begin_arm(token)
    elif action == "arm_commit":
        runtime.safety.commit_arm(token)
    elif action == "arm_cancel":
        runtime.safety.cancel_arm(token)
    elif action == "disarm":
        runtime.safety.disarm(token)
        await runtime.neutral_now()
    elif action == "release":
        runtime.safety.release(token)
        await runtime.neutral_now()
    else:
        raise SafetyError(f"unknown control action: {action}")
    return {"type": "ack", "action": action, "state": runtime.safety.snapshot()}


async def api_control(request: web.Request) -> web.Response:
    runtime = _runtime(request)
    try:
        data = await request.json()
        result = await _perform_action(runtime, f"http:{request.remote}", data)
        return web.json_response(result)
    except (json.JSONDecodeError, TypeError, ValueError, SafetyError) as error:
        return web.json_response({"error": str(error)}, status=409)


async def ws_telemetry(request: web.Request) -> web.WebSocketResponse:
    runtime = _runtime(request)
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    runtime.telemetry_clients.add(ws)
    await runtime.broadcast()
    try:
        async for message in ws:
            if message.type == WSMsgType.ERROR:
                break
    finally:
        runtime.telemetry_clients.discard(ws)
    return ws


async def ws_control(request: web.Request) -> web.WebSocketResponse:
    runtime = _runtime(request)
    ws = web.WebSocketResponse(heartbeat=10, max_msg_size=16 * 1024, compress=False)
    await ws.prepare(request)
    runtime.control_clients.add(ws)
    owner = f"ws:{uuid.uuid4()}"
    owned_token: str | None = None
    await ws.send_json({"type": "state", "state": runtime.state()})
    try:
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                if message.type == WSMsgType.ERROR:
                    break
                continue
            data: Any = None
            try:
                data = json.loads(message.data)
                result = await _perform_action(runtime, owner, data)
                if result.get("type") == "takeover":
                    owned_token = result["token"]
                await ws.send_json(result)
            except (json.JSONDecodeError, TypeError, ValueError, SafetyError) as error:
                await ws.send_json(
                    {
                        "type": "error",
                        "action": data.get("type") if isinstance(data, dict) else None,
                        "message": str(error),
                    }
                )
    finally:
        runtime.control_clients.discard(ws)
        LOG.info("Control websocket closed: code=%s error=%r", ws.close_code, ws.exception())
        if runtime.safety.owns(owned_token):
            runtime.safety.release(owned_token)
            await runtime.neutral_now()
    return ws


async def cockpit_extras(request: web.Request) -> web.Response:
    runtime = _runtime(request)
    host = f"{runtime.config.server.host}:{runtime.config.server.port}"
    return web.json_response(
        {
            "name": "Haimuyun ROV Bridge",
            "telemetry_websocket": f"ws://{host}/ws/telemetry",
            "control_widget": f"http://{host}/widget/",
            "video": "rtsp://192.168.1.168/",
            "mode": runtime.config.rov.mode,
        }
    )


async def widget_redirect(request: web.Request) -> web.StreamResponse:
    raise web.HTTPFound("/widget/")


async def widget_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(files("hmu_bridge").joinpath("static", "index.html"))


async def on_startup(app: web.Application) -> None:
    await app[RUNTIME_KEY].start()


async def on_shutdown(app: web.Application) -> None:
    # Close Cockpit WebSockets before aiohttp waits for request handlers.
    # on_cleanup runs after that wait and can stall the experiment handoff.
    await app[RUNTIME_KEY].stop()


def create_app(config: AppConfig | None = None, *, start_rov: bool = True) -> web.Application:
    config = config or load_config()
    app = web.Application(middlewares=[cors_middleware])
    app[RUNTIME_KEY] = BridgeRuntime(config, start_rov=start_rov)
    static_dir = files("hmu_bridge").joinpath("static")
    app.router.add_get("/health", health)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/control", api_control)
    app.router.add_get("/ws/telemetry", ws_telemetry)
    app.router.add_get("/ws/control", ws_control)
    app.router.add_get("/cockpit-extras.json", cockpit_extras)
    app.router.add_get("/", widget_redirect)
    app.router.add_get("/widget/", widget_index)
    app.router.add_static("/widget/", static_dir, show_index=True, append_version=True)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Haimuyun ROV to Cockpit bridge")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    web.run_app(create_app(config), host=config.server.host, port=config.server.port, access_log=LOG)


if __name__ == "__main__":
    main()
