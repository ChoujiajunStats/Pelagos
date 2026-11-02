from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass(frozen=True)
class RovConfig:
    host: str = "192.168.1.7"
    port: int = 8887
    mode: str = "observe"
    command_protocol: str = "motor"
    motor_channels: tuple[str, ...] = ("L1", "R1", "L3", "R3", "L2", "R2")
    motor_signs: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    connect_timeout_seconds: float = 3.0
    reconnect_delay_seconds: float = 1.0
    heartbeat_period_seconds: float = 1.0
    command_rate_hz: float = 10.0

    def __post_init__(self) -> None:
        if self.mode not in {"observe", "control"}:
            raise ValueError("rov.mode must be 'observe' or 'control'")
        if self.command_protocol not in {"motor", "channel"}:
            raise ValueError("rov.command_protocol must be 'motor' or 'channel'")
        six_channels = {"L1", "L2", "L3", "R1", "R2", "R3"}
        if len(self.motor_channels) != 6 or set(self.motor_channels) != six_channels:
            raise ValueError("motor_channels must map the six installed thrusters exactly once")
        if len(self.motor_signs) != 6 or any(int(sign) not in (-1, 1) for sign in self.motor_signs):
            raise ValueError("motor_signs must contain six values of +1 or -1")
        if not self.host or not 1 <= self.port <= 65535:
            raise ValueError("invalid ROV address")
        if min(
            self.connect_timeout_seconds,
            self.reconnect_delay_seconds,
            self.heartbeat_period_seconds,
            self.command_rate_hz,
        ) <= 0:
            raise ValueError("ROV timing values must be positive")
        if self.heartbeat_period_seconds > 5:
            raise ValueError("heartbeat period cannot exceed the vendor's 5 second limit")


@dataclass(frozen=True)
class SafetyConfig:
    command_timeout_seconds: float = 0.3
    arm_hold_seconds: float = 1.5
    neutral_deadzone: float = 0.08

    def __post_init__(self) -> None:
        if self.command_timeout_seconds <= 0 or self.arm_hold_seconds <= 0:
            raise ValueError("safety timeouts must be positive")
        if not 0 <= self.neutral_deadzone <= 0.25:
            raise ValueError("neutral_deadzone must be between 0 and 0.25")


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path = Path("~/.local/share/hmu/data").expanduser()
    record_raw_frames: bool = True


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = ServerConfig()
    rov: RovConfig = RovConfig()
    safety: SafetyConfig = SafetyConfig()
    storage: StorageConfig = StorageConfig()


def load_config(path: str | Path | None = None) -> AppConfig:
    raw: dict = {}
    if path is not None:
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)

    server = ServerConfig(**raw.get("server", {}))
    rov_values = dict(raw.get("rov", {}))
    for key in ("motor_channels", "motor_signs"):
        if key in rov_values:
            rov_values[key] = tuple(rov_values[key])
    rov = RovConfig(**rov_values)
    safety = SafetyConfig(**raw.get("safety", {}))
    storage_values = dict(raw.get("storage", {}))
    if "data_dir" in storage_values:
        storage_values["data_dir"] = Path(storage_values["data_dir"]).expanduser()
    storage = StorageConfig(**storage_values)
    return AppConfig(server=server, rov=rov, safety=safety, storage=storage)
