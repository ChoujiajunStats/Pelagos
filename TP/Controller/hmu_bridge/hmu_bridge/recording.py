from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, TextIO


class TelemetryRecorder:
    def __init__(self, data_dir: Path, enabled: bool = True) -> None:
        self.data_dir = data_dir
        self.enabled = enabled
        self._handle: TextIO | None = None
        self.path: Path | None = None

    def start(self) -> None:
        if not self.enabled or self._handle is not None:
            return
        session_dir = self.data_dir / "telemetry"
        session_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = session_dir / f"hmu-telemetry-{stamp}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def record(self, kind: str, raw: bytes, decoded: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.start()
        assert self._handle is not None
        row = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "raw_hex": raw.hex(),
            "decoded": decoded,
        }
        self._handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

