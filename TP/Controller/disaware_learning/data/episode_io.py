"""Structured I/O for paired episode NPZ and JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def episode_path(path: str | Path) -> Path:
    """Return the absolute NPZ path for an NPZ or metadata JSON input."""
    resolved = Path(path).expanduser().resolve()
    return resolved.with_suffix(".npz") if resolved.suffix == ".json" else resolved


def metadata_path(path: str | Path) -> Path:
    """Return the JSON metadata path paired with an episode input."""
    return episode_path(path).with_suffix(".json")


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load an episode without permitting pickled object arrays."""
    with np.load(episode_path(path), allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load the JSON metadata paired with an episode."""
    return json.loads(metadata_path(path).read_text(encoding="utf-8"))


def load_episode(
    path: str | Path,
) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    """Load an episode path, all arrays, and its metadata."""
    resolved = episode_path(path)
    return resolved, load_arrays(resolved), load_metadata(resolved)


def write_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Write deterministic, human-readable episode metadata."""
    destination = metadata_path(path)
    destination.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
