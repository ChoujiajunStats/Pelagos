"""Build the current JEPA manifest from a hard-reset batch report only."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from disaware_learning.data.episode_schema import (
    FORMAT_VERSION,
    GROUND_TRUTH_ARRAYS,
    PRIVILEGED_ACTUATOR_ARRAYS,
    PRIVILEGED_ACTUATOR_NATIVE_ARRAYS,
    PROTOCOL_ID,
)
from disaware_learning.data.episode_validation import inspect_episode
from disaware_learning.estimation.pressure_depth import PRESSURE_DEPTH_SOURCE_ID

MANIFEST_FORMAT_VERSION = 2
MANIFEST_PROTOCOL_ID = "motor_jepa_admitted_dataset_v2"
# Only the current hard-reset VINS collection is admissible for training.
# Older batches used a GT-derived VIO proxy and intentionally fail this gate.
BATCH_PROTOCOL_ID = "hard_reset_vins_domain_collection_batch_v3"
VALID_SPLITS = {"train", "validation", "test"}
ENTRY_PATH_FIELDS = ("admitted_episode_path", "episode_sidecar_path")


def _resolve_collection_file(value: object, root: Path) -> Path:
    """Resolve a copied collection file whose archived path used another root."""
    raw = Path(str(value)).expanduser()
    direct = raw if raw.is_absolute() else root / raw
    if direct.is_file():
        return direct.resolve()
    if raw.is_absolute():
        for index in reversed(
            [index for index, part in enumerate(raw.parts) if part == root.name]
        ):
            candidate = root.joinpath(*raw.parts[index + 1 :])
            if candidate.is_file():
                return candidate.resolve()
    return direct.resolve()


def _resolve_manifest_entry_path(
    value: object,
    *,
    manifest_path: Path,
    declared_root: Path | None,
) -> Path:
    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        return (manifest_path.parent / raw).resolve()
    if raw.is_file():
        return raw.resolve()
    if declared_root is not None:
        try:
            relative = raw.relative_to(declared_root)
        except ValueError:
            pass
        else:
            relocated = manifest_path.parent / relative
            if relocated.is_file():
                return relocated.resolve()
    return _resolve_collection_file(raw, manifest_path.parent)


def _partition(command_split: str, domain_split: str) -> str:
    if "test" in (command_split, domain_split):
        return "test"
    if "validation" in (command_split, domain_split):
        return "validation"
    return "train"


def _read_candidate(job: dict[str, Any], root: Path) -> dict[str, Any] | None:
    accepted = job.get("accepted_attempt")
    if job.get("status") != "passed" or not isinstance(accepted, dict):
        return None
    if accepted.get("pass") is not True:
        raise ValueError(f"accepted attempt is not passed: {job.get('job_id')}")
    report_path = _resolve_collection_file(accepted.get("report_path", ""), root)
    if not report_path.is_file():
        raise ValueError(f"missing rollout report for {job.get('job_id')}: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("execution_pass") is not True or report.get("episode_contract_pass") is not True:
        raise ValueError(f"rollout report is not admitted: {job.get('job_id')}")
    episode_path = _resolve_collection_file(report.get("episode_path", ""), root)
    if not episode_path.is_file():
        raise ValueError(f"missing episode for {job.get('job_id')}: {episode_path}")
    sidecar_path = episode_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise ValueError(f"missing episode sidecar: {sidecar_path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION or metadata.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"episode protocol mismatch: {episode_path}")
    if metadata.get("ground_truth_recording", {}).get("enabled") is not False:
        raise ValueError(f"ground-truth recording must be disabled: {episode_path}")
    if metadata.get("applied_thrust_recording", {}).get("enabled") is not False:
        raise ValueError(f"privileged actuator recording must be disabled: {episode_path}")
    depth_sensor = metadata.get("depth_sensor", {})
    controller_depth = metadata.get("controller_depth_sensor", {})
    if not (
        depth_sensor.get("source_id") == PRESSURE_DEPTH_SOURCE_ID
        and depth_sensor.get("derived_from_archive_array") == "pressure_pa"
        and depth_sensor.get("vio_position_z_used") is False
        and depth_sensor.get("ground_truth_position_z_used") is False
        and controller_depth.get("source_id") == PRESSURE_DEPTH_SOURCE_ID
        and controller_depth.get("replaces_vio_position_z_for_control") is True
        and controller_depth.get("ground_truth_position_z_used") is False
    ):
        raise ValueError(f"pressure-depth provenance is not deployable: {episode_path}")
    with np.load(episode_path, allow_pickle=False) as archive:
        forbidden = set(GROUND_TRUTH_ARRAYS) | set(PRIVILEGED_ACTUATOR_ARRAYS) | set(PRIVILEGED_ACTUATOR_NATIVE_ARRAYS)
        present = sorted(forbidden.intersection(archive.files))
        if present:
            raise ValueError(f"episode contains privileged arrays: {present}")
    inspection = inspect_episode(episode_path)
    if inspection.get("pass") is not True:
        raise ValueError(f"independent episode inspection failed: {episode_path}")

    command_split = str(job.get("command_split", "train"))
    domain_split = str(job.get("domain_split", ""))
    if command_split not in VALID_SPLITS or domain_split not in VALID_SPLITS:
        raise ValueError(f"invalid split on {job.get('job_id')}")
    expected_cell = f"{command_split}_command__{domain_split}_domain"
    if metadata.get("joint_split_cell") != expected_cell:
        raise ValueError(f"joint split mismatch on {job.get('job_id')}")
    if metadata.get("domain_split") != domain_split:
        raise ValueError(f"domain split mismatch on {job.get('job_id')}")
    return {
        "job_id": str(job["job_id"]),
        "reference_id": str(job["reference_id"]),
        "domain_id": str(job["domain_id"]),
        "command_split": command_split,
        "domain_split": domain_split,
        "joint_split_cell": expected_cell,
        "partition": _partition(command_split, domain_split),
        "admitted_episode_path": str(episode_path),
        "episode_sidecar_path": str(sidecar_path),
        "sample_count": int(metadata.get("sample_count", 0)),
        "duration_seconds": float(metadata.get("duration_seconds", 0.0)),
        "sample_rate_hz": float(metadata.get("actual_sample_rate_hz", 0.0)),
    }


def build_dataset_manifest(
    collection_root: str | Path,
    destination: str | Path,
    *,
    train_limit: int | None = None,
    evaluation_limit_per_cell: int | None = None,
    seed: int = 20260731,
    require_all_cells: bool = True,
    overwrite: bool = False,
) -> Path:
    """Freeze only passed jobs from the current ``batch_report.json`` protocol."""
    root = Path(collection_root).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    batch_path = root / "batch_report.json"
    if not batch_path.is_file():
        raise FileNotFoundError(f"missing current batch report under {root}")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if batch.get("protocol_id") != BATCH_PROTOCOL_ID:
        raise ValueError("batch report protocol mismatch")
    candidates: dict[str, list[dict[str, Any]]] = {}
    rejected = 0
    for job in batch.get("jobs", []):
        candidate = _read_candidate(job, root)
        if candidate is None:
            rejected += 1
        else:
            candidates.setdefault(candidate["domain_split"], []).append(candidate)
    required = {"train", "validation", "test"}
    missing = sorted(required - set(candidates))
    if require_all_cells and missing:
        raise ValueError("collection does not cover domain splits: " + ", ".join(missing))

    selected: list[dict[str, Any]] = []
    available_by_split: dict[str, int] = {}
    selected_by_split: dict[str, int] = {}
    for split, values in sorted(candidates.items()):
        values.sort(key=lambda item: item["job_id"])
        random.Random(f"{seed}:{split}").shuffle(values)
        available_by_split[split] = len(values)
        limit = train_limit if split == "train" else evaluation_limit_per_cell
        chosen = values if limit is None else values[:limit]
        selected.extend(chosen)
        selected_by_split[split] = len(chosen)
    selected.sort(key=lambda item: ({"train": 0, "validation": 1, "test": 2}[item["partition"]], item["job_id"]))
    if not selected:
        raise ValueError("no admitted episodes")
    for item in selected:
        for field in ENTRY_PATH_FIELDS:
            item[field] = os.path.relpath(item[field], output.parent)
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "protocol_id": MANIFEST_PROTOCOL_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "collection_root": str(root),
        "selection_seed": int(seed),
        "selection": {
            "train_limit": train_limit,
            "evaluation_limit_per_cell": evaluation_limit_per_cell,
            "policy": "seeded_shuffle_within_domain_split",
            "require_all_domain_splits": require_all_cells,
        },
        "collection_protocol_id": BATCH_PROTOCOL_ID,
        "split_contract": {
            "train": "train domain episodes",
            "validation": "validation domain episodes",
            "test": "test domain episodes",
        },
        "source_counts": {
            "passing_admitted_jobs": len(selected),
            "non_admitted_terminal_jobs": rejected,
            "available_by_domain_split": available_by_split,
            "missing_domain_splits": missing,
        },
        "selected_counts": {
            "total": len(selected),
            "train": sum(item["partition"] == "train" for item in selected),
            "validation": sum(item["partition"] == "validation" for item in selected),
            "test": sum(item["partition"] == "test" for item in selected),
            "by_domain_split": selected_by_split,
        },
        "entries": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION or manifest.get("protocol_id") != MANIFEST_PROTOCOL_ID:
        raise ValueError("JEPA dataset manifest protocol mismatch")
    if not manifest.get("entries"):
        raise ValueError("JEPA dataset manifest is empty")
    declared_value = manifest.get("collection_root")
    declared_root = (
        Path(str(declared_value)).expanduser()
        if isinstance(declared_value, str) and declared_value
        else None
    )
    for entry in manifest["entries"]:
        for field in ENTRY_PATH_FIELDS:
            entry[field] = str(
                _resolve_manifest_entry_path(
                    entry[field],
                    manifest_path=manifest_path,
                    declared_root=declared_root,
                )
            )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--evaluation-limit-per-cell", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--allow-incomplete-cells", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(argv)
    output = build_dataset_manifest(
        options.collection_root,
        options.output,
        train_limit=options.train_limit or None,
        evaluation_limit_per_cell=options.evaluation_limit_per_cell or None,
        seed=options.seed,
        require_all_cells=not options.allow_incomplete_cells,
        overwrite=options.overwrite,
    )
    print(json.dumps(load_dataset_manifest(output)["selected_counts"], indent=2, sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
