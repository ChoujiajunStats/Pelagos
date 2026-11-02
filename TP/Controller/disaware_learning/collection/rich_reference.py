"""Slow independent multi-sine body-twist references for data collection.

The bank deliberately excites each body axis with low-frequency, randomized
sinusoids so the vehicle can establish its response before the reference
changes direction.  Every rollout uses this one protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from disaware_learning.control.contract import MANUAL_INTENT_ORDER

PROTOCOL_ID = "slow_multisine_body_twist_v5"
GEOMETRIC_BENCHMARK_PROTOCOL_ID = "underwater_geometric_trajectory_benchmark_v3"
TURNING_GEOMETRIC_BENCHMARK_PROTOCOL_ID = (
    "underwater_geometric_trajectory_benchmark_turning_v1"
)
GEOMETRIC_BENCHMARK_PROTOCOLS = frozenset(
    {
        GEOMETRIC_BENCHMARK_PROTOCOL_ID,
        TURNING_GEOMETRIC_BENCHMARK_PROTOCOL_ID,
    }
)
SUPPORTED_REFERENCE_PROTOCOLS = frozenset(
    {PROTOCOL_ID, *GEOMETRIC_BENCHMARK_PROTOCOLS}
)
MANIFEST_NAME = "rich_reference_manifest.json"
REPORT_NAME = "rich_reference_report.json"
INTENT_COUNT = 4
SAMPLE_RATE_HZ = 25.0


@dataclass(frozen=True)
class RichReferenceSpec:
    """Physical body-twist sampling settings for one reference bank."""

    sample_rate_hz: float = SAMPLE_RATE_HZ
    duration_seconds: float = 30.0
    warmup_seconds: float = 2.0
    cooldown_seconds: float = 2.0
    periods_seconds: tuple[tuple[float, ...], ...] = (
        (70.0, 35.0, 20.0),
        (65.0, 32.0, 18.0),
        (60.0, 30.0, 16.0),
        (50.0, 25.0, 12.0),
    )
    # v3 was a stale 50% scale and made every collected episode low-speed.
    # The full v2 envelope (v4) drove the rate-only collector into command
    # saturation on the nominal domains. v5 keeps 75% of that envelope,
    # giving 1.5x v3 coverage while retaining control headroom.
    amplitude_min: tuple[float, ...] = (0.06, 0.06, 0.0375, 0.06)
    amplitude_max: tuple[float, ...] = (0.2625, 0.2625, 0.1875, 0.2625)
    axis_limits: tuple[float, ...] = (0.50, 0.50, 0.35, 0.40)
    slew_rate_per_second: float = 1.20
    preview_start_position_ned_m: tuple[float, ...] = (0.0, 0.0, 3.0)
    workspace_xy_half_extent_m: float = 8.0
    workspace_depth_m: tuple[float, float] = (0.5, 5.5)

    def __post_init__(self) -> None:
        if self.sample_rate_hz < 2.0 or self.duration_seconds <= 2.0:
            raise ValueError("sample rate or duration is too small")
        if self.warmup_seconds < 0.0 or self.cooldown_seconds < 0.0:
            raise ValueError("warmup and cooldown must be nonnegative")
        if self.warmup_seconds + self.cooldown_seconds >= self.duration_seconds:
            raise ValueError("warmup and cooldown consume the reference")
        for name in (
            "periods_seconds",
            "amplitude_min",
            "amplitude_max",
            "axis_limits",
        ):
            values = getattr(self, name)
            if name == "periods_seconds":
                if len(values) != INTENT_COUNT or any(
                    len(item) != 3 for item in values
                ):
                    raise ValueError(
                        "periods_seconds must contain three periods per axis"
                    )
                if any(value <= 0.0 for item in values for value in item):
                    raise ValueError("periods must be positive")
                continue
            if (
                len(values) != INTENT_COUNT
                if name in {"amplitude_min", "amplitude_max", "axis_limits"}
                else len(values) != 3
            ):
                raise ValueError(f"{name} has an invalid width")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must be finite")
        if any(value <= 0.0 for value in self.amplitude_min):
            raise ValueError("amplitude_min must be positive")
        if any(
            lo > hi
            for lo, hi in zip(self.amplitude_min, self.amplitude_max, strict=True)
        ):
            raise ValueError("amplitude_min must not exceed amplitude_max")
        if any(value <= 0.0 or value > 1.0 for value in self.axis_limits):
            raise ValueError("axis limits must be in (0, 1]")
        if self.slew_rate_per_second <= 0.0:
            raise ValueError("slew rate must be positive")
        if len(self.preview_start_position_ned_m) != 3:
            raise ValueError("preview_start_position_ned_m must contain three values")
        if self.workspace_xy_half_extent_m <= 0.0:
            raise ValueError("workspace_xy_half_extent_m must be positive")
        if not 0.0 <= self.workspace_depth_m[0] < self.workspace_depth_m[1]:
            raise ValueError("workspace_depth_m must be an increasing positive range")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _time_values(spec: RichReferenceSpec) -> np.ndarray:
    count = int(round(spec.duration_seconds * spec.sample_rate_hz)) + 1
    values = np.arange(count, dtype=np.float64) / spec.sample_rate_hz
    values[-1] = spec.duration_seconds
    return values


def _fade(time_values: np.ndarray, spec: RichReferenceSpec) -> np.ndarray:
    envelope = np.ones_like(time_values)
    if spec.warmup_seconds:
        envelope *= np.clip(time_values / spec.warmup_seconds, 0.0, 1.0)
    if spec.cooldown_seconds:
        envelope *= np.clip(
            (spec.duration_seconds - time_values) / spec.cooldown_seconds,
            0.0,
            1.0,
        )
    return np.sin(0.5 * math.pi * envelope) ** 2


def _normalize(signal: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    active = envelope > 1e-3
    centered = signal - float(np.mean(signal[active]))
    scale = float(np.sqrt(np.mean(centered[active] ** 2)))
    return centered / max(scale, 1e-9)


def _sample_slow_multisine_axis(
    time_values: np.ndarray,
    envelope: np.ndarray,
    periods: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Sample a low-bandwidth independent multi-sine excitation.

    The periods intentionally match the slow excitation used for underwater
    system identification.  Random phases, small period jitter, and signed
    component weights make each bank member distinct without introducing
    frequencies faster than the vehicle can physically respond to.
    """
    signal = np.zeros_like(time_values)
    weights = rng.normal(0.0, 1.0, size=len(periods))
    weights /= max(float(np.linalg.norm(weights)), 1e-12)
    actual_periods: list[float] = []
    phases: list[float] = []
    for period, weight in zip(periods, weights, strict=True):
        period_jitter = float(rng.uniform(0.92, 1.08))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        actual_period = float(period * period_jitter)
        actual_periods.append(actual_period)
        phases.append(phase)
        signal += float(weight) * np.sin(
            2.0 * math.pi * time_values / actual_period + phase
        )
    return _normalize(signal, envelope), {
        "periods_seconds": actual_periods,
        "phases_rad": phases,
        "weights": weights.tolist(),
    }


def _slew(values: np.ndarray, spec: RichReferenceSpec) -> np.ndarray:
    output = np.zeros_like(values)
    maximum_step = spec.slew_rate_per_second / spec.sample_rate_hz
    for index in range(1, len(output)):
        output[index] = output[index - 1] + np.clip(
            values[index] - output[index - 1],
            -maximum_step,
            maximum_step,
        )
    # The fade should already reach zero. Make the endpoint exact only when
    # doing so respects the same slew contract as every other transition.
    if np.any(np.abs(output[-2]) > maximum_step + 1e-12):
        raise ValueError("cooldown is too short to reach a neutral endpoint")
    output[-1] = 0.0
    return output


def _integrate_body_twist(
    reference: np.ndarray,
    time_values: np.ndarray,
    start_position: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros((len(reference), 3), dtype=np.float64)
    yaw = np.zeros(len(reference), dtype=np.float64)
    position[0] = np.asarray(start_position, dtype=np.float64)
    dt = float(np.median(np.diff(time_values)))
    for index in range(1, len(reference)):
        yaw[index] = yaw[index - 1] + reference[index - 1, 3] * dt
        c = math.cos(yaw[index - 1])
        s = math.sin(yaw[index - 1])
        body_horizontal = reference[index - 1, :2]
        position[index, 0] = (
            position[index - 1, 0]
            + (c * body_horizontal[0] - s * body_horizontal[1]) * dt
        )
        position[index, 1] = (
            position[index - 1, 1]
            + (s * body_horizontal[0] + c * body_horizontal[1]) * dt
        )
        position[index, 2] = position[index - 1, 2] + reference[index - 1, 2] * dt
    return position, yaw


def _workspace_scale(
    position: np.ndarray,
    spec: RichReferenceSpec,
) -> float:
    scale = 1.0
    horizontal_peak = float(np.max(np.abs(position[:, :2])))
    if horizontal_peak > spec.workspace_xy_half_extent_m:
        scale = min(scale, spec.workspace_xy_half_extent_m / horizontal_peak)
    start_depth = float(spec.preview_start_position_ned_m[2])
    depth_span = max(
        abs(float(np.min(position[:, 2])) - start_depth),
        abs(float(np.max(position[:, 2])) - start_depth),
    )
    allowed_span = min(
        start_depth - spec.workspace_depth_m[0],
        spec.workspace_depth_m[1] - start_depth,
    )
    if depth_span > allowed_span:
        scale = min(scale, allowed_span / depth_span)
    return float(min(1.0, scale))


def generate_reference(
    seed: int,
    spec: RichReferenceSpec = RichReferenceSpec(),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate one independent four-axis body-twist reference."""
    time_values = _time_values(spec)
    envelope = _fade(time_values, spec)
    children = np.random.SeedSequence(int(seed)).spawn(INTENT_COUNT)
    reference = np.zeros((len(time_values), INTENT_COUNT), dtype=np.float64)
    amplitudes: dict[str, float] = {}
    components: dict[str, dict[str, list[float]]] = {}
    for axis, (name, child) in enumerate(
        zip(MANUAL_INTENT_ORDER, children, strict=True)
    ):
        rng = np.random.Generator(np.random.PCG64DXSM(child))
        signal, axis_components = _sample_slow_multisine_axis(
            time_values,
            envelope,
            spec.periods_seconds[axis],
            rng,
        )
        amplitude = float(
            rng.uniform(spec.amplitude_min[axis], spec.amplitude_max[axis])
        )
        amplitudes[name] = amplitude
        components[name] = axis_components
        reference[:, axis] = amplitude * np.tanh(signal) * envelope
    reference = _slew(reference, spec)
    position, yaw = _integrate_body_twist(
        reference,
        time_values,
        spec.preview_start_position_ned_m,
    )
    workspace_scale = _workspace_scale(position, spec)
    if workspace_scale < 1.0:
        reference *= workspace_scale
        position, yaw = _integrate_body_twist(
            reference,
            time_values,
            spec.preview_start_position_ned_m,
        )
    arrays = {
        "time_seconds": time_values,
        "body_twist_reference": reference,
        "position_preview_ned_m": position,
        "yaw_preview_rad": yaw,
    }
    metadata = {
        "seed": int(seed),
        "axis_order": list(MANUAL_INTENT_ORDER),
        "amplitudes": amplitudes,
        "components": components,
        "workspace_scale": workspace_scale,
        "body_twist_units": ["m/s", "m/s", "m/s", "rad/s"],
        "warmup_seconds": spec.warmup_seconds,
        "cooldown_seconds": spec.cooldown_seconds,
        "spec": spec.as_dict(),
    }
    return arrays, metadata


def load_rich_reference(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate one generated reference for a closed-loop collector."""
    source = Path(path).expanduser().resolve()
    if source.suffix != ".npz" or not source.is_file():
        raise ValueError(f"rich reference file is unavailable: {source}")
    metadata_path = source.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"rich reference metadata is unavailable: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    protocol_id = str(metadata.get("protocol_id", ""))
    if protocol_id not in SUPPORTED_REFERENCE_PROTOCOLS:
        raise ValueError(f"unsupported reference protocol: {protocol_id!r}")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key], dtype=np.float64) for key in archive.files
        }
    required = {
        "time_seconds",
        "body_twist_reference",
        "position_preview_ned_m",
        "yaw_preview_rad",
    }
    if not required.issubset(arrays):
        raise ValueError(
            f"rich reference is missing arrays: {sorted(required - set(arrays))}"
        )
    time_values = arrays["time_seconds"]
    reference = arrays["body_twist_reference"]
    forbidden = {"intent_normalized", "manual_control_normalized"} & set(arrays)
    if forbidden:
        raise ValueError(
            "physical references cannot contain normalized action arrays: "
            f"{sorted(forbidden)}"
        )
    position = arrays["position_preview_ned_m"]
    yaw = arrays["yaw_preview_rad"]
    if (
        time_values.ndim != 1
        or reference.shape != (time_values.size, 4)
        or position.shape != (time_values.size, 3)
        or yaw.shape != (time_values.size,)
        or time_values.size < 2
        or not np.all(np.isfinite(time_values))
        or not np.all(np.diff(time_values) > 0.0)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(yaw))
    ):
        raise ValueError("rich reference arrays have invalid shapes or values")
    return arrays, metadata


def rich_reference_provenance(
    path: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return collection metadata for a reference bank member."""
    source = Path(path).expanduser().resolve()
    duration = 0.0
    with np.load(source, allow_pickle=False) as archive:
        duration = float(np.asarray(archive["time_seconds"])[-1])
    warmup = float(metadata.get("warmup_seconds", 2.0))
    cooldown = float(metadata.get("cooldown_seconds", 2.0))
    reference_spec = metadata.get("spec", {})
    periods_seconds = reference_spec.get("periods_seconds")
    manifest_path = source.parent.parent / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = manifest.get("spec", {})
            warmup = float(spec.get("warmup_seconds", warmup))
            cooldown = float(spec.get("cooldown_seconds", cooldown))
            periods_seconds = spec.get("periods_seconds", periods_seconds)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    protocol_id = str(metadata.get("protocol_id", PROTOCOL_ID))
    if protocol_id not in SUPPORTED_REFERENCE_PROTOCOLS:
        raise ValueError(f"unsupported reference protocol: {protocol_id!r}")
    units = metadata.get(
        "body_twist_units",
        metadata.get("reference_units", {}).get(
            "body_twist", ["m/s", "m/s", "m/s", "rad/s"]
        ),
    )
    is_geometric = protocol_id in GEOMETRIC_BENCHMARK_PROTOCOLS
    return {
        "protocol_id": protocol_id,
        # References are intentionally reused across domain splits.  The
        # split of an admitted episode is therefore determined by its physics
        # domain, not by the reference id.
        "data_split": str(metadata.get("data_split", "train")),
        "training_bank_protocol_id": protocol_id,
        "duration_seconds": duration,
        "warmup_seconds": warmup,
        "cooldown_seconds": cooldown,
        "periods_seconds": periods_seconds,
        "command_path": str(source),
        "recipe": {
            "generator": protocol_id,
            "reference_id": metadata.get("reference_id", source.stem),
            "command_split": str(
                metadata.get("data_split", "test" if is_geometric else "train")
            ),
            "seed": metadata.get("seed"),
            "axis_order": metadata.get("axis_order", list(MANUAL_INTENT_ORDER)),
            "execution_reference": "body_twist_reference",
            "reference_units": units,
            "excitation": (
                "analytic_geometric_trajectory"
                if is_geometric
                else "independent_slow_multisine_axes"
            ),
            "components": metadata.get("components", {}),
        },
    }


def _checks(arrays: dict[str, np.ndarray], spec: RichReferenceSpec) -> dict[str, bool]:
    time_values = arrays["time_seconds"]
    reference = arrays["body_twist_reference"]
    position = arrays["position_preview_ned_m"]
    yaw = arrays["yaw_preview_rad"]
    checks = {
        "shape": bool(
            time_values.ndim == 1
            and reference.shape == (len(time_values), 4)
            and position.shape == (len(time_values), 3)
            and yaw.shape == (len(time_values),)
        ),
        "finite": bool(
            np.all(np.isfinite(time_values))
            and np.all(np.isfinite(reference))
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(yaw))
        ),
        "time_increasing": bool(np.all(np.diff(time_values) > 0.0)),
        "neutral_endpoints": bool(
            np.allclose(reference[0], 0.0) and np.allclose(reference[-1], 0.0)
        ),
        "axis_bounds": bool(
            np.all(np.abs(reference) <= np.asarray(spec.axis_limits) + 1e-9)
        ),
        "slew_bound": bool(
            np.max(np.abs(np.diff(reference, axis=0)))
            <= spec.slew_rate_per_second / spec.sample_rate_hz + 1e-9
        ),
        "workspace": _workspace_scale(position, spec) >= 1.0 - 1e-9,
    }
    return checks


def create_reference_bank(
    output_dir: str | Path,
    *,
    count: int = 64,
    seed: int = 20260804,
    spec: RichReferenceSpec = RichReferenceSpec(),
    data_split: str = "train",
    split_counts: tuple[int, int, int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Generate a directory-backed offline reference bank."""
    if count < 1:
        raise ValueError("count must be positive")
    if data_split not in {"train", "validation", "test"}:
        raise ValueError("data_split must be train, validation, or test")
    if split_counts is not None:
        if len(split_counts) != 3 or any(int(value) < 0 for value in split_counts):
            raise ValueError("split_counts must contain nonnegative train/validation/test counts")
        if sum(int(value) for value in split_counts) != count:
            raise ValueError("split_counts must sum to count")
        split_labels = (
            ["train"] * int(split_counts[0])
            + ["validation"] * int(split_counts[1])
            + ["test"] * int(split_counts[2])
        )
    else:
        split_labels = [data_split] * count
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"reference output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    references: list[dict[str, Any]] = []
    for index in range(count):
        reference_seed = int(seed + index)
        arrays, generation = generate_reference(reference_seed, spec)
        checks = _checks(arrays, spec)
        if not all(checks.values()):
            raise RuntimeError(f"reference {index} failed checks: {checks}")
        relative_path = Path("references") / f"reference_{index:04d}.npz"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        record = {
            "reference_id": f"rich_reference_{index:04d}",
            "index": index,
            "seed": reference_seed,
            "path": str(relative_path),
            "data_split": split_labels[index],
            "checks": checks,
            **generation,
        }
        path.with_suffix(".json").write_text(
            json.dumps({"protocol_id": PROTOCOL_ID, **record}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        references.append(record)
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "format_version": 1,
        "seed": int(seed),
        "count": count,
        "data_split": data_split if split_counts is None else "mixed",
        "split_counts": {
            "train": split_labels.count("train"),
            "validation": split_labels.count("validation"),
            "test": split_labels.count("test"),
        },
        "spec": spec.as_dict(),
        "references": references,
    }
    manifest = root / MANIFEST_NAME
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = audit_reference_bank(manifest)
    (root / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["pass"]:
        raise RuntimeError(f"generated reference bank failed audit: {report}")
    return manifest


def audit_reference_bank(path: str | Path) -> dict[str, Any]:
    """Audit all files in one offline reference bank."""
    source = Path(path).expanduser().resolve()
    manifest = source / MANIFEST_NAME if source.is_dir() else source
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("protocol_id") in GEOMETRIC_BENCHMARK_PROTOCOLS:
            references = payload["references"]
            all_checks = True
            root = manifest.parent
            rms: list[np.ndarray] = []
            for record in references:
                file_path = root / record["path"]
                try:
                    with np.load(file_path, allow_pickle=False) as archive:
                        arrays = {key: np.asarray(value, dtype=np.float64) for key, value in archive.items()}
                    time_values = arrays["time_seconds"]
                    body_twist = arrays["body_twist_reference"]
                    position = arrays["position_preview_ned_m"]
                    yaw = arrays["yaw_preview_rad"]
                    valid = bool(
                        time_values.ndim == 1
                        and body_twist.shape == (len(time_values), 4)
                        and position.shape == (len(time_values), 3)
                        and yaw.shape == (len(time_values),)
                        and len(time_values) >= 2
                        and np.all(np.isfinite(time_values))
                        and np.all(np.diff(time_values) > 0.0)
                        and np.all(np.isfinite(body_twist))
                        and np.all(np.isfinite(position))
                        and np.all(np.isfinite(yaw))
                    )
                    all_checks &= bool(file_path.is_file() and valid)
                    rms.append(np.sqrt(np.mean(body_twist**2, axis=0)))
                except (OSError, KeyError, ValueError, TypeError):
                    all_checks = False
            rms_array = np.asarray(rms, dtype=np.float64)
            count_valid = len(references) == int(payload["reference_count"])
            return {
                "pass": bool(all_checks and count_valid and len(references) > 0),
                "path": str(manifest),
                "protocol_id": payload.get("protocol_id"),
                "count": len(references),
                "checks": {
                    "protocol": True,
                    "reference_count": count_valid,
                    "files_and_constraints": bool(all_checks),
                    "geometric_contract": True,
                },
                "statistics": {
                    "axis_rms_min": np.min(rms_array, axis=0).tolist() if len(rms_array) else [],
                    "axis_rms_median": np.median(rms_array, axis=0).tolist() if len(rms_array) else [],
                    "axis_rms_max": np.max(rms_array, axis=0).tolist() if len(rms_array) else [],
                },
            }
        spec = RichReferenceSpec(**payload["spec"])
        references = payload["references"]
        root = manifest.parent
        all_checks = True
        component_metadata_valid = True
        rms: list[np.ndarray] = []
        positive_peaks: list[np.ndarray] = []
        negative_peaks: list[np.ndarray] = []
        workspace_scales: list[float] = []
        for record in references:
            file_path = root / record["path"]
            arrays = {
                key: np.asarray(value)
                for key, value in np.load(file_path, allow_pickle=False).items()
            }
            checks = _checks(arrays, spec)
            all_checks &= (
                file_path.is_file()
                and all(checks.values())
            )
            reference = arrays["body_twist_reference"]
            rms.append(np.sqrt(np.mean(reference**2, axis=0)))
            positive_peaks.append(np.max(reference, axis=0))
            negative_peaks.append(np.min(reference, axis=0))
            workspace_scales.append(float(record["workspace_scale"]))
            components = record.get("components", {})
            component_metadata_valid &= bool(
                set(components) == set(MANUAL_INTENT_ORDER)
                and all(
                    set(components[axis])
                    == {"periods_seconds", "phases_rad", "weights"}
                    and all(
                        len(components[axis][name]) == 3
                        for name in (
                            "periods_seconds",
                            "phases_rad",
                            "weights",
                        )
                    )
                    and all(
                        math.isfinite(float(value))
                        for name in (
                            "periods_seconds",
                            "phases_rad",
                            "weights",
                        )
                        for value in components[axis][name]
                    )
                    and all(
                        float(value) > 0.0
                        for value in components[axis]["periods_seconds"]
                    )
                    for axis in MANUAL_INTENT_ORDER
                )
            )
        rms_array = np.asarray(rms)
        positive_peak = np.max(np.asarray(positive_peaks), axis=0)
        negative_peak = np.min(np.asarray(negative_peaks), axis=0)
        bidirectional_coverage = bool(
            np.all(positive_peak > 0.01) and np.all(negative_peak < -0.01)
        )
        protocol_valid = payload.get("protocol_id") == PROTOCOL_ID
        count_valid = len(references) == int(payload["count"])
        return {
            "pass": bool(
                protocol_valid
                and count_valid
                and all_checks
                and component_metadata_valid
                and bidirectional_coverage
            ),
            "path": str(manifest),
            "protocol_id": payload.get("protocol_id"),
            "count": len(references),
            "checks": {
                "protocol": protocol_valid,
                "reference_count": count_valid,
                "files_and_constraints": bool(all_checks),
                "component_metadata": bool(component_metadata_valid),
                "bank_bidirectional_axis_coverage": bidirectional_coverage,
            },
            "statistics": {
                "axis_rms_min": np.min(rms_array, axis=0).tolist(),
                "axis_rms_median": np.median(rms_array, axis=0).tolist(),
                "axis_rms_max": np.max(rms_array, axis=0).tolist(),
                "axis_positive_peak": positive_peak.tolist(),
                "axis_negative_peak": negative_peak.tolist(),
                "workspace_scale_min": min(workspace_scales, default=1.0),
                "workspace_scale_max": max(workspace_scales, default=1.0),
            },
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {"pass": False, "path": str(manifest), "error": str(error)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--count", type=int, default=64)
    generate.add_argument("--seed", type=int, default=20260804)
    generate.add_argument("--duration", type=float, default=30.0)
    generate.add_argument(
        "--amplitude-scale",
        type=float,
        default=1.0,
        help="multiply the protocol amplitude envelope without changing its frequency band",
    )
    generate.add_argument("--data-split", choices=("train", "validation", "test"), default="train")
    generate.add_argument("--train-count", type=int)
    generate.add_argument("--validation-count", type=int)
    generate.add_argument("--test-count", type=int)
    generate.add_argument("--overwrite", action="store_true")
    audit = subparsers.add_parser("audit")
    audit.add_argument("path")
    parsed = parser.parse_args(argv)
    if parsed.command == "generate":
        if not math.isfinite(parsed.amplitude_scale) or parsed.amplitude_scale <= 0.0:
            raise SystemExit("--amplitude-scale must be positive and finite")
        defaults = RichReferenceSpec()
        spec = RichReferenceSpec(
            duration_seconds=parsed.duration,
            amplitude_min=tuple(
                value * parsed.amplitude_scale for value in defaults.amplitude_min
            ),
            amplitude_max=tuple(
                value * parsed.amplitude_scale for value in defaults.amplitude_max
            ),
        )
        split_counts = None
        if any(value is not None for value in (parsed.train_count, parsed.validation_count, parsed.test_count)):
            if None in (parsed.train_count, parsed.validation_count, parsed.test_count):
                raise SystemExit("--train-count, --validation-count, and --test-count must be provided together")
            split_counts = (parsed.train_count, parsed.validation_count, parsed.test_count)
        manifest = create_reference_bank(
            parsed.output_dir,
            count=parsed.count,
            seed=parsed.seed,
            spec=spec,
            data_split=parsed.data_split,
            split_counts=split_counts,
            overwrite=parsed.overwrite,
        )
        report = audit_reference_bank(manifest)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["pass"] else 1
    report = audit_reference_bank(parsed.path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
