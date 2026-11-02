"""Hard-reset concurrent collection over reference and physics-domain banks."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

from disaware_learning.collection.physics_domain import (
    DOMAIN_BANK_PROTOCOL_ID,
    DOMAIN_MANIFEST_NAME,
    inspect_domain_bank,
)
from disaware_learning.collection.rich_reference import (
    MANIFEST_NAME as REFERENCE_MANIFEST_NAME,
    PROTOCOL_ID as REFERENCE_PROTOCOL_ID,
    SUPPORTED_REFERENCE_PROTOCOLS,
    audit_reference_bank,
)
from disaware_learning.data.episode_validation import inspect_episode
from disaware_learning.data.episode_schema import FORMAT_VERSION, PROTOCOL_ID
from disaware_learning.estimation.pressure_depth import (
    PRESSURE_DEPTH_CONVERSION_ID,
    PRESSURE_DEPTH_SOURCE_ID,
)


BATCH_PROTOCOL_ID = "hard_reset_vins_domain_collection_batch_v3"
PLAN_NAME = "batch_plan.json"
REPORT_NAME = "batch_report.json"
ACTIVE_PROCESS_NAME = "active_process.json"
_CONTROLLER_IMPLEMENTATION_IDS = {
    "manual_rollout": "fixed_body_twist_velocity_rate_pid_v2",
    "nominal_nmpc_rollout": "nominal_bluerov2_nmpc_v1",
    "nominal_mppi_rollout": "fixed_nominal_underwater_mppi_v1",
    "jepa_mppi_rollout": (
        "context_hybrid_absolute_action_jepa_mppi_v54_projected_weighted_realtime"
    ),
    "snn_policy_rollout": "multirate_anchored_snn_correction_policy_v2",
}
_MPPI_CONTROLLER_EXECUTABLES = {
    "jepa_mppi_rollout",
    "snn_policy_rollout",
    "nominal_mppi_rollout",
}
_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_KNOWN_ROS_UNDERLAYS = (
    Path("/home/hong/vins_fusion_ros2_ws/install/camera_models"),
    Path("/home/hong/vins_fusion_ros2_ws/install/vins"),
    Path("/home/hong/stonefish_simulator/install/stonefish_ros2"),
    Path("/home/hong/stonefish_simulator/install/stonefish_bluerov2"),
    Path("/home/hong/stonefish_simulator/install/udg"),
    Path("/home/hong/stonefish_simulator/install/oceanbee"),
    Path("/home/hong/stonefish_simulator/install/harbor"),
)
_VINS_CONFIG_FILENAMES = (
    "vins_stonefish_stereo_imu.yaml",
    "vins_stonefish_cam0.yaml",
    "vins_stonefish_cam1.yaml",
)


def _augment_ros_pythonpath(environment: dict[str, str]) -> None:
    """Keep ros2cli importable even when the batch was started with PYTHONPATH=... ."""
    paths = [item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item]
    prefixes = [
        item for item in environment.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if item
    ]
    for variable in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
        prefixes.extend(
            item
            for item in environment.get(variable, "").split(os.pathsep)
            if item and item not in prefixes
        )
    for prefix in _KNOWN_ROS_UNDERLAYS:
        if prefix.is_dir() and str(prefix) not in prefixes:
            prefixes.append(str(prefix))
    if prefixes:
        environment["AMENT_PREFIX_PATH"] = os.pathsep.join(prefixes)
        library_paths = [
            item
            for item in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if item
        ]
        for prefix in prefixes:
            library = str(Path(prefix) / "lib")
            if Path(library).is_dir() and library not in library_paths:
                library_paths.append(library)
        if library_paths:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    candidates = [
        "/opt/ros/jazzy/lib/python3.12/site-packages",
        str(Path(sys.executable).parent.parent / "lib/python3.12/site-packages"),
    ]
    candidates.extend(
        str(Path(prefix) / "lib/python3.12/site-packages") for prefix in prefixes
    )
    for candidate in candidates:
        if Path(candidate).is_dir() and candidate not in paths:
            paths.append(candidate)
    if paths:
        environment["PYTHONPATH"] = os.pathsep.join(paths)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class WorkerRuntime:
    worker_id: int
    ros_domain_id: int
    sitl_instance: int
    bind_port: int
    mavlink_port: int
    external_nav_port: int
    ahrs_port: int
    contract_port: int


def worker_runtime(worker_id: int, ros_domain_base: int = 100) -> WorkerRuntime:
    """Return the process and network namespace assigned to one worker."""
    if worker_id < 0:
        raise ValueError("worker_id must be nonnegative")
    if not 0 <= ros_domain_base <= 231 or ros_domain_base + worker_id > 231:
        raise ValueError("ROS_DOMAIN_ID must stay in [0, 231]")
    mavlink_base = 14550 + 10 * worker_id
    return WorkerRuntime(
        worker_id=worker_id,
        ros_domain_id=ros_domain_base + worker_id,
        sitl_instance=worker_id,
        # sim_vehicle.py offsets the JSON FDM ports by ten per SITL instance.
        bind_port=9002 + 10 * worker_id,
        mavlink_port=mavlink_base,
        external_nav_port=mavlink_base + 1,
        ahrs_port=mavlink_base + 2,
        contract_port=mavlink_base + 3,
    )


def _manifest_path(path: str | Path, filename: str) -> Path:
    source = Path(path).expanduser().resolve()
    return source / filename if source.is_dir() else source


def inspect_vins_scenario(path: str | Path) -> dict[str, Any]:
    """Verify that a materialized Stonefish scenario can drive stereo VINS."""
    source = Path(path).expanduser().resolve()
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        return {"pass": False, "path": str(source), "error": str(error), "checks": {}}

    sensors = {sensor.get("name"): sensor for sensor in root.findall(".//sensor")}
    robot_transform = root.find("./robot/world_transform")
    try:
        spawn_xyz = tuple(
            float(value) for value in str(robot_transform.get("xyz", "")).split()
        )
    except (AttributeError, TypeError, ValueError):
        spawn_xyz = ()
    spawn_depth = spawn_xyz[2] if len(spawn_xyz) == 3 else float("nan")

    def topic(sensor_name: str) -> str:
        sensor = sensors.get(sensor_name)
        publisher = None if sensor is None else sensor.find("./ros_publisher")
        return "" if publisher is None else str(publisher.get("topic", ""))

    def rate(sensor_name: str) -> float:
        sensor = sensors.get(sensor_name)
        try:
            return float("nan") if sensor is None else float(sensor.get("rate", "nan"))
        except ValueError:
            return float("nan")

    checks = {
        "stereo_left_camera_present": bool(
            sensors.get("stereo_left") is not None
            and sensors["stereo_left"].get("type") == "camera"
        ),
        "stereo_right_camera_present": bool(
            sensors.get("stereo_right") is not None
            and sensors["stereo_right"].get("type") == "camera"
        ),
        "stereo_left_topic": topic("stereo_left").endswith("/stereo/left"),
        "stereo_right_topic": topic("stereo_right").endswith("/stereo/right"),
        "stereo_rate_sufficient": bool(
            rate("stereo_left") >= 20.0 and rate("stereo_right") >= 20.0
        ),
        "imu_present": bool(
            sensors.get("imu_filter") is not None
            and sensors["imu_filter"].get("type") == "imu"
        ),
        "imu_topic": topic("imu_filter").endswith("/imu"),
        "imu_rate_sufficient": rate("imu_filter") >= 100.0,
        "pressure_sensor_present": bool(
            sensors.get("pressure") is not None
            and sensors["pressure"].get("type") == "pressure"
        ),
        "pressure_topic": topic("pressure").endswith("/pressure"),
        "pressure_rate_sufficient": rate("pressure") >= 20.0,
        "spawn_depth_available": bool(
            math.isfinite(spawn_depth) and 0.5 <= spawn_depth <= 20.0
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "path": str(source),
        "checks": checks,
        "topics": {
            "stereo_left": topic("stereo_left"),
            "stereo_right": topic("stereo_right"),
            "imu": topic("imu_filter"),
            "pressure": topic("pressure"),
        },
        "rates_hz": {
            "stereo_left": rate("stereo_left"),
            "stereo_right": rate("stereo_right"),
            "imu": rate("imu_filter"),
            "pressure": rate("pressure"),
        },
        "spawn_depth_down_m": spawn_depth,
    }


def build_job_plan(
    reference_manifest_path: str | Path,
    domain_manifest_path: str | Path,
    *,
    domain_count: int,
    references_per_domain: int = 1,
    balanced_reference_pairs: bool = False,
    pair_seed: int = 20260815,
    controller_executable: str = "manual_rollout",
    pid_config_path: str | Path | None = None,
    nmpc_config_path: str | Path | None = None,
    nmpc_update_steps: int = 8,
    stage2_checkpoint: str | Path | None = None,
    jepa_mppi_config_path: str | Path | None = None,
    jepa_mppi_device: str = "auto",
) -> dict[str, Any]:
    """Build a deterministic domain-major collection plan."""
    if domain_count < 1 or references_per_domain < 1:
        raise ValueError("domain and reference counts must be positive")
    executable = str(controller_executable).strip()
    if executable not in _CONTROLLER_IMPLEMENTATION_IDS:
        raise ValueError(f"unsupported controller executable: {executable}")
    if nmpc_update_steps < 1:
        raise ValueError("nmpc_update_steps must be positive")
    config_path = (
        Path(nmpc_config_path).expanduser().resolve()
        if nmpc_config_path
        else None
    )
    pid_path = (
        Path(pid_config_path).expanduser().resolve()
        if pid_config_path
        else None
    )
    checkpoint_path = (
        Path(stage2_checkpoint).expanduser().resolve()
        if stage2_checkpoint
        else None
    )
    mppi_config_path = (
        Path(jepa_mppi_config_path).expanduser().resolve()
        if jepa_mppi_config_path
        else None
    )
    mppi_device = str(jepa_mppi_device).strip().lower()
    if executable == "nominal_nmpc_rollout" and config_path is None:
        raise ValueError("nominal_nmpc_rollout requires nmpc_config_path")
    if executable in _MPPI_CONTROLLER_EXECUTABLES:
        if checkpoint_path is None:
            raise ValueError(f"{executable} requires stage2_checkpoint")
        if mppi_config_path is None:
            raise ValueError(f"{executable} requires jepa_mppi_config_path")
        if mppi_device not in ("auto", "cpu", "cuda"):
            raise ValueError("jepa_mppi_device must be auto, cpu, or cuda")
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(config_path)
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if mppi_config_path is not None and not mppi_config_path.is_file():
        raise FileNotFoundError(mppi_config_path)
    if pid_path is not None and not pid_path.is_file():
        raise FileNotFoundError(pid_path)
    controller = {
        "executable": executable,
        "implementation_id": _CONTROLLER_IMPLEMENTATION_IDS[executable],
        "pid_config_path": str(pid_path) if pid_path is not None else "",
        "nmpc_config_path": str(config_path) if config_path is not None else "",
        "nmpc_update_steps": nmpc_update_steps,
        "stage2_checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else ""
        ),
        "jepa_mppi_config_path": (
            str(mppi_config_path) if mppi_config_path is not None else ""
        ),
        "jepa_mppi_device": mppi_device,
    }
    reference_path = _manifest_path(reference_manifest_path, REFERENCE_MANIFEST_NAME)
    domain_path = _manifest_path(domain_manifest_path, DOMAIN_MANIFEST_NAME)
    reference_audit = audit_reference_bank(reference_path)
    domain_audit = inspect_domain_bank(domain_path)
    if not reference_audit.get("pass"):
        raise ValueError(f"reference bank failed audit: {reference_audit}")
    if not domain_audit.get("pass"):
        raise ValueError(f"domain bank failed audit: {domain_audit}")
    references_payload = _load_json(reference_path)
    domains_payload = _load_json(domain_path)
    if references_payload.get("protocol_id") not in SUPPORTED_REFERENCE_PROTOCOLS:
        raise ValueError("reference manifest protocol does not match the collector")
    if domains_payload.get("protocol_id") != DOMAIN_BANK_PROTOCOL_ID:
        raise ValueError("physics-domain manifest protocol does not match the collector")
    references = references_payload["references"]
    domains = domains_payload["domains"]
    if domain_count > len(domains):
        raise ValueError("domain_count exceeds the available domain bank")
    if not references:
        raise ValueError("reference bank is empty")
    if balanced_reference_pairs:
        assignment_indices, pairing_audit = _balanced_split_reference_pairs(
            domains[:domain_count],
            len(references),
            references_per_domain,
            seed=int(pair_seed),
        )
        pairing_rng = random.Random(int(pair_seed))
    else:
        assignment_indices = [
            [index % len(references) for index in range(references_per_domain)]
            for _ in range(domain_count)
        ]
        pairing_audit = None
    jobs: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(domains[:domain_count]):
        if "scenario_path" not in domain:
            raise ValueError(
                f"domain {domain['domain_id']} has no materialized Stonefish scenario"
            )
        scenario_path = (domain_path.parent / domain["scenario_path"]).resolve()
        vins_scenario_audit = inspect_vins_scenario(scenario_path)
        if not vins_scenario_audit["pass"]:
            failed = [
                name
                for name, passed in vins_scenario_audit.get("checks", {}).items()
                if not passed
            ]
            raise ValueError(
                f"domain {domain['domain_id']} is not stereo-VINS capable: "
                f"{', '.join(failed) or vins_scenario_audit.get('error', 'unknown error')}"
            )
        for reference_index in assignment_indices[domain_index]:
            # Reuse the same command references in every physics domain. This
            # makes domain generalization measurable instead of confounding a
            # domain change with a different trajectory.
            reference = references[reference_index]
            recommended_spawn_depth = float(
                vins_scenario_audit["spawn_depth_down_m"]
            )
            job_id = f"{domain['domain_id']}__{reference['reference_id']}"
            jobs.append(
                {
                    "job_id": job_id,
                    "domain_id": domain["domain_id"],
                    "domain_split": domain["data_split"],
                    "command_split": str(reference.get("data_split", "train")),
                    "spawn_depth": recommended_spawn_depth,
                    "scenario_path": str(scenario_path),
                    "vins_scenario_audit": vins_scenario_audit,
                    "reference_id": reference["reference_id"],
                    "reference_path": str(
                        (reference_path.parent / reference["path"]).resolve()
                    ),
                    "controller_executable": controller["executable"],
                    "controller_implementation_id": controller[
                        "implementation_id"
                    ],
                    "pid_config_path": controller["pid_config_path"],
                    "nmpc_config_path": controller["nmpc_config_path"],
                    "nmpc_update_steps": controller["nmpc_update_steps"],
                    "stage2_checkpoint": controller["stage2_checkpoint"],
                    "jepa_mppi_config_path": controller[
                        "jepa_mppi_config_path"
                    ],
                    "jepa_mppi_device": controller["jepa_mppi_device"],
                }
            )
    if balanced_reference_pairs:
        pairing_rng.shuffle(jobs)
    plan: dict[str, Any] = {
        "protocol_id": BATCH_PROTOCOL_ID,
        "reference_manifest_path": str(reference_path),
        "domain_manifest_path": str(domain_path),
        "domain_count": domain_count,
        "references_per_domain": references_per_domain,
        "reference_bank_count": len(references),
        "pairing": {
            "mode": (
                "split_sparse_balanced_bipartite_v1"
                if balanced_reference_pairs
                else "cyclic_prefix"
            ),
            "seed": int(pair_seed) if balanced_reference_pairs else None,
            "unique_pair_count": len(jobs),
            "audit": pairing_audit,
        },
        "controller": controller,
        "data_contract": {
            "episode_format_version": FORMAT_VERSION,
            "episode_protocol_id": PROTOCOL_ID,
            "record_ground_truth": False,
            "ground_truth_arrays_admitted": False,
            "simulation_truth_use": "separate_rmse_audit_trace_only",
            "controller_feedback": (
                "canonical VINS pose/velocity + ArduSub AHRS/IMU + "
                "Stonefish pressure sensor depth"
            ),
            "training_position_target": (
                "VINS north/east position + pressure-derived down position"
            ),
            "depth": {
                "source_id": PRESSURE_DEPTH_SOURCE_ID,
                "topic": "/bluerov2/pressure",
                "message_type": "sensor_msgs/msg/FluidPressure",
                "surface_pressure_pa": 0.0,
                "fluid_density_kg_m3": 1025.0,
                "conversion": PRESSURE_DEPTH_CONVERSION_ID,
                "replaces_vio_position_z_for_control": True,
                "ground_truth_position_z_used": False,
            },
        },
        "job_count": len(jobs),
        "jobs": jobs,
    }
    return plan


def _balanced_split_reference_pairs(
    domains: list[Mapping[str, Any]],
    reference_count: int,
    references_per_domain: int,
    *,
    seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Build a balanced simple domain-reference graph within every split.

    The reference pool is shared across domain splits, while graph construction
    is isolated by split.  Each domain has the requested degree, each pair is
    unique, and reference degrees within one split differ by at most one.
    """
    if reference_count < 1:
        raise ValueError("reference_count must be positive")
    if not 1 <= references_per_domain <= reference_count:
        raise ValueError(
            "balanced reference count per domain cannot exceed the reference bank"
        )

    assignments: list[list[int]] = [[] for _ in domains]
    split_indices: dict[str, list[int]] = {}
    for index, domain in enumerate(domains):
        split = str(domain.get("data_split", ""))
        if not split:
            raise ValueError("every physics domain must declare data_split")
        split_indices.setdefault(split, []).append(index)

    split_audits: dict[str, Any] = {}
    for split, indices in sorted(split_indices.items()):
        if len(indices) > reference_count:
            raise ValueError(
                "split-sparse pairing requires at least as many references as "
                f"domains in split {split}"
            )
        split_seed = random.Random(f"{seed}:{split}").randrange(2**64)
        rng = random.Random(split_seed)
        reference_order = list(range(reference_count))
        rng.shuffle(reference_order)
        seen_by_domain = {index: set() for index in indices}
        split_pair_count = len(indices) * references_per_domain

        # Enumerating slots before matching fixes every reference degree to
        # floor/ceil(total_slots / reference_count).  A fresh matching per
        # round then distributes those slots without duplicate pairs.
        slots = [
            reference_order[slot % reference_count]
            for slot in range(split_pair_count)
        ]
        for round_index in range(references_per_domain):
            round_references = slots[
                round_index * len(indices) : (round_index + 1) * len(indices)
            ]
            domain_order = indices.copy()
            rng.shuffle(domain_order)
            candidate_order: dict[int, list[int]] = {}
            for domain_index in domain_order:
                candidates = [
                    reference_index
                    for reference_index in round_references
                    if reference_index not in seen_by_domain[domain_index]
                ]
                rng.shuffle(candidates)
                candidate_order[domain_index] = candidates

            reference_owner: dict[int, int] = {}

            def augment(domain_index: int, visited: set[int]) -> bool:
                for reference_index in candidate_order[domain_index]:
                    if reference_index in visited:
                        continue
                    visited.add(reference_index)
                    owner = reference_owner.get(reference_index)
                    if owner is None or augment(owner, visited):
                        reference_owner[reference_index] = domain_index
                        return True
                return False

            if not all(augment(domain_index, set()) for domain_index in domain_order):
                raise RuntimeError(
                    f"could not construct unique balanced reference matching for {split}"
                )
            for reference_index, domain_index in reference_owner.items():
                assignments[domain_index].append(reference_index)
                seen_by_domain[domain_index].add(reference_index)

        reference_degrees = Counter(
            reference_index
            for domain_index in indices
            for reference_index in assignments[domain_index]
        )
        degrees = [reference_degrees[index] for index in range(reference_count)]
        pair_count = sum(degrees)
        unique_pair_count = len(
            {
                (domain_index, reference_index)
                for domain_index in indices
                for reference_index in assignments[domain_index]
            }
        )
        split_audits[split] = {
            "domain_count": len(indices),
            "pair_count": pair_count,
            "unique_pair_count": unique_pair_count,
            "domain_degree": references_per_domain,
            "reference_degree_min": min(degrees),
            "reference_degree_max": max(degrees),
            "references_used": sum(degree > 0 for degree in degrees),
        }
        if (
            pair_count != len(indices) * references_per_domain
            or unique_pair_count != pair_count
            or max(degrees) - min(degrees) > 1
            or any(len(assignments[index]) != references_per_domain for index in indices)
        ):
            raise AssertionError(f"invalid balanced pairing for split {split}")

    return assignments, {
        "split_isolation": True,
        "reference_pool_shared_across_splits": True,
        "domain_degree": references_per_domain,
        "splits": split_audits,
    }


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_process_identity(process_id: int) -> dict[str, Any] | None:
    """Read stable Linux process identity fields without invoking ps."""
    process_root = Path("/proc") / str(int(process_id))
    try:
        stat = (process_root / "stat").read_text(encoding="utf-8")
        command_line = tuple(
            os.fsdecode(value)
            for value in (process_root / "cmdline").read_bytes().split(b"\0")
            if value
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    closing_parenthesis = stat.rfind(")")
    fields = stat[closing_parenthesis + 2 :].split()
    if closing_parenthesis < 0 or len(fields) <= 19:
        return None
    try:
        return {
            "pid": int(process_id),
            "process_group_id": int(fields[2]),
            "start_time_ticks": int(fields[19]),
            "command_line": command_line,
        }
    except ValueError:
        return None


def _command_references_attempt(
    command_line: tuple[str, ...], attempt_dir: str | Path
) -> bool:
    attempt = str(Path(attempt_dir).expanduser().resolve())
    return any(attempt in argument for argument in command_line)


def _process_group_members(process_group_id: int) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        identity = _read_process_identity(int(entry.name))
        if identity and identity["process_group_id"] == process_group_id:
            members.append(identity)
    return members


def _marker_owns_process_group(marker: Mapping[str, Any]) -> bool:
    """Require both process identity and the exact attempt path before signaling."""
    try:
        process_id = int(marker["pid"])
        process_group_id = int(marker["process_group_id"])
        start_time_ticks = int(marker["start_time_ticks"])
        attempt_dir = Path(str(marker["attempt_dir"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError):
        return False
    if process_id <= 1 or process_group_id <= 1:
        return False
    identity = _read_process_identity(process_id)
    if identity is not None:
        return bool(
            identity["process_group_id"] == process_group_id
            and identity["start_time_ticks"] == start_time_ticks
            and _command_references_attempt(identity["command_line"], attempt_dir)
        )
    # The launch leader may have exited while descendants remain. In that case,
    # retain the same exact-path ownership check on a surviving group member.
    return any(
        _command_references_attempt(member["command_line"], attempt_dir)
        for member in _process_group_members(process_group_id)
    )


def terminate_process_group(
    process: subprocess.Popen[Any] | int,
    *,
    interrupt_grace_seconds: float = 4.0,
    terminate_grace_seconds: float = 3.0,
) -> dict[str, Any]:
    """Stop a launch process tree with SIGINT, SIGTERM, then SIGKILL."""
    process_object = process if isinstance(process, subprocess.Popen) else None
    process_group_id = process.pid if process_object is not None else int(process)

    def group_exists() -> bool:
        if process_object is not None:
            process_object.poll()
        else:
            try:
                os.waitpid(process_group_id, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass
        return _process_group_exists(process_group_id)

    stages: list[str] = []
    for name, signum, grace in (
        ("SIGINT", signal.SIGINT, interrupt_grace_seconds),
        ("SIGTERM", signal.SIGTERM, terminate_grace_seconds),
        ("SIGKILL", signal.SIGKILL, 1.0),
    ):
        if not group_exists():
            break
        try:
            os.killpg(process_group_id, signum)
            stages.append(name)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + max(0.0, grace)
        while time.monotonic() < deadline and group_exists():
            time.sleep(0.05)
    return {
        "signals_sent": stages,
        "process_group_exited": not group_exists(),
    }


class BatchProcessSupervisor:
    """Own launch process groups and recover only this batch's stale attempts."""

    def __init__(self, output_dir: str | Path, stop_event: threading.Event) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self._active: dict[int, tuple[subprocess.Popen[Any], Path, Path]] = {}

    def register(
        self,
        process: subprocess.Popen[Any],
        attempt_dir: Path,
        command: list[str],
    ) -> None:
        attempt = attempt_dir.resolve()
        identity: dict[str, Any] | None = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and process.poll() is None:
            candidate = _read_process_identity(process.pid)
            if candidate and _command_references_attempt(
                candidate["command_line"], attempt
            ):
                identity = candidate
                break
            time.sleep(0.01)
        if identity is None:
            raise RuntimeError("launch command line does not identify its attempt directory")
        marker_path = attempt / ACTIVE_PROCESS_NAME
        marker = {
            "active": True,
            "pid": process.pid,
            "process_group_id": identity["process_group_id"],
            "start_time_ticks": identity["start_time_ticks"],
            "attempt_dir": str(attempt),
            "command": command,
            "registered_at": datetime.now().astimezone().isoformat(),
        }
        _write_json_atomic(marker_path, marker)
        with self._lock:
            self._active[identity["process_group_id"]] = (
                process,
                attempt,
                marker_path,
            )

    def finish(
        self,
        process: subprocess.Popen[Any],
        cleanup: Mapping[str, Any],
        return_code: int,
    ) -> None:
        process_group_id = process.pid
        with self._lock:
            owned = self._active.pop(process_group_id, None)
        if owned is None:
            return
        _, attempt_dir, marker_path = owned
        try:
            marker = _load_json(marker_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            marker = {
                "pid": process.pid,
                "process_group_id": process_group_id,
                "attempt_dir": str(attempt_dir),
            }
        marker.update(
            {
                "active": False,
                "finished_at": datetime.now().astimezone().isoformat(),
                "return_code": int(return_code),
                "cleanup": dict(cleanup),
            }
        )
        _write_json_atomic(marker_path, marker)

    def interrupt_active(self) -> None:
        """Request prompt shutdown; worker finalizers complete escalation/reaping."""
        self.stop_event.set()
        with self._lock:
            process_groups = tuple(self._active)
        for process_group_id in process_groups:
            try:
                os.killpg(process_group_id, signal.SIGINT)
            except ProcessLookupError:
                pass

    def cleanup_stale_attempts(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        jobs_root = self.output_dir / "jobs"
        if not jobs_root.is_dir():
            return results
        for marker_path in sorted(jobs_root.glob(f"*/attempt_*/{ACTIVE_PROCESS_NAME}")):
            try:
                marker = _load_json(marker_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                results.append(
                    {"marker_path": str(marker_path), "error": str(error), "signaled": False}
                )
                continue
            if marker.get("active") is not True:
                continue
            try:
                process_group_id = int(marker["process_group_id"])
            except (KeyError, TypeError, ValueError):
                process_group_id = -1
            group_exists = process_group_id > 1 and _process_group_exists(
                process_group_id
            )
            ownership_verified = group_exists and _marker_owns_process_group(marker)
            cleanup = (
                terminate_process_group(process_group_id)
                if ownership_verified
                else {"signals_sent": [], "process_group_exited": not group_exists}
            )
            result = {
                "marker_path": str(marker_path),
                "process_group_id": process_group_id,
                "ownership_verified": ownership_verified,
                "signaled": bool(cleanup["signals_sent"]),
                **cleanup,
            }
            marker.update(
                {
                    "active": False,
                    "recovered_at": datetime.now().astimezone().isoformat(),
                    "startup_recovery": result,
                }
            )
            _write_json_atomic(marker_path, marker)
            results.append(result)
        return results


def validate_attempt(
    job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    episode_inspector: Callable[[str | Path], dict[str, Any]] = inspect_episode,
) -> dict[str, Any]:
    """Independently validate launch status, report identity, and v15 episode."""
    report_path = Path(str(attempt.get("report_path", ""))).expanduser()
    report: dict[str, Any] | None = None
    report_error: str | None = None
    try:
        report = _load_json(report_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        report_error = str(error)
    episode_path = Path(str(report.get("episode_path", ""))) if report else Path("")
    episode_report = (
        episode_inspector(episode_path)
        if report and str(report.get("episode_path", "")).strip() and episode_path.is_file()
        else {"pass": False, "error": "episode path is missing"}
    )
    physics = report.get("physics_domain") if report else None
    runtime = report.get("physics_domain_runtime") if report else None
    activation = runtime.get("current_activation") if isinstance(runtime, dict) else None
    recipe = report.get("command_recipe", {}) if report else {}
    command_path = Path(str(report.get("command_path", ""))).expanduser() if report else None
    checks = {
        "launch_return_code_zero": int(attempt.get("return_code", -1)) == 0,
        "launch_not_timed_out": attempt.get("timed_out") is False,
        "launch_process_group_exited": attempt.get("process_group_exited") is True,
        "report_readable": report is not None,
        "execution_pass": bool(report and report.get("execution_pass") is True),
        "episode_contract_pass": bool(
            report and report.get("episode_contract_pass") is True
        ),
        "current_activation_pass": bool(
            isinstance(activation, dict) and activation.get("passed") is True
        ),
        "domain_identity": bool(
            isinstance(physics, dict)
            and physics.get("domain_id") == job["domain_id"]
        ),
        "reference_identity": bool(
            recipe.get("reference_id") == job["reference_id"]
            and report
            and report.get("data_split") == job.get("command_split", "train")
            and command_path is not None
            and command_path.resolve() == Path(job["reference_path"]).resolve()
        ),
        "controller_identity": bool(
            not job.get("controller_implementation_id")
            or (
                report
                and report.get("controller_provenance", {}).get("controller_id")
                == job["controller_implementation_id"]
            )
        ),
        "episode_file_exists": bool(
            report and str(report.get("episode_path", "")).strip() and episode_path.is_file()
        ),
        "independent_episode_inspection": episode_report.get("pass") is True,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "report_error": report_error,
        "episode_inspection": episode_report,
        "report": report,
    }


def _attempt_directory(job_root: Path, attempts: list[dict[str, Any]]) -> tuple[int, Path]:
    index = len(attempts) + 1
    while True:
        path = job_root / f"attempt_{index:02d}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return index, path
        except FileExistsError:
            index += 1


def _materialize_vins_config(attempt_dir: Path) -> Path:
    """Give each VINS process private result files and calibration copies."""
    source_dir = (
        Path(get_package_share_directory("disaware_learning")) / "config"
    )
    config_dir = attempt_dir / "vins_config"
    output_dir = attempt_dir / "vins_output"
    config_dir.mkdir()
    output_dir.mkdir()
    for filename in _VINS_CONFIG_FILENAMES[1:]:
        shutil.copy2(source_dir / filename, config_dir / filename)

    source = source_dir / _VINS_CONFIG_FILENAMES[0]
    destination = config_dir / source.name
    lines = source.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith("output_path:")]
    if len(matches) != 1:
        raise ValueError(f"VINS config must define exactly one output_path: {source}")
    lines[matches[0]] = f'output_path: "{output_dir}"'
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _launch_command(
    job: Mapping[str, Any],
    runtime: WorkerRuntime,
    attempt_dir: Path,
    domain_manifest_path: Path,
    spawn_depth: float,
) -> list[str]:
    episode_dir = attempt_dir / "episodes"
    episode_dir.mkdir()
    sitl_dir = attempt_dir / "ardusub_sitl"
    sitl_dir.mkdir()
    vins_config_path = _materialize_vins_config(attempt_dir)
    planned_spawn_depth = float(job.get("spawn_depth", spawn_depth))
    command = [
        "ros2",
        "launch",
        "disaware_learning",
        "manual_rollout.launch.py",
        f"controller_executable:={job.get('controller_executable', 'manual_rollout')}",
        f"nmpc_update_steps:={int(job.get('nmpc_update_steps', 8))}",
        f"command_path:={job['reference_path']}",
        f"domain_bank_path:={domain_manifest_path}",
        f"domain_id:={job['domain_id']}",
        f"scenario_path:={job['scenario_path']}",
        f"spawn_depth:={planned_spawn_depth}",
        f"maximum_depth_down_m:={planned_spawn_depth + 2.0}",
        "reset_convergence_timeout_seconds:=30.0",
        "startup_max_linear_speed_m_s:=0.10",
        "startup_max_angular_speed_rad_s:=0.10",
        "startup_max_tilt_degrees:=8.0",
        f"output_dir:={episode_dir}",
        f"report_path:={attempt_dir / 'report.json'}",
        f"pid_trace_path:={attempt_dir / 'simulation_audit_control_trace.npz'}",
        "record_ground_truth:=false",
        "depth_source:=fluid_pressure",
        "pressure_topic:=/bluerov2/pressure",
        "fluid_density_kg_m3:=1025.0",
        "surface_pressure_pa:=0.0",
        f"bind_port:={runtime.bind_port}",
        f"sitl_instance:={runtime.sitl_instance}",
        f"mavlink_out:=udp:127.0.0.1:{runtime.mavlink_port}",
        f"mavlink_endpoint:=udpin:0.0.0.0:{runtime.mavlink_port}",
        f"external_nav_out:=udp:127.0.0.1:{runtime.external_nav_port}",
        f"ahrs_out:=udp:127.0.0.1:{runtime.ahrs_port}",
        f"ahrs_endpoint:=udpin:0.0.0.0:{runtime.ahrs_port}",
        f"contract_out:=udp:127.0.0.1:{runtime.contract_port}",
        f"ardusub_contract_endpoint:=udpin:0.0.0.0:{runtime.contract_port}",
        f"ardupilot_use_dir:={sitl_dir}",
        "start_simulator:=true",
        "simulation_reset:=true",
        "use_nogpu:=false",
        "start_vins:=true",
        f"vins_config_path:={vins_config_path}",
        "vio_odometry_topic:=/bluerov2/vio/odometry",
        "vio_source_id:=vins_fusion_ros2_imu_propagate",
    ]
    nmpc_config_path = str(job.get("nmpc_config_path", "")).strip()
    if nmpc_config_path:
        command.insert(5, f"nmpc_config_path:={nmpc_config_path}")
    pid_config_path = str(job.get("pid_config_path", "")).strip()
    if pid_config_path:
        command.insert(5, f"pid_config_path:={pid_config_path}")
    if job.get("controller_executable") in _MPPI_CONTROLLER_EXECUTABLES:
        command.extend(
            [
                "command_rate:=50.0",
                "resample_reference_to_command_rate:=true",
                f"stage2_checkpoint:={job['stage2_checkpoint']}",
                f"jepa_mppi_config:={job['jepa_mppi_config_path']}",
                f"jepa_mppi_device:={job.get('jepa_mppi_device', 'auto')}",
                f"mppi_trace_path:={attempt_dir / 'mppi_trace.npz'}",
            ]
        )
    return command


def _run_attempt(
    job: Mapping[str, Any],
    runtime: WorkerRuntime,
    *,
    attempt_index: int,
    attempt_dir: Path,
    domain_manifest_path: Path,
    timeout_seconds: float,
    native_threads: int,
    spawn_depth: float,
    stop_event: threading.Event | None = None,
    process_supervisor: BatchProcessSupervisor | None = None,
) -> dict[str, Any]:
    command = _launch_command(job, runtime, attempt_dir, domain_manifest_path, spawn_depth)
    log_path = attempt_dir / "ros_launch.log"
    ros_log_dir = attempt_dir / "ros_logs"
    temporary_dir = attempt_dir / "tmp"
    ros_home = attempt_dir / "ros_home"
    for directory in (ros_log_dir, temporary_dir, ros_home):
        directory.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "ROS_DOMAIN_ID": str(runtime.ros_domain_id),
            "ROS_LOG_DIR": str(ros_log_dir),
            "ROS_HOME": str(ros_home),
            "TMPDIR": str(temporary_dir),
        }
    )
    for variable in _NATIVE_THREAD_VARIABLES:
        environment[variable] = str(native_threads)
    _augment_ros_pythonpath(environment)
    started_at = datetime.now().astimezone().isoformat()
    started = time.monotonic()
    timed_out = False
    batch_interrupted = False
    return_code = -1
    cleanup: dict[str, Any] = {
        "signals_sent": [],
        "process_group_exited": True,
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {shlex.join(command)}\n")
        log.write(f"ROS_DOMAIN_ID: {runtime.ros_domain_id}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env=environment,
        )
        registered = False
        try:
            if process_supervisor is not None:
                process_supervisor.register(process, attempt_dir, command)
                registered = True
            deadline = time.monotonic() + timeout_seconds
            while process.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    batch_interrupted = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    timed_out = True
                    break
                try:
                    return_code = process.wait(timeout=min(0.25, remaining))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if stop_event is not None and stop_event.is_set():
                batch_interrupted = True
            cleanup = terminate_process_group(process)
            if process.poll() is None:
                try:
                    return_code = process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    return_code = -signal.SIGKILL
            else:
                return_code = int(process.returncode)
            if registered and process_supervisor is not None:
                process_supervisor.finish(process, cleanup, return_code)
    attempt: dict[str, Any] = {
        "attempt_index": attempt_index,
        "attempt_dir": str(attempt_dir),
        "started_at": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "return_code": return_code,
        "timed_out": timed_out,
        "batch_interrupted": batch_interrupted,
        "process_group_exited": cleanup["process_group_exited"],
        "cleanup_signals": cleanup["signals_sent"],
        "report_path": str(attempt_dir / "report.json"),
        "pid_trace_path": str(attempt_dir / "simulation_audit_control_trace.npz"),
        "log_path": str(log_path),
        "worker_runtime": asdict(runtime),
    }
    validation = validate_attempt(job, attempt)
    attempt["pass"] = validation["pass"]
    attempt["validation"] = {
        "checks": validation["checks"],
        "report_error": validation["report_error"],
        "episode_inspection": validation["episode_inspection"],
    }
    report = validation["report"]
    attempt["episode_path"] = report.get("episode_path") if report else None
    return attempt


def _summary(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["status"] for record in records)
    return {
        "total": len(records),
        "pending": counts["pending"],
        "running": counts["running"],
        "passed": counts["passed"],
        "failed": counts["failed"],
    }


def _new_or_resumed_report(
    path: Path,
    plan: Mapping[str, Any],
    runtimes: list[WorkerRuntime],
) -> dict[str, Any]:
    if path.is_file():
        report = _load_json(path)
        for record in report["jobs"]:
            if record["status"] == "running":
                record["status"] = "pending"
            if record["status"] == "passed":
                accepted = record.get("accepted_attempt")
                if not accepted or not validate_attempt(record, accepted)["pass"]:
                    record["status"] = "pending"
                    record["accepted_attempt"] = None
        report["resumed_at"] = datetime.now().astimezone().isoformat()
        report["summary"] = _summary(report["jobs"])
        return report
    records = [
        {**job, "status": "pending", "attempts": [], "accepted_attempt": None}
        for job in plan["jobs"]
    ]
    return {
        "protocol_id": BATCH_PROTOCOL_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "worker_runtimes": [asdict(runtime) for runtime in runtimes],
        "jobs": records,
        "summary": _summary(records),
    }


def run_batch(
    plan: Mapping[str, Any],
    output_dir: str | Path,
    *,
    workers: int = 2,
    ros_domain_base: int = 100,
    max_attempts: int = 2,
    timeout_seconds: float = 180.0,
    launch_stagger_seconds: float = 2.0,
    native_threads: int = 2,
    spawn_depth: float = 9.0,
) -> dict[str, Any]:
    if workers < 1 or max_attempts < 1 or timeout_seconds <= 0.0:
        raise ValueError("workers, attempts, and timeout must be positive")
    if launch_stagger_seconds < 0.0 or native_threads < 1 or not 0.5 <= spawn_depth <= 20.0:
        raise ValueError("stagger must be nonnegative and native threads positive")
    worker_count = min(workers, len(plan["jobs"]))
    runtimes = [worker_runtime(index, ros_domain_base) for index in range(worker_count)]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / PLAN_NAME
    if plan_path.is_file():
        existing_plan = _load_json(plan_path)
        if existing_plan != plan:
            raise RuntimeError("output directory contains a different batch plan")
    else:
        _write_json_atomic(plan_path, plan)
    report_path = output / REPORT_NAME
    stop_event = threading.Event()
    process_supervisor = BatchProcessSupervisor(output, stop_event)
    stale_cleanup = process_supervisor.cleanup_stale_attempts()
    report = _new_or_resumed_report(report_path, plan, runtimes)
    if stale_cleanup:
        report.setdefault("startup_process_recovery", []).append(
            {
                "recovered_at": datetime.now().astimezone().isoformat(),
                "attempts": stale_cleanup,
            }
        )
    report_lock = threading.Lock()
    _write_json_atomic(report_path, report)
    started = time.monotonic()
    domain_manifest_path = Path(plan["domain_manifest_path"])

    def persist() -> None:
        report["summary"] = _summary(report["jobs"])
        _write_json_atomic(report_path, report)

    def run_worker(worker_id: int) -> None:
        runtime = runtimes[worker_id]
        for record in report["jobs"][worker_id::worker_count]:
            if stop_event.is_set():
                return
            with report_lock:
                if record["status"] == "passed":
                    continue
                record["status"] = "running"
                persist()
            completed_attempts = sum(
                attempt.get("batch_interrupted") is not True
                for attempt in record["attempts"]
            )
            while completed_attempts < max_attempts and not stop_event.is_set():
                attempt_index, attempt_dir = _attempt_directory(
                    output / "jobs" / record["job_id"], record["attempts"]
                )
                attempt = _run_attempt(
                    record,
                    runtime,
                    attempt_index=attempt_index,
                    attempt_dir=attempt_dir,
                    domain_manifest_path=domain_manifest_path,
                    timeout_seconds=timeout_seconds,
                    native_threads=native_threads,
                    spawn_depth=spawn_depth,
                    stop_event=stop_event,
                    process_supervisor=process_supervisor,
                )
                with report_lock:
                    record["attempts"].append(attempt)
                    if attempt["pass"]:
                        record["status"] = "passed"
                        record["accepted_attempt"] = attempt
                    else:
                        record["status"] = "pending"
                    persist()
                if attempt.get("batch_interrupted") is not True:
                    completed_attempts += 1
                if stop_event.is_set():
                    return
                if attempt["pass"]:
                    break
            with report_lock:
                if record["status"] != "passed":
                    record["status"] = (
                        "pending" if stop_event.is_set() else "failed"
                    )
                persist()

    received_signal: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def handle_stop(signum: int, _frame: Any) -> None:
        if not received_signal:
            received_signal.append(signum)
        process_supervisor.interrupt_active()

    install_signal_handlers = threading.current_thread() is threading.main_thread()
    if install_signal_handlers:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_stop)
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for worker_id in range(worker_count):
                if stop_event.is_set():
                    break
                futures.append(executor.submit(run_worker, worker_id))
                if worker_id + 1 < worker_count and launch_stagger_seconds:
                    stop_event.wait(launch_stagger_seconds)
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                process_supervisor.interrupt_active()
                raise
    finally:
        if stop_event.is_set():
            process_supervisor.interrupt_active()
        if install_signal_handlers:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    if stop_event.is_set():
        report["interrupted_at"] = datetime.now().astimezone().isoformat()
        report["interrupt_signal"] = received_signal[0] if received_signal else None
    else:
        report["completed_at"] = datetime.now().astimezone().isoformat()
    report["wall_time_seconds"] = time.monotonic() - started
    report["summary"] = _summary(report["jobs"])
    _write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_manifest")
    parser.add_argument("domain_manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--domain-count", type=int, default=4)
    parser.add_argument("--references-per-domain", type=int, default=1)
    parser.add_argument(
        "--balanced-reference-pairs",
        action="store_true",
        help="assign distinct references per domain with balanced random marginals",
    )
    parser.add_argument("--pair-seed", type=int, default=20260815)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ros-domain-base", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--native-threads", type=int, default=2)
    parser.add_argument("--spawn-depth", type=float, default=9.0)
    parser.add_argument(
        "--controller-executable",
        choices=sorted(_CONTROLLER_IMPLEMENTATION_IDS),
        default="manual_rollout",
    )
    parser.add_argument("--nmpc-config-path", default="")
    parser.add_argument("--pid-config-path", default="")
    parser.add_argument("--nmpc-update-steps", type=int, default=8)
    parser.add_argument("--stage2-checkpoint", default="")
    parser.add_argument("--jepa-mppi-config-path", default="")
    parser.add_argument(
        "--jepa-mppi-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    plan = build_job_plan(
        arguments.reference_manifest,
        arguments.domain_manifest,
        domain_count=arguments.domain_count,
        references_per_domain=arguments.references_per_domain,
        balanced_reference_pairs=arguments.balanced_reference_pairs,
        pair_seed=arguments.pair_seed,
        controller_executable=arguments.controller_executable,
        pid_config_path=arguments.pid_config_path or None,
        nmpc_config_path=arguments.nmpc_config_path or None,
        nmpc_update_steps=arguments.nmpc_update_steps,
        stage2_checkpoint=arguments.stage2_checkpoint or None,
        jepa_mppi_config_path=arguments.jepa_mppi_config_path or None,
        jepa_mppi_device=arguments.jepa_mppi_device,
    )
    output = Path(arguments.output_dir).expanduser().resolve()
    if arguments.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output / PLAN_NAME, plan)
        print(json.dumps({"plan": str(output / PLAN_NAME), "jobs": plan["job_count"]}))
        return 0
    report = run_batch(
        plan,
        output,
        workers=arguments.workers,
        ros_domain_base=arguments.ros_domain_base,
        max_attempts=arguments.max_attempts,
        timeout_seconds=arguments.timeout_seconds,
        launch_stagger_seconds=arguments.launch_stagger_seconds,
        native_threads=arguments.native_threads,
        spawn_depth=arguments.spawn_depth,
    )
    print(json.dumps({"report": str(output / REPORT_NAME), **report["summary"]}))
    if report.get("interrupted_at"):
        return 128 + int(report.get("interrupt_signal") or signal.SIGINT)
    return 0 if report["summary"]["failed"] == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
