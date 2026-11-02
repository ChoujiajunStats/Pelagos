"""Generate and materialize continuous Stonefish physics domains."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import qmc

from disaware_learning.control.contract import ACTUATOR_COUNT
from disaware_learning.collection.current_generator import (
    CURRENT_MODEL_ID,
    CURRENT_SECONDARY_PHASE_OFFSET_RAD,
    CURRENT_SECONDARY_PERIOD_FRACTION,
    CURRENT_SECONDARY_WEIGHT,
    CURRENT_DIRECTION_WOBBLE_AMPLITUDE_RAD,
    CURRENT_DIRECTION_WOBBLE_PERIOD_FRACTION,
    CURRENT_DIRECTION_WOBBLE_PERIOD_MIN_SECONDS,
    ExternalCurrentGenerator,
)


DOMAIN_FORMAT_VERSION = 5
DOMAIN_PROTOCOL_ID = "manual_six_thruster_physics_domain_v5"
DOMAIN_BANK_PROTOCOL_ID = "manual_six_thruster_physics_domain_bank_v5"
SENSOR_ROBUSTNESS_PROTOCOL_ID = "manual_sensor_robustness_v1"
DOMAIN_MANIFEST_NAME = "physics_domain_bank_manifest.json"
DOMAIN_SPLITS = ("train", "validation", "test")
# External zero-mean current remains below the slow-reference velocity scale.
# It is published only after controller initialization, never baked into the
# scenario as a static velocity bias.
TRAINING_CURRENT_AMPLITUDE_RANGE_M_S = (0.0, 0.05)
STRESS_CURRENT_AMPLITUDE_RANGE_M_S = (0.08, 0.10)
EXTERNAL_CURRENT_PERIOD_RANGE_SECONDS = (20.0, 60.0)
EXTERNAL_CURRENT_RAMP_SECONDS = 2.0
EXTERNAL_CURRENT_TOPIC = "/bluerov2/disturbance/current_command_ned"
THRUSTER_ORDER = (
    "FrontRight",
    "FrontLeft",
    "BackRight",
    "BackLeft",
    "DiveRight",
    "DiveLeft",
)
BALLAST_PART_NAMES = ("WeightCenter", "WeightLeft", "WeightRight")
EXTERNAL_DRAG_PART_NAMES = ("HullBottom", "HeavyFit")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _finite_tuple(value: object, width: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != width or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {width} finite values")
    return result


def _range(value: object, name: str) -> tuple[float, float]:
    result = _finite_tuple(value, 2, name)
    # Degenerate bounds are intentional for the nominal-only benchmark.  A
    # fixed range materializes one exact plant while retaining the same domain
    # manifest/provenance contract as randomized banks.
    if result[0] > result[1]:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    return result


def _scale(unit_value: float, bounds: tuple[float, float]) -> float:
    return float(bounds[0] + unit_value * (bounds[1] - bounds[0]))


@dataclass(frozen=True)
class PhysicsDomainSpec:
    """Ranges for one continuous, physically materializable domain bank."""

    rotor_tau_seconds: tuple[float, float] = (0.05, 0.20)
    global_thrust_gain: tuple[float, float] = (0.80, 1.20)
    thruster_gain_residual: tuple[float, float] = (-0.08, 0.08)
    payload_mass_delta_kg: tuple[float, float] = (-0.40, 0.80)
    # Six-thruster vehicles are physically trimmed before deployment. Keep the
    # static bias modest; dynamic response variation remains broad below.
    ballast_offset_x_m: tuple[float, float] = (-0.006, 0.006)
    ballast_offset_y_m: tuple[float, float] = (-0.005, 0.005)
    ballast_offset_z_m: tuple[float, float] = (-0.008, 0.008)
    net_buoyancy_force_n: tuple[float, float] = (0.0, 2.0)
    quadratic_drag_coefficient: tuple[float, float] = (0.50, 1.20)
    viscous_to_quadratic_ratio: float = 0.10
    current_amplitude_m_s: tuple[float, float] = (
        TRAINING_CURRENT_AMPLITUDE_RANGE_M_S
    )
    current_direction_rad: tuple[float, float] = (-math.pi, math.pi)
    current_period_seconds: tuple[float, float] = (
        EXTERNAL_CURRENT_PERIOD_RANGE_SECONDS
    )
    current_phase_rad: tuple[float, float] = (-math.pi, math.pi)
    # The second low-frequency component is sampled independently within a
    # narrow band. The mixture weight and direction drift remain fixed release
    # constants, so the domain dimensionality stays controlled.
    current_secondary_period_seconds: tuple[float, float] = (20.0, 36.0)
    current_secondary_phase_rad: tuple[float, float] = (-math.pi, math.pi)
    current_ramp_seconds: float = EXTERNAL_CURRENT_RAMP_SECONDS
    nominal_vehicle_mass_kg: float = 11.189
    nominal_buoyant_volume_m3: float = 10.92e-3
    water_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.80665
    nominal_moved_ballast_mass_kg: float = 4.0
    vertical_thruster_arm_y_m: float = 0.1105
    vertical_thruster_max_setpoint_rad_s: float = 376.8
    vertical_thruster_diameter_m: float = 0.076
    vertical_thruster_thrust_coefficient: float = 0.277
    maximum_static_vertical_command: float = 0.30
    maximum_static_pitch_moment_nm: float = 0.30

    def __post_init__(self) -> None:
        range_names = (
            "rotor_tau_seconds",
            "global_thrust_gain",
            "thruster_gain_residual",
            "payload_mass_delta_kg",
            "ballast_offset_x_m",
            "ballast_offset_y_m",
            "ballast_offset_z_m",
            "net_buoyancy_force_n",
                "quadratic_drag_coefficient",
                "current_amplitude_m_s",
                "current_direction_rad",
                "current_period_seconds",
                "current_phase_rad",
                "current_secondary_period_seconds",
                "current_secondary_phase_rad",
        )
        for name in range_names:
            object.__setattr__(self, name, _range(getattr(self, name), name))
        positive = (
            "nominal_vehicle_mass_kg",
            "nominal_buoyant_volume_m3",
            "water_density_kg_m3",
            "gravity_m_s2",
            "viscous_to_quadratic_ratio",
            "nominal_moved_ballast_mass_kg",
            "vertical_thruster_arm_y_m",
            "vertical_thruster_max_setpoint_rad_s",
            "vertical_thruster_diameter_m",
            "vertical_thruster_thrust_coefficient",
            "maximum_static_vertical_command",
            "maximum_static_pitch_moment_nm",
            "current_ramp_seconds",
        )
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in positive
        ):
            raise ValueError("nominal physical constants must be positive and finite")
        if self.rotor_tau_seconds[0] <= 0.0:
            raise ValueError("first-order rotor time constants must be positive")
        if self.global_thrust_gain[0] <= 0.0:
            raise ValueError("global thrust gain must stay positive")
        if self.thruster_gain_residual[0] <= -1.0:
            raise ValueError("thruster residuals must preserve positive gain")
        if self.net_buoyancy_force_n[0] < 0.0:
            raise ValueError("the domain bank requires neutral or positive buoyancy")
        if self.quadratic_drag_coefficient[0] <= 0.0:
            raise ValueError("quadratic drag coefficients must stay positive")
        if self.current_amplitude_m_s[0] < 0.0:
            raise ValueError("current amplitude cannot be negative")
        if self.current_period_seconds[0] <= 0.0:
            raise ValueError("current period must be positive")
        minimum_moved_mass = (
            self.nominal_moved_ballast_mass_kg + self.payload_mass_delta_kg[0]
        )
        if minimum_moved_mass <= 0.0:
            raise ValueError("payload range must preserve positive moved ballast mass")
        feasibility = self.static_feasibility_contract()
        if not feasibility["pass"]:
            raise ValueError(
                "physics-domain bounds violate the six-thruster static "
                f"feasibility contract: {feasibility}"
            )

    @property
    def nominal_vertical_thruster_max_force_n(self) -> float:
        revolutions_per_second = self.vertical_thruster_max_setpoint_rad_s / (
            2.0 * math.pi
        )
        return float(
            self.vertical_thruster_thrust_coefficient
            * self.water_density_kg_m3
            * revolutions_per_second**2
            * self.vertical_thruster_diameter_m**4
        )

    def static_feasibility_contract(self) -> dict[str, Any]:
        """Conservatively bound level heave/roll trim at the range corners."""
        maximum_moved_mass = (
            self.nominal_moved_ballast_mass_kg + self.payload_mass_delta_kg[1]
        )
        maximum_abs_x = max(abs(value) for value in self.ballast_offset_x_m)
        maximum_abs_y = max(abs(value) for value in self.ballast_offset_y_m)
        maximum_roll_moment = (
            maximum_moved_mass * self.gravity_m_s2 * maximum_abs_y
        )
        maximum_pitch_moment = (
            maximum_moved_mass * self.gravity_m_s2 * maximum_abs_x
        )
        maximum_thruster_force = 0.5 * (
            self.net_buoyancy_force_n[1]
            + maximum_roll_moment / self.vertical_thruster_arm_y_m
        )
        minimum_thruster_gain = self.global_thrust_gain[0] * (
            1.0 + self.thruster_gain_residual[0]
        )
        maximum_vertical_command = math.sqrt(
            maximum_thruster_force
            / (self.nominal_vertical_thruster_max_force_n * minimum_thruster_gain)
        )
        passed = bool(
            maximum_vertical_command <= self.maximum_static_vertical_command
            and maximum_pitch_moment <= self.maximum_static_pitch_moment_nm
        )
        return {
            "method": "independent_range_corner_bound",
            "maximum_abs_roll_moment_nm": maximum_roll_moment,
            "maximum_abs_pitch_moment_nm": maximum_pitch_moment,
            "maximum_abs_vertical_command": maximum_vertical_command,
            "vertical_command_limit": self.maximum_static_vertical_command,
            "pitch_moment_limit_nm": self.maximum_static_pitch_moment_nm,
            "nominal_vertical_thruster_max_force_n": (
                self.nominal_vertical_thruster_max_force_n
            ),
            "pass": passed,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            name: list(getattr(self, name))
            for name in (
                "rotor_tau_seconds",
                "global_thrust_gain",
                "thruster_gain_residual",
                "payload_mass_delta_kg",
                "ballast_offset_x_m",
                "ballast_offset_y_m",
                "ballast_offset_z_m",
                "net_buoyancy_force_n",
                "quadratic_drag_coefficient",
                "current_amplitude_m_s",
                "current_direction_rad",
            "current_period_seconds",
            "current_phase_rad",
            "current_secondary_period_seconds",
            "current_secondary_phase_rad",
            )
        } | {
            "nominal_vehicle_mass_kg": self.nominal_vehicle_mass_kg,
            "nominal_buoyant_volume_m3": self.nominal_buoyant_volume_m3,
            "water_density_kg_m3": self.water_density_kg_m3,
            "gravity_m_s2": self.gravity_m_s2,
            "viscous_to_quadratic_ratio": self.viscous_to_quadratic_ratio,
            "nominal_moved_ballast_mass_kg": self.nominal_moved_ballast_mass_kg,
            "vertical_thruster_arm_y_m": self.vertical_thruster_arm_y_m,
            "vertical_thruster_max_setpoint_rad_s": (
                self.vertical_thruster_max_setpoint_rad_s
            ),
            "vertical_thruster_diameter_m": self.vertical_thruster_diameter_m,
            "vertical_thruster_thrust_coefficient": (
                self.vertical_thruster_thrust_coefficient
            ),
            "maximum_static_vertical_command": (
                self.maximum_static_vertical_command
            ),
            "maximum_static_pitch_moment_nm": (
                self.maximum_static_pitch_moment_nm
            ),
            "current_ramp_seconds": self.current_ramp_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicsDomainSpec":
        expected = set(cls().as_dict())
        if set(payload) != expected:
            raise ValueError("physics-domain spec has unexpected or missing fields")
        return cls(**dict(payload))


@dataclass(frozen=True)
class PhysicsDomain:
    """One episode-constant physical domain for the six-thruster vehicle."""

    domain_id: str
    data_split: str
    sample_index: int
    rotor_tau_seconds: float
    global_thrust_gain: float
    thruster_gain_residual: tuple[float, ...]
    payload_mass_delta_kg: float
    ballast_offset_xyz_m: tuple[float, ...]
    net_buoyancy_force_n: float
    buoyant_volume_scale: float
    quadratic_drag_coefficient_xyz: tuple[float, ...]
    viscous_drag_coefficient_xyz: tuple[float, ...]
    current_amplitude_m_s: float
    current_direction_rad: float
    current_period_seconds: float
    current_phase_rad: float
    current_ramp_seconds: float
    current_secondary_period_seconds: float | None = None
    current_secondary_phase_rad: float | None = None

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("domain_id must be non-empty")
        if self.data_split not in DOMAIN_SPLITS:
            raise ValueError(f"unsupported domain split: {self.data_split}")
        if int(self.sample_index) < 0:
            raise ValueError("sample_index must be nonnegative")
        object.__setattr__(self, "sample_index", int(self.sample_index))
        for name, width in (
            ("thruster_gain_residual", ACTUATOR_COUNT),
            ("ballast_offset_xyz_m", 3),
            ("quadratic_drag_coefficient_xyz", 3),
            ("viscous_drag_coefficient_xyz", 3),
        ):
            object.__setattr__(
                self,
                name,
                _finite_tuple(getattr(self, name), width, name),
            )
        scalars = (
            self.rotor_tau_seconds,
            self.global_thrust_gain,
            self.payload_mass_delta_kg,
            self.net_buoyancy_force_n,
            self.buoyant_volume_scale,
            self.current_amplitude_m_s,
            self.current_direction_rad,
            self.current_period_seconds,
            self.current_phase_rad,
            self.current_ramp_seconds,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("domain scalar parameters must be finite")
        if self.rotor_tau_seconds <= 0.0:
            raise ValueError("rotor time constant must be positive")
        if self.global_thrust_gain <= 0.0 or any(
            gain <= 0.0 for gain in self.thruster_gain
        ):
            raise ValueError("realized thruster gains must be positive")
        if self.net_buoyancy_force_n < 0.0 or self.buoyant_volume_scale <= 0.0:
            raise ValueError("vehicle must retain nonnegative net buoyancy")
        if any(value <= 0.0 for value in self.quadratic_drag_coefficient_xyz):
            raise ValueError("quadratic drag coefficients must be positive")
        if any(value <= 0.0 for value in self.viscous_drag_coefficient_xyz):
            raise ValueError("viscous drag coefficients must be positive")
        if self.current_amplitude_m_s < 0.0:
            raise ValueError("current amplitude cannot be negative")
        if self.current_period_seconds <= 0.0 or self.current_ramp_seconds <= 0.0:
            raise ValueError("current period and ramp must be positive")
        if self.current_secondary_period_seconds is not None and (
            not math.isfinite(float(self.current_secondary_period_seconds))
            or self.current_secondary_period_seconds <= 0.0
        ):
            raise ValueError("secondary current period must be positive and finite")
        if self.current_secondary_phase_rad is not None and not math.isfinite(
            float(self.current_secondary_phase_rad)
        ):
            raise ValueError("secondary current phase must be finite")

    @property
    def thruster_gain(self) -> tuple[float, ...]:
        return tuple(
            self.global_thrust_gain * (1.0 + residual)
            for residual in self.thruster_gain_residual
        )

    def external_current_velocity_ned_m_s(
        self, elapsed_seconds: float
    ) -> tuple[float, float, float]:
        """Return the commanded zero-bias current on the formal episode clock."""
        return self.current_generator().velocity_ned_m_s(elapsed_seconds)

    def current_generator(self) -> ExternalCurrentGenerator:
        """Build the deterministic generator represented by this domain."""
        return ExternalCurrentGenerator(
            amplitude_m_s=self.current_amplitude_m_s,
            direction_rad=self.current_direction_rad,
            period_seconds=self.current_period_seconds,
            phase_rad=self.current_phase_rad,
            ramp_seconds=self.current_ramp_seconds,
            secondary_period_seconds_override=self.current_secondary_period_seconds,
            secondary_phase_rad_override=self.current_secondary_phase_rad,
        )

    def as_record(self) -> dict[str, Any]:
        generator = self.current_generator()
        environment = generator.as_dict()
        environment.update(
            {
                "current_topic": EXTERNAL_CURRENT_TOPIC,
                "static_current_velocity_ned_m_s": [0.0, 0.0, 0.0],
            }
        )
        return {
            "domain_id": self.domain_id,
            "data_split": self.data_split,
            "sample_index": self.sample_index,
            "parameters": {
                "actuator": {
                    "rotor_model": "first_order",
                    "rotor_tau_seconds": self.rotor_tau_seconds,
                    "global_thrust_gain": self.global_thrust_gain,
                    "thruster_gain_residual": list(self.thruster_gain_residual),
                    "realized_thruster_gain": list(self.thruster_gain),
                },
                "vehicle": {
                    "payload_mass_delta_kg": self.payload_mass_delta_kg,
                    "ballast_offset_xyz_m": list(self.ballast_offset_xyz_m),
                    "net_buoyancy_force_n": self.net_buoyancy_force_n,
                    "buoyant_volume_scale": self.buoyant_volume_scale,
                },
                "hydrodynamics": {
                    "quadratic_drag_coefficient_xyz": list(
                        self.quadratic_drag_coefficient_xyz
                    ),
                    "viscous_drag_coefficient_xyz": list(
                        self.viscous_drag_coefficient_xyz
                    ),
                },
                "environment": environment,
            },
        }

    def manifest_entry(self) -> dict[str, Any]:
        return self.as_record()

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> "PhysicsDomain":
        parameters = payload["parameters"]
        actuator = parameters["actuator"]
        vehicle = parameters["vehicle"]
        hydrodynamics = parameters["hydrodynamics"]
        environment = parameters["environment"]
        if actuator.get("rotor_model") != "first_order":
            raise ValueError("physics-domain actuator must use first_order")
        domain = cls(
            domain_id=str(payload["domain_id"]),
            data_split=str(payload["data_split"]),
            sample_index=int(payload["sample_index"]),
            rotor_tau_seconds=float(actuator["rotor_tau_seconds"]),
            global_thrust_gain=float(actuator["global_thrust_gain"]),
            thruster_gain_residual=tuple(actuator["thruster_gain_residual"]),
            payload_mass_delta_kg=float(vehicle["payload_mass_delta_kg"]),
            ballast_offset_xyz_m=tuple(vehicle["ballast_offset_xyz_m"]),
            net_buoyancy_force_n=float(vehicle["net_buoyancy_force_n"]),
            buoyant_volume_scale=float(vehicle["buoyant_volume_scale"]),
            quadratic_drag_coefficient_xyz=tuple(
                hydrodynamics["quadratic_drag_coefficient_xyz"]
            ),
            viscous_drag_coefficient_xyz=tuple(
                hydrodynamics["viscous_drag_coefficient_xyz"]
            ),
            current_amplitude_m_s=float(environment["current_amplitude_m_s"]),
            current_direction_rad=float(environment["current_direction_rad"]),
            current_period_seconds=float(environment["current_period_seconds"]),
            current_phase_rad=float(environment["current_phase_rad"]),
            current_ramp_seconds=float(environment["current_ramp_seconds"]),
            current_secondary_period_seconds=float(
                environment["current_secondary_period_seconds"]
            ),
            current_secondary_phase_rad=float(
                environment["current_secondary_phase_rad"]
            ),
        )
        expected_derived = {
            "realized_thruster_gain": list(domain.thruster_gain),
            "static_current_velocity_ned_m_s": [0.0, 0.0, 0.0],
        }
        expected_current = domain.current_generator().as_dict()
        current_provenance_matches = True
        for key, expected in expected_current.items():
            stored = environment.get(key)
            if key == "current_model":
                current_provenance_matches &= stored == expected
            else:
                try:
                    current_provenance_matches &= math.isclose(
                        float(stored),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                except (TypeError, ValueError):
                    current_provenance_matches = False
        if not np.allclose(
            actuator["realized_thruster_gain"],
            expected_derived["realized_thruster_gain"],
            rtol=0.0,
            atol=1e-12,
        ) or not np.allclose(
            environment["static_current_velocity_ned_m_s"],
            expected_derived["static_current_velocity_ned_m_s"],
            rtol=0.0,
            atol=1e-12,
        ) or not current_provenance_matches or environment.get(
            "current_topic"
        ) != EXTERNAL_CURRENT_TOPIC:
            raise ValueError("stored domain derived values are inconsistent")
        return domain


def _split_labels(
    count: int,
    train_count: int | None,
    validation_count: int | None,
) -> tuple[str, ...]:
    if count == 1:
        if train_count not in (None, 1) or validation_count not in (None, 0):
            raise ValueError("a one-domain nominal bank only has a train split")
        return ("train",)
    if count < 3:
        raise ValueError("domain bank must contain at least three domains")
    if train_count is None and validation_count is None:
        validation = max(1, int(round(0.1 * count)))
        test = max(1, int(round(0.1 * count)))
        train = count - validation - test
    else:
        train = int(round(0.8 * count)) if train_count is None else int(train_count)
        validation = (
            max(1, int(round(0.1 * count)))
            if validation_count is None
            else int(validation_count)
        )
        test = count - train - validation
    if min(train, validation, test) < 1:
        raise ValueError("train, validation, and test domain splits must be nonempty")
    return ("train",) * train + ("validation",) * validation + ("test",) * test


def _scrambled_sobol(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return a reproducible scrambled Sobol design in the unit cube."""
    if count < 1 or dimensions < 1:
        raise ValueError("Sobol count and dimensions must be positive")
    engine = qmc.Sobol(d=dimensions, scramble=True, seed=int(seed))
    return np.asarray(engine.random(n=count), dtype=np.float64)


def sample_physics_domains(
    *,
    count: int = 160,
    seed: int = 20260802,
    spec: PhysicsDomainSpec = PhysicsDomainSpec(),
    train_count: int | None = None,
    validation_count: int | None = None,
    split_seeds: Mapping[str, int] | None = None,
) -> list[PhysicsDomain]:
    """Sample deterministic continuous domains with balanced Sobol coverage."""
    labels = _split_labels(count, train_count, validation_count)
    if split_seeds is None:
        unit = _scrambled_sobol(count, 22, seed)
    else:
        split_counts = {
            split: labels.count(split) for split in DOMAIN_SPLITS
        }
        missing = set(DOMAIN_SPLITS) - set(split_seeds)
        if missing:
            raise ValueError(f"split_seeds missing: {sorted(missing)}")
        unit = np.vstack(
            [
                _scrambled_sobol(
                    split_counts[split],
                    22,
                    int(split_seeds[split]),
                )
                for split in DOMAIN_SPLITS
            ]
        )
    domains: list[PhysicsDomain] = []
    residual_bounds = spec.thruster_gain_residual
    residual_limit = max(abs(residual_bounds[0]), abs(residual_bounds[1]))
    for index, row in enumerate(unit):
        cursor = 0

        def take(bounds: tuple[float, float]) -> float:
            nonlocal cursor
            value = _scale(float(row[cursor]), bounds)
            cursor += 1
            return value

        rotor_tau = take(spec.rotor_tau_seconds)
        global_gain = take(spec.global_thrust_gain)
        raw_residual = np.asarray(
            [take(spec.thruster_gain_residual) for _ in range(ACTUATOR_COUNT)],
            dtype=np.float64,
        )
        residual = raw_residual - float(np.mean(raw_residual))
        maximum = float(np.max(np.abs(residual)))
        if maximum > residual_limit:
            residual *= residual_limit / maximum
        payload_delta = take(spec.payload_mass_delta_kg)
        ballast_offset = (
            take(spec.ballast_offset_x_m),
            take(spec.ballast_offset_y_m),
            take(spec.ballast_offset_z_m),
        )
        net_buoyancy = take(spec.net_buoyancy_force_n)
        quadratic_drag = tuple(
            take(spec.quadratic_drag_coefficient) for _ in range(3)
        )
        viscous_drag = tuple(
            value * spec.viscous_to_quadratic_ratio for value in quadratic_drag
        )
        current_amplitude = take(spec.current_amplitude_m_s)
        current_direction = take(spec.current_direction_rad)
        current_period = take(spec.current_period_seconds)
        current_phase = take(spec.current_phase_rad)
        current_secondary_period = take(spec.current_secondary_period_seconds)
        current_secondary_phase = take(spec.current_secondary_phase_rad)
        assert cursor == unit.shape[1]
        displaced_mass = (
            spec.nominal_vehicle_mass_kg
            + payload_delta
            + net_buoyancy / spec.gravity_m_s2
        )
        buoyancy_scale = displaced_mass / (
            spec.water_density_kg_m3 * spec.nominal_buoyant_volume_m3
        )
        domains.append(
            PhysicsDomain(
                domain_id=f"physics_domain_{index:04d}",
                data_split=labels[index],
                sample_index=index,
                rotor_tau_seconds=rotor_tau,
                global_thrust_gain=global_gain,
                thruster_gain_residual=tuple(residual.tolist()),
                payload_mass_delta_kg=payload_delta,
                ballast_offset_xyz_m=ballast_offset,
                net_buoyancy_force_n=net_buoyancy,
                buoyant_volume_scale=buoyancy_scale,
                quadratic_drag_coefficient_xyz=quadratic_drag,
                viscous_drag_coefficient_xyz=viscous_drag,
                current_amplitude_m_s=current_amplitude,
                current_direction_rad=current_direction,
                current_period_seconds=current_period,
                current_phase_rad=current_phase,
                current_ramp_seconds=spec.current_ramp_seconds,
                current_secondary_period_seconds=current_secondary_period,
                current_secondary_phase_rad=current_secondary_phase,
            )
        )
    return domains


def _format_vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _parse_vector(value: str, width: int = 3) -> tuple[float, ...]:
    return _finite_tuple(value.split(), width, "XML vector")


def _replace_arguments(element: ET.Element, replacements: Mapping[str, str]) -> None:
    for item in element.iter():
        for key, value in tuple(item.attrib.items()):
            for name, replacement in replacements.items():
                value = value.replace(f"$(arg {name})", replacement)
            item.set(key, value)


def _xml_semantic_signature(element: ET.Element) -> tuple[Any, ...]:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_xml_semantic_signature(child) for child in element),
    )


def _base_scenario_parts(
    base_tank_path: str | Path,
    base_vehicle_path: str | Path,
) -> tuple[ET.Element, ET.Element, ET.Element, dict[str, str]]:
    tank_root = ET.parse(Path(base_tank_path)).getroot()
    vehicle_root = ET.parse(Path(base_vehicle_path)).getroot()
    include = tank_root.find("./include")
    robot = vehicle_root.find("./robot")
    if include is None or robot is None:
        raise ValueError("base scenarios must contain one include and one robot")
    replacements = {
        str(item.get("name")): str(item.get("value"))
        for item in include.findall("./arg")
        if item.get("name") and item.get("value") is not None
    }
    required = {"vehicle_name", "position", "orientation"}
    if not required.issubset(replacements):
        raise ValueError("base tank include does not define required robot arguments")
    return tank_root, vehicle_root, include, replacements


def render_domain_scenario(
    *,
    base_tank_path: str | Path,
    base_vehicle_path: str | Path,
    domain: PhysicsDomain,
    destination: str | Path,
) -> dict[str, Any]:
    """Inline and modify one base scenario using only effective parameters."""
    tank_root, vehicle_root, include, replacements = _base_scenario_parts(
        base_tank_path,
        base_vehicle_path,
    )
    water = tank_root.find("./environment/ocean/water")
    ocean = tank_root.find("./environment/ocean")
    uniform_current = tank_root.find(
        "./environment/ocean/current[@type='uniform']"
    )
    current_velocity = (
        None if uniform_current is None else uniform_current.find("./velocity")
    )
    if water is None or ocean is None or current_velocity is None:
        raise ValueError("base tank must define water and one uniform current")

    tank_looks = tank_root.find("./looks")
    vehicle_looks = vehicle_root.find("./looks")
    if tank_looks is None or vehicle_looks is None:
        raise ValueError("base scenarios must define looks")
    existing_looks = {item.get("name") for item in tank_looks.findall("./look")}
    for look in vehicle_looks.findall("./look"):
        if look.get("name") not in existing_looks:
            tank_looks.append(deepcopy(look))

    robot = deepcopy(vehicle_root.find("./robot"))
    assert robot is not None
    _replace_arguments(robot, replacements)

    # Keep exactly one externally commanded uniform field. Its scenario value
    # is always zero so reset, VINS initialization and controller preroll are
    # free of an unobserved static current bias.
    for current_field in tuple(ocean.findall("./current")):
        if current_field is not uniform_current:
            ocean.remove(current_field)
    subscribers = tuple(uniform_current.findall("./ros_subscriber"))
    if len(subscribers) != 1:
        raise ValueError("base tank must define one uniform-current ROS subscriber")
    subscribers[0].set("velocity", EXTERNAL_CURRENT_TOPIC)
    current_velocity.set("xyz", _format_vector((0.0, 0.0, 0.0)))
    center_mass = robot.find(".//internal_part[@name='WeightCenter']/mass")
    if center_mass is None:
        raise ValueError("base vehicle has no WeightCenter mass")
    new_center_mass = float(center_mass.get("value")) + domain.payload_mass_delta_kg
    if new_center_mass <= 0.0:
        raise ValueError("payload domain produced non-positive center ballast mass")
    center_mass.set("value", f"{new_center_mass:.12g}")

    for name in BALLAST_PART_NAMES:
        transform = robot.find(
            f".//internal_part[@name='{name}']/compound_transform"
        )
        if transform is None:
            raise ValueError(f"base vehicle has no {name} ballast transform")
        original = _parse_vector(str(transform.get("xyz")))
        transform.set(
            "xyz",
            _format_vector(
                tuple(
                    value + delta
                    for value, delta in zip(
                        original,
                        domain.ballast_offset_xyz_m,
                    )
                )
            ),
        )

    buoyant_dimensions = robot.findall(
        ".//internal_part[@buoyant='true']/dimensions"
    )
    if len(buoyant_dimensions) != 4:
        raise ValueError("base vehicle must define four explicit buoyant boxes")
    for dimensions in buoyant_dimensions:
        values = list(_parse_vector(str(dimensions.get("xyz"))))
        values[2] *= domain.buoyant_volume_scale
        dimensions.set("xyz", _format_vector(tuple(values)))

    external_parts = robot.findall(".//external_part")
    external_by_name = {part.get("name"): part for part in external_parts}
    if not set(EXTERNAL_DRAG_PART_NAMES).issubset(external_by_name):
        raise ValueError("base vehicle is missing expected external drag parts")
    for name in EXTERNAL_DRAG_PART_NAMES:
        part = external_by_name[name]
        hydrodynamics = part.find("./hydrodynamics")
        if hydrodynamics is None:
            hydrodynamics = ET.SubElement(part, "hydrodynamics")
        hydrodynamics.set(
            "quadratic_drag",
            _format_vector(domain.quadratic_drag_coefficient_xyz),
        )
        hydrodynamics.set(
            "viscous_drag",
            _format_vector(domain.viscous_drag_coefficient_xyz),
        )

    thrusters = robot.findall(".//actuator[@type='thruster']")
    if len(thrusters) != ACTUATOR_COUNT or tuple(
        item.get("name") for item in thrusters
    ) != THRUSTER_ORDER:
        raise ValueError("base vehicle does not match the standard six-thruster order")
    for index, thruster in enumerate(thrusters):
        rotor = thruster.find("./rotor_dynamics")
        thrust_model = thruster.find("./thrust_model")
        coefficients = thruster.find("./thrust_model/thrust_coeff")
        if rotor is None or thrust_model is None or coefficients is None:
            raise ValueError("base thruster is missing dynamics or thrust parameters")
        if thrust_model.get("type") != "fluid_dynamics":
            raise ValueError("domain renderer requires fluid_dynamics thrust models")
        rotor.attrib.clear()
        rotor[:] = []
        rotor.set("type", "first_order")
        ET.SubElement(
            rotor,
            "time_constant",
            {"value": f"{domain.rotor_tau_seconds:.12g}"},
        )
        gain = domain.thruster_gain[index]
        nominal_forward = float(coefficients.get("forward"))
        nominal_reverse = float(coefficients.get("reverse"))
        coefficients.set("forward", f"{nominal_forward * gain:.12g}")
        coefficients.set("reverse", f"{nominal_reverse * gain:.12g}")

    publisher = robot.find("./ros_publisher[@thrusters]")
    if publisher is None:
        publisher = ET.SubElement(robot, "ros_publisher")
    publisher.set(
        "thrusters",
        f"/{replacements['vehicle_name']}/simulation/thruster_state",
    )

    tank_root.remove(include)
    tank_root.append(robot)
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    ET.indent(tank_root, space="\t")
    ET.ElementTree(tank_root).write(
        temporary,
        encoding="utf-8",
        xml_declaration=True,
    )
    temporary.replace(destination_path)
    return {
        "scenario_path": str(destination_path),
    }


def _allclose_xml_vector(
    element: ET.Element | None,
    attribute: str,
    expected: tuple[float, ...],
) -> bool:
    if element is None or element.get(attribute) is None:
        return False
    try:
        actual = _parse_vector(str(element.get(attribute)), len(expected))
    except ValueError:
        return False
    return bool(np.allclose(actual, expected, rtol=0.0, atol=1e-9))


def audit_domain_scenario(
    *,
    scenario_path: str | Path,
    base_tank_path: str | Path,
    base_vehicle_path: str | Path,
    domain: PhysicsDomain,
) -> dict[str, Any]:
    """Independently verify that every declared physics value reached XML."""
    root = ET.parse(Path(scenario_path)).getroot()
    _, base_vehicle_root, _, replacements = _base_scenario_parts(
        base_tank_path,
        base_vehicle_path,
    )
    base_robot = deepcopy(base_vehicle_root.find("./robot"))
    assert base_robot is not None
    _replace_arguments(base_robot, replacements)
    robot = root.find("./robot")
    thrusters = root.findall(".//actuator[@type='thruster']")
    base_thrusters = base_robot.findall(".//actuator[@type='thruster']")
    weights = {
        name: root.find(f".//internal_part[@name='{name}']")
        for name in BALLAST_PART_NAMES
    }
    base_weights = {
        name: base_robot.find(f".//internal_part[@name='{name}']")
        for name in BALLAST_PART_NAMES
    }

    ballast_ok = True
    for name in BALLAST_PART_NAMES:
        item = weights[name]
        base_item = base_weights[name]
        if item is None or base_item is None:
            ballast_ok = False
            continue
        actual_transform = item.find("./compound_transform")
        base_transform = base_item.find("./compound_transform")
        if actual_transform is None or base_transform is None:
            ballast_ok = False
            continue
        expected = tuple(
            value + delta
            for value, delta in zip(
                _parse_vector(str(base_transform.get("xyz"))),
                domain.ballast_offset_xyz_m,
            )
        )
        ballast_ok &= _allclose_xml_vector(actual_transform, "xyz", expected)

    base_center_mass = base_robot.find(
        ".//internal_part[@name='WeightCenter']/mass"
    )
    center_mass = root.find(".//internal_part[@name='WeightCenter']/mass")
    mass_ok = bool(
        base_center_mass is not None
        and center_mass is not None
        and math.isclose(
            float(center_mass.get("value")),
            float(base_center_mass.get("value")) + domain.payload_mass_delta_kg,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )

    base_buoyant = base_robot.findall(
        ".//internal_part[@buoyant='true']/dimensions"
    )
    buoyant = root.findall(".//internal_part[@buoyant='true']/dimensions")
    buoyancy_ok = len(base_buoyant) == len(buoyant) == 4 and all(
        np.allclose(
            _parse_vector(str(actual.get("xyz"))),
            (
                _parse_vector(str(base.get("xyz")))[0],
                _parse_vector(str(base.get("xyz")))[1],
                _parse_vector(str(base.get("xyz")))[2]
                * domain.buoyant_volume_scale,
            ),
            rtol=0.0,
            atol=1e-9,
        )
        for actual, base in zip(buoyant, base_buoyant)
    )

    thruster_order_ok = tuple(item.get("name") for item in thrusters) == THRUSTER_ORDER
    actuator_ok = len(thrusters) == len(base_thrusters) == ACTUATOR_COUNT
    if actuator_ok:
        for index, (actual, base) in enumerate(zip(thrusters, base_thrusters)):
            rotor = actual.find("./rotor_dynamics")
            time_constant = actual.find("./rotor_dynamics/time_constant")
            model = actual.find("./thrust_model")
            coefficients = actual.find("./thrust_model/thrust_coeff")
            base_coefficients = base.find("./thrust_model/thrust_coeff")
            if (
                rotor is None
                or time_constant is None
                or model is None
                or coefficients is None
                or base_coefficients is None
            ):
                actuator_ok = False
                continue
            expected_forward = (
                float(base_coefficients.get("forward"))
                * domain.thruster_gain[index]
            )
            expected_reverse = (
                float(base_coefficients.get("reverse"))
                * domain.thruster_gain[index]
            )
            actuator_ok &= bool(
                rotor.get("type") == "first_order"
                and math.isclose(
                    float(time_constant.get("value")),
                    domain.rotor_tau_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and model.get("type") == "fluid_dynamics"
                and math.isclose(
                    float(coefficients.get("forward")),
                    expected_forward,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(coefficients.get("reverse")),
                    expected_reverse,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )

    drag_ok = True
    for name in EXTERNAL_DRAG_PART_NAMES:
        hydrodynamics = root.find(
            f".//external_part[@name='{name}']/hydrodynamics"
        )
        drag_ok &= _allclose_xml_vector(
            hydrodynamics,
            "quadratic_drag",
            domain.quadratic_drag_coefficient_xyz,
        ) and _allclose_xml_vector(
            hydrodynamics,
            "viscous_drag",
            domain.viscous_drag_coefficient_xyz,
        )

    current_fields = root.findall("./environment/ocean/current")
    uniform_current = root.find("./environment/ocean/current[@type='uniform']")
    current = None if uniform_current is None else uniform_current.find("./velocity")
    thruster_state_publisher = root.find("./robot/ros_publisher[@thrusters]")
    sensors = [] if robot is None else robot.findall("./sensor")
    base_sensors = base_robot.findall("./sensor")
    sensors_unchanged = [
        _xml_semantic_signature(item) for item in sensors
    ] == [_xml_semantic_signature(item) for item in base_sensors]
    checks = {
        "single_inlined_robot": robot is not None
        and len(root.findall("./robot")) == 1
        and root.find("./include") is None,
        "standard_six_thruster_order": thruster_order_ok,
        "first_order_and_directional_thrust_applied": actuator_ok,
        "payload_mass_applied": mass_ok,
        "ballast_offset_applied": ballast_ok,
        "buoyant_volume_applied": buoyancy_ok,
        "xyz_drag_coefficients_applied": drag_ok,
        "zero_static_current_configured_in_scenario": _allclose_xml_vector(
            current,
            "xyz",
            (0.0, 0.0, 0.0),
        ),
        "single_external_uniform_current": bool(
            len(current_fields) == 1
            and uniform_current is not None
            and uniform_current.find("./ros_subscriber") is not None
            and uniform_current.find("./ros_subscriber").get("velocity")
            == EXTERNAL_CURRENT_TOPIC
        ),
        "applied_thrust_truth_published": bool(
            thruster_state_publisher is not None
            and thruster_state_publisher.get("thrusters")
            == f"/{replacements['vehicle_name']}/simulation/thruster_state"
        ),
        "fluid_dynamics_model_preserved": all(
            item.find("./thrust_model").get("type") == "fluid_dynamics"
            for item in thrusters
        ),
        "sensor_configuration_unchanged": sensors_unchanged,
        "no_deadband_proxy_claimed": all(
            item.find("./thrust_model").get("type") != "deadband"
            for item in thrusters
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "pass": all(checks.values()),
        "path": str(Path(scenario_path).resolve()),
        "checks": checks,
    }


def generate_domain_bank(
    directory: str | Path,
    *,
    count: int = 160,
    seed: int = 20260802,
    spec: PhysicsDomainSpec = PhysicsDomainSpec(),
    train_count: int | None = None,
    validation_count: int | None = None,
    base_tank_path: str | Path | None = None,
    base_vehicle_path: str | Path | None = None,
    split_seeds: Mapping[str, int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one deterministic domain manifest and optional runnable scenarios."""
    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / DOMAIN_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"domain manifest already exists: {manifest_path}")
    if (base_tank_path is None) != (base_vehicle_path is None):
        raise ValueError("base tank and vehicle paths must be provided together")
    domains = sample_physics_domains(
        count=count,
        seed=seed,
        spec=spec,
        train_count=train_count,
        validation_count=validation_count,
        split_seeds=split_seeds,
    )
    entries: list[dict[str, Any]] = []
    scenarios_materialized = base_tank_path is not None
    for domain in domains:
        entry = domain.manifest_entry()
        if scenarios_materialized:
            relative_path = Path("scenarios") / f"{domain.domain_id}.scn"
            report = render_domain_scenario(
                base_tank_path=base_tank_path,
                base_vehicle_path=base_vehicle_path,
                domain=domain,
                destination=output / relative_path,
            )
            entry["scenario_path"] = str(relative_path)
        entries.append(entry)

    split_counts = {
        split: sum(domain.data_split == split for domain in domains)
        for split in DOMAIN_SPLITS
    }
    manifest: dict[str, Any] = {
        "format_version": DOMAIN_FORMAT_VERSION,
        "protocol_id": DOMAIN_BANK_PROTOCOL_ID,
        "domain_protocol_id": DOMAIN_PROTOCOL_ID,
        "sampling": {
            "method": "scrambled_sobol",
            "seed": int(seed),
            "count": int(count),
            "split_counts": split_counts,
            "spec": spec.as_dict(),
            "split_seeds": (
                {key: int(value) for key, value in split_seeds.items()}
                if split_seeds is not None
                else None
            ),
        },
        "static_feasibility_contract": spec.static_feasibility_contract(),
        "physical_semantics": {
            "actuator": (
                "first-order rotor and per-thruster directional fluid-dynamics "
                "coefficients"
            ),
            "vehicle": (
                "center ballast mass plus shared ballast offset; compound CoM "
                "and inertia are derived by Stonefish"
            ),
            "hydrodynamics": (
                "xyz coefficients on both external meshes; yaw damping is an "
                "emergent geometry-coupled response, not an independent scalar"
            ),
            "environment": (
                "zero static scenario current plus a deterministic externally "
                f"published {CURRENT_MODEL_ID} on the formal episode clock"
            ),
        },
        "excluded_from_physics_domain": {
            "deadband": (
                "Stonefish deadband and fluid_dynamics thrust models are "
                "mutually exclusive; no deadband proxy is claimed in v1"
            ),
            "independent_inertia_scale": (
                "inertia changes only through physical mass distribution"
            ),
            "independent_yaw_drag": (
                "Stonefish scenario XML exposes xyz drag coefficients only"
            ),
            "sensor": {
                "included": False,
                "protocol_id": SENSOR_ROBUSTNESS_PROTOCOL_ID,
            },
        },
        "scenarios_materialized": scenarios_materialized,
        "domains": entries,
    }
    if scenarios_materialized:
        tank = Path(base_tank_path).expanduser().resolve()
        vehicle = Path(base_vehicle_path).expanduser().resolve()
        manifest["base_scenarios"] = {
            "tank_path": str(tank),
            "vehicle_path": str(vehicle),
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = inspect_domain_bank(manifest_path)
    if not report.get("pass"):
        raise RuntimeError(f"generated domain bank failed audit: {report}")
    return manifest_path


def _resolve_archived_project_file(value: object, context_path: Path) -> Path:
    """Rebase an archived absolute path against a copied project tree."""
    raw = Path(str(value)).expanduser()
    direct = raw if raw.is_absolute() else context_path.parent / raw
    if direct.is_file():
        return direct.resolve()
    if raw.is_absolute():
        for root in context_path.resolve().parents:
            for index in reversed(
                [index for index, part in enumerate(raw.parts) if part == root.name]
            ):
                candidate = root.joinpath(*raw.parts[index + 1 :])
                if candidate.is_file():
                    return candidate.resolve()
    return direct.resolve()


def inspect_domain_bank(path: str | Path) -> dict[str, Any]:
    """Regenerate or independently audit one physics-domain bank.

    Normal training banks are regenerated from their Sobol recipe.  Evaluation
    benchmarks may instead contain a deliberately small fixed set of explicit
    domains; those are audited from their immutable records and declared IDs.
    """
    source = Path(path).expanduser().resolve()
    manifest_path = source / DOMAIN_MANIFEST_NAME if source.is_dir() else source
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sampling = manifest["sampling"]
        sampling_method = str(sampling.get("method", "scrambled_sobol"))
        spec = PhysicsDomainSpec.from_dict(sampling["spec"])
        expected_feasibility = spec.static_feasibility_contract()
        stored_feasibility = manifest["static_feasibility_contract"]
        split_counts = sampling["split_counts"]
        if sampling_method == "scrambled_sobol":
            regenerated = sample_physics_domains(
                count=int(sampling["count"]),
                seed=int(sampling["seed"]),
                spec=spec,
                train_count=int(split_counts["train"]),
                validation_count=int(split_counts["validation"]),
                split_seeds=sampling.get("split_seeds"),
            )
        elif sampling_method == "explicit_fixed_v1":
            # Explicit benchmark banks do not claim a Sobol recipe.  Their
            # records are the sampling definition and are checked below.
            regenerated = None
        else:
            raise ValueError(f"unsupported domain sampling method: {sampling_method}")
        entries = manifest["domains"]
        loaded_domains = [PhysicsDomain.from_record(entry) for entry in entries]
        ids = [domain.domain_id for domain in loaded_domains]
        if regenerated is None:
            expected_ids = [str(value) for value in sampling["domain_ids"]]
            regenerated_match = (
                len(entries) == int(sampling["count"])
                and ids == expected_ids
                and all(
                    int(domain.sample_index) == index
                    for index, domain in enumerate(loaded_domains)
                )
            )
        else:
            regenerated_match = len(entries) == len(regenerated) and all(
                domain.as_record() == expected.as_record()
                for domain, expected in zip(loaded_domains, regenerated)
            )
        parameter_records = [
            _canonical_json(entry["parameters"]) for entry in entries
        ]
        actual_split_counts = {
            split: sum(domain.data_split == split for domain in loaded_domains)
            for split in DOMAIN_SPLITS
        }
        scenario_audits: list[dict[str, Any]] = []
        base_scenarios_available = True
        if manifest["scenarios_materialized"]:
            base = manifest["base_scenarios"]
            tank_path = _resolve_archived_project_file(
                base["tank_path"], manifest_path
            )
            vehicle_path = _resolve_archived_project_file(
                base["vehicle_path"], manifest_path
            )
            base_scenarios_available = bool(
                tank_path.is_file() and vehicle_path.is_file()
            )
            for entry, domain in zip(entries, loaded_domains):
                scenario_path = manifest_path.parent / entry["scenario_path"]
                report = audit_domain_scenario(
                    scenario_path=scenario_path,
                    base_tank_path=tank_path,
                    base_vehicle_path=vehicle_path,
                    domain=domain,
                )
                scenario_audits.append(report)
        checks = {
            "manifest_protocol": (
                manifest["format_version"] == DOMAIN_FORMAT_VERSION
                and manifest["protocol_id"] == DOMAIN_BANK_PROTOCOL_ID
                and manifest["domain_protocol_id"] == DOMAIN_PROTOCOL_ID
            ),
            "deterministic_sampling_reproduced": regenerated_match,
            "sampling_method_supported": sampling_method
            in {"scrambled_sobol", "explicit_fixed_v1"},
            "domain_ids_unique": len(ids) == len(set(ids)),
            "domain_contents_unique": len(parameter_records)
            == len(set(parameter_records)),
            "split_counts_match": actual_split_counts == split_counts,
            "static_feasibility_contract": bool(
                stored_feasibility == expected_feasibility
                and expected_feasibility["pass"]
            ),
            "sensor_domain_excluded": (
                manifest["excluded_from_physics_domain"]["sensor"]["included"]
                is False
            ),
            "base_scenarios_available": base_scenarios_available,
            "materialized_scenarios_audited": (
                not manifest["scenarios_materialized"]
                or all(report["pass"] for report in scenario_audits)
            ),
        }
        checks = {name: bool(value) for name, value in checks.items()}
        return {
            "pass": all(checks.values()),
            "path": str(manifest_path),
            "checks": checks,
            "domain_count": len(loaded_domains),
            "split_counts": actual_split_counts,
            "scenario_count": len(scenario_audits),
            "static_feasibility_contract": expected_feasibility,
            "failed_scenarios": [
                report["path"]
                for report in scenario_audits
                if not report["pass"]
            ],
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {
            "pass": False,
            "path": str(manifest_path),
            "error": str(error),
        }


def load_domain_from_bank(
    path: str | Path,
    domain_id: str,
) -> tuple[PhysicsDomain, dict[str, Any]]:
    """Load one verified domain and its immutable provenance."""
    source = Path(path).expanduser().resolve()
    manifest_path = source / DOMAIN_MANIFEST_NAME if source.is_dir() else source
    report = inspect_domain_bank(manifest_path)
    if not report.get("pass"):
        raise ValueError(f"domain bank failed audit: {report}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["domains"]:
        if entry["domain_id"] == domain_id:
            domain = PhysicsDomain.from_record(entry)
            provenance = {
                "domain_protocol_id": DOMAIN_PROTOCOL_ID,
                "domain_bank_protocol_id": DOMAIN_BANK_PROTOCOL_ID,
                "domain_id": domain.domain_id,
                "domain_split": domain.data_split,
                "static_feasibility_contract": manifest[
                    "static_feasibility_contract"
                ],
                "parameters": entry["parameters"],
                "sensor_robustness_included": False,
            }
            if "scenario_path" in entry:
                provenance["scenario_path"] = str(
                    (manifest_path.parent / entry["scenario_path"]).resolve()
                )
            return domain, provenance
    raise KeyError(f"domain_id not found in bank: {domain_id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic continuous Stonefish physics-domain bank."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--base-tank")
    parser.add_argument("--base-vehicle")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="write one fixed nominal first-order domain (no randomization)",
    )
    parser.add_argument(
        "--current-amplitude-min",
        type=float,
        help="override the external-current amplitude lower bound in m/s",
    )
    parser.add_argument(
        "--current-amplitude-max",
        type=float,
        help="override the external-current amplitude upper bound in m/s",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(args)
    tank = None if options.manifest_only else options.base_tank
    vehicle = None if options.manifest_only else options.base_vehicle
    if not options.manifest_only and (not tank or not vehicle):
        parser.error("--base-tank and --base-vehicle are required unless --manifest-only")
    current_bounds = (options.current_amplitude_min, options.current_amplitude_max)
    if (options.current_amplitude_min is None) != (options.current_amplitude_max is None):
        parser.error("current-amplitude bounds must be provided together")
    if options.nominal and options.current_amplitude_min is not None:
        parser.error("--nominal cannot be combined with current-amplitude overrides")
    if options.nominal:
        spec = PhysicsDomainSpec(
            rotor_tau_seconds=(0.10, 0.10),
            global_thrust_gain=(1.0, 1.0),
            thruster_gain_residual=(0.0, 0.0),
            payload_mass_delta_kg=(0.0, 0.0),
            ballast_offset_x_m=(0.0, 0.0),
            ballast_offset_y_m=(0.0, 0.0),
            ballast_offset_z_m=(0.0, 0.0),
            net_buoyancy_force_n=(0.0, 0.0),
            quadratic_drag_coefficient=(0.85, 0.85),
            current_amplitude_m_s=(0.0, 0.0),
            current_direction_rad=(0.0, 0.0),
            current_period_seconds=(30.0, 30.0),
            current_phase_rad=(0.0, 0.0),
        )
        count = 1
        train_count = 1
        validation_count = 0
    else:
        spec = (
            PhysicsDomainSpec(current_amplitude_m_s=current_bounds)
            if options.current_amplitude_min is not None
            else PhysicsDomainSpec()
        )
        count = options.count
        train_count = options.train_count
        validation_count = options.validation_count
    path = generate_domain_bank(
        options.output_dir,
        count=count,
        seed=options.seed,
        train_count=train_count,
        validation_count=validation_count,
        spec=spec,
        base_tank_path=tank,
        base_vehicle_path=vehicle,
        overwrite=options.overwrite,
    )
    print(json.dumps(inspect_domain_bank(path), indent=2, sort_keys=True))
    return 0


def inspect_main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a physics-domain bank.")
    parser.add_argument("path")
    options = parser.parse_args(args)
    report = inspect_domain_bank(options.path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
