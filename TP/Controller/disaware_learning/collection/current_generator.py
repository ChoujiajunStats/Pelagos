"""Deterministic external-current profiles for Stonefish episodes.

The release profile is a bounded low-frequency two-tone current with a small
slow direction drift.  It remains horizontal, zero-bias, and deterministic on
the formal episode clock so a domain/reference pair is reproducible across
runs and concurrency levels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CURRENT_MODEL_ID = "external_ramped_multisine_drifting_direction_v1"
CURRENT_SECONDARY_WEIGHT = 0.25
CURRENT_SECONDARY_PERIOD_FRACTION = 0.60
CURRENT_DIRECTION_WOBBLE_AMPLITUDE_RAD = math.radians(12.0)
CURRENT_DIRECTION_WOBBLE_PERIOD_MIN_SECONDS = 60.0
CURRENT_DIRECTION_WOBBLE_PERIOD_FRACTION = 1.75
CURRENT_SECONDARY_PHASE_OFFSET_RAD = math.pi / 2.0


@dataclass(frozen=True)
class ExternalCurrentGenerator:
    """Generate a bounded, horizontal, zero-bias release current profile."""

    amplitude_m_s: float
    direction_rad: float
    period_seconds: float
    phase_rad: float
    ramp_seconds: float
    # Release v1 stores these values explicitly in each domain record.  The
    # optional defaults keep small unit-test/proxy constructions deterministic
    # without making older in-memory call sites carry release metadata.
    secondary_period_seconds_override: float | None = None
    secondary_phase_rad_override: float | None = None

    def __post_init__(self) -> None:
        finite = (
            self.amplitude_m_s,
            self.direction_rad,
            self.period_seconds,
            self.phase_rad,
            self.ramp_seconds,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("current generator parameters must be finite")
        if self.amplitude_m_s < 0.0:
            raise ValueError("current amplitude cannot be negative")
        if self.period_seconds <= 0.0:
            raise ValueError("current period must be positive")
        if self.ramp_seconds <= 0.0:
            raise ValueError("current ramp must be positive")
        for value, name in (
            (self.secondary_period_seconds_override, "secondary period"),
            (self.secondary_phase_rad_override, "secondary phase"),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"current {name} must be finite")
        if (
            self.secondary_period_seconds_override is not None
            and self.secondary_period_seconds_override <= 0.0
        ):
            raise ValueError("secondary current period must be positive")

    @property
    def secondary_period_seconds(self) -> float:
        """Return the deterministic low-frequency secondary period."""

        if self.secondary_period_seconds_override is not None:
            return float(self.secondary_period_seconds_override)
        return max(20.0, CURRENT_SECONDARY_PERIOD_FRACTION * self.period_seconds)

    @property
    def secondary_phase_rad(self) -> float:
        if self.secondary_phase_rad_override is not None:
            return float(self.secondary_phase_rad_override)
        return self.phase_rad + CURRENT_SECONDARY_PHASE_OFFSET_RAD

    @property
    def direction_wobble_amplitude_rad(self) -> float:
        return CURRENT_DIRECTION_WOBBLE_AMPLITUDE_RAD

    @property
    def direction_wobble_period_seconds(self) -> float:
        return max(
            CURRENT_DIRECTION_WOBBLE_PERIOD_MIN_SECONDS,
            CURRENT_DIRECTION_WOBBLE_PERIOD_FRACTION * self.period_seconds,
        )

    @property
    def direction_wobble_phase_rad(self) -> float:
        return self.secondary_phase_rad

    def sample_ned_m_s(self, elapsed_seconds: float | np.ndarray) -> np.ndarray:
        """Return current samples with shape ``time_shape + (3,)``.

        ``elapsed_seconds`` is an episode-relative formal time.  Negative or
        non-finite values are rejected instead of silently extrapolating a
        disturbance before the episode starts.
        """

        elapsed = np.asarray(elapsed_seconds, dtype=np.float64)
        if not np.all(np.isfinite(elapsed)) or np.any(elapsed < 0.0):
            raise ValueError("current elapsed time must be finite and nonnegative")
        ramp = np.minimum(1.0, elapsed / self.ramp_seconds)
        envelope = 0.5 - 0.5 * np.cos(np.pi * ramp)
        primary = np.sin(
            2.0 * np.pi * elapsed / self.period_seconds + self.phase_rad
        )
        secondary = np.sin(
            2.0 * np.pi * elapsed / self.secondary_period_seconds
            + self.secondary_phase_rad
        )
        scalar = envelope * self.amplitude_m_s * (
            (1.0 - CURRENT_SECONDARY_WEIGHT) * primary
            + CURRENT_SECONDARY_WEIGHT * secondary
        )
        direction = self.direction_rad + self.direction_wobble_amplitude_rad * np.sin(
            2.0 * np.pi * elapsed / self.direction_wobble_period_seconds
            + self.direction_wobble_phase_rad
        )
        direction_vector = np.stack(
            (np.cos(direction), np.sin(direction), np.zeros_like(direction)),
            axis=-1,
        )
        return scalar[..., None] * direction_vector

    def velocity_ned_m_s(self, elapsed_seconds: float) -> tuple[float, float, float]:
        """Return one current sample as a tuple for ROS message publication."""

        sample = self.sample_ned_m_s(float(elapsed_seconds))
        return tuple(float(value) for value in sample.reshape(3))

    def as_dict(self) -> dict[str, float | str]:
        """Return the reproducible generator parameters for provenance."""

        return {
            "current_model": CURRENT_MODEL_ID,
            "current_amplitude_m_s": float(self.amplitude_m_s),
            "current_direction_rad": float(self.direction_rad),
            "current_period_seconds": float(self.period_seconds),
            "current_phase_rad": float(self.phase_rad),
            "current_ramp_seconds": float(self.ramp_seconds),
            "current_secondary_period_seconds": float(
                self.secondary_period_seconds
            ),
            "current_secondary_phase_rad": float(self.secondary_phase_rad),
            "current_secondary_weight": float(CURRENT_SECONDARY_WEIGHT),
            "current_direction_wobble_amplitude_rad": float(
                self.direction_wobble_amplitude_rad
            ),
            "current_direction_wobble_period_seconds": float(
                self.direction_wobble_period_seconds
            ),
            "current_direction_wobble_phase_rad": float(
                self.direction_wobble_phase_rad
            ),
        }
