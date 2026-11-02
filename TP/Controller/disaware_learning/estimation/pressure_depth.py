"""Pressure-sensor conversion shared by simulation collection and control."""

from __future__ import annotations

import math

STANDARD_GRAVITY_M_S2 = 9.80665
PRESSURE_DEPTH_SOURCE_ID = "ardusub_fluid_pressure_hydrostatic_v1"
PRESSURE_DEPTH_CONVERSION_ID = "depth_down=(pressure-surface_pressure)/(density*g)"


def pressure_to_depth_down_m(
    pressure_pa: float,
    *,
    fluid_density_kg_m3: float,
    surface_pressure_pa: float = 0.0,
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
) -> float:
    """Convert hydrostatic pressure to positive-down depth."""

    values = (
        float(pressure_pa),
        float(fluid_density_kg_m3),
        float(surface_pressure_pa),
        float(gravity_m_s2),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pressure-depth conversion requires finite values")
    if fluid_density_kg_m3 <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("fluid density and gravity must be positive")
    return (pressure_pa - surface_pressure_pa) / (fluid_density_kg_m3 * gravity_m_s2)
