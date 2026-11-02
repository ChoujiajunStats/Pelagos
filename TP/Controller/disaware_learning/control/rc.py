"""ArduSub MANUAL RC conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from disaware_learning.control.contract import (
    MANUAL_CONTROL_ORDER,
    enforce_manual_authority,
)


PWM_MIN = 1100
PWM_TRIM = 1500
PWM_MAX = 1900
PWM_SCALE = 400


def clamp_manual_control(values: Iterable[float], limit: float = 1.0) -> np.ndarray:
    """Return six finite MANUAL axes clipped to a symmetric limit."""
    axes = np.asarray(list(values), dtype=np.float64)
    if axes.shape != (len(MANUAL_CONTROL_ORDER),):
        raise ValueError(
            f"expected {len(MANUAL_CONTROL_ORDER)} MANUAL axes, got {axes.shape}"
        )
    if not np.all(np.isfinite(axes)):
        raise ValueError("MANUAL axes must be finite")
    bounded_limit = max(0.0, min(1.0, float(limit)))
    return enforce_manual_authority(np.clip(axes, -bounded_limit, bounded_limit))


def normalized_to_pwm(values: Iterable[float]) -> np.ndarray:
    """Map normalized MANUAL axes to ArduSub reversible PWM."""
    axes = clamp_manual_control(values)
    return np.rint(PWM_TRIM + PWM_SCALE * axes).astype(np.int32)


def pwm_to_normalized(values: Iterable[float]) -> np.ndarray:
    """Map six RC or motor PWM values into [-1, 1]."""
    pwm = np.asarray(list(values), dtype=np.float64)
    if pwm.shape != (len(MANUAL_CONTROL_ORDER),):
        raise ValueError(f"expected six PWM channels, got {pwm.shape}")
    if not np.all(np.isfinite(pwm)):
        raise ValueError("PWM values must be finite")
    return np.clip((pwm - PWM_TRIM) / PWM_SCALE, -1.0, 1.0)


def expected_manual_pwm(values: Iterable[float]) -> np.ndarray:
    """Return the eight-channel RC packet for one MANUAL control sample."""
    axes = clamp_manual_control(values)
    expected = np.full(8, float(PWM_TRIM), dtype=np.float64)
    expected[:6] = np.rint(PWM_TRIM + PWM_SCALE * axes)
    return expected


def manual_pair_consistent(normalized: Iterable[float], pwm: Iterable[float]) -> bool:
    """Check that normalized MANUAL axes and an eight-channel PWM packet agree."""
    values = np.asarray(normalized, dtype=np.float64)
    channels = np.asarray(pwm, dtype=np.float64)
    return bool(
        values.shape == (6,)
        and channels.shape == (8,)
        and np.all(np.isfinite(values))
        and np.all(np.isfinite(channels))
        and np.array_equal(channels, expected_manual_pwm(values))
    )


def override_channels(values: Iterable[float]) -> tuple[int, ...]:
    """Build RC_CHANNELS_OVERRIDE channels 1..8 from MANUAL axes."""
    return tuple(expected_manual_pwm(values).astype(np.int32).tolist())


def slew_towards(
    current: Iterable[float],
    target: Iterable[float],
    max_rate: float,
    dt: float,
) -> np.ndarray:
    """Apply a per-axis normalized command-rate limit."""
    current_axes = clamp_manual_control(current)
    target_axes = clamp_manual_control(target)
    max_step = max(0.0, float(max_rate)) * max(0.0, float(dt))
    return current_axes + np.clip(target_axes - current_axes, -max_step, max_step)
