"""Aligned sample-buffer construction for episode protocol v15."""

from __future__ import annotations

from disaware_learning.data.episode_schema import (
    ALIGNED_ARRAYS,
    GROUND_TRUTH_ARRAYS,
    PRIVILEGED_ACTUATOR_ARRAYS,
)


def create_sample_buffers(*, record_ground_truth: bool) -> dict[str, list]:
    """Create fresh aligned buffers for one episode."""
    names = list(ALIGNED_ARRAYS)
    if record_ground_truth:
        names.extend(GROUND_TRUTH_ARRAYS)
        names.extend(PRIVILEGED_ACTUATOR_ARRAYS)
    if len(names) != len(set(names)):
        raise RuntimeError("episode schema contains duplicate sample arrays")
    return {name: [] for name in names}
