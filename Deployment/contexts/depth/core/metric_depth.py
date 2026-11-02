"""沿用离线基线的视差到米制光轴 Z 深度转换。"""
import numpy as np


def mapping_depth(disparity, fx, baseline_m, min_depth_m=0.30):
    """Return optical Z without a far cap; invalid/sub-minimum values stay NaN.

    Approved 2026-11-01: retain finite far measurements for the mapper's native
    free-space beam policy. The mapper owns its update range; do not clamp Z.
    """
    if (not np.all(np.isfinite([fx, baseline_m, min_depth_m]))
            or fx <= 0 or baseline_m <= 0 or min_depth_m <= 0):
        raise ValueError('焦距、基线或最小深度无效')
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        np.divide(fx * baseline_m, disparity, out=depth, where=valid)
    valid &= np.isfinite(depth) & (depth >= min_depth_m)
    depth[~valid] = np.nan
    return depth, valid


def metric_depth(disparity, fx, baseline_m, min_depth_m=0.30, max_depth_m=12.0):
    if (not np.all(np.isfinite([fx, baseline_m, min_depth_m, max_depth_m]))
            or fx <= 0 or baseline_m <= 0 or not (0 < min_depth_m < max_depth_m)):
        raise ValueError('焦距、基线或深度范围无效')
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        np.divide(fx * baseline_m, disparity, out=depth, where=valid)
    valid &= np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    depth[~valid] = np.nan
    return depth, valid
