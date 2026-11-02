"""Depth validity and error calculations in metres."""
import numpy as np


def compare_depths(gt, estimate, minimum, maximum):
    if gt.shape != estimate.shape:
        raise ValueError('GT and estimated depth sizes differ')
    valid_gt = np.isfinite(gt) & (gt >= minimum) & (gt <= maximum)
    valid_est = np.isfinite(estimate) & (estimate >= minimum) & (estimate <= maximum)
    valid = valid_gt & valid_est
    if not valid.any():
        raise ValueError('no overlapping valid depths')
    error = np.abs(estimate - gt)
    values = error[valid]
    metrics = dict(samples=int(valid.sum()), valid_fraction=float(valid.mean()),
                   mae_m=float(values.mean()), rmse_m=float(np.sqrt(np.mean(values ** 2))),
                   median_abs_error_m=float(np.median(values)))
    return valid_gt, valid_est, valid, error, metrics
