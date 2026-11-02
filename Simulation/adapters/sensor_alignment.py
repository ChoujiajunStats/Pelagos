"""Separate sensor-stream liveness from acquisition-time causal selection."""
import inspect
import math


def aligned_sensor_window(source_class):
    """Retain the source window and its age limit, with causal aligned lookup.

    A delayed estimator pose can need a sample received before the latest live
    sample. Stream freshness uses the latest accepted receipt; selected values
    still come exclusively from the requested acquisition time or earlier.
    """
    default_age = inspect.signature(source_class.at).parameters['max_age'].default
    if (default_age is inspect.Parameter.empty or
            not math.isfinite(default_age) or default_age <= 0):
        raise ValueError('Source SensorWindow.at must declare a finite positive max_age')

    class AlignedSensorWindow(source_class):
        source_default_max_age = default_age

        def _aligned_sample(self, stamp, now_ns, max_age):
            valid_request = math.isfinite(stamp)
            valid_age = math.isfinite(max_age) and max_age > 0
            latest = self.samples[-1] if self.samples else None
            causal = (next((sample for sample in reversed(self.samples) if sample[0] <= stamp), None)
                      if valid_request else None)
            latest_age_ns = now_ns-latest[2] if latest else None
            selected_age_ns = now_ns-causal[2] if causal else None
            gap = stamp-causal[0] if causal else None
            diagnostic = dict(requested_stamp_s=stamp if valid_request else None,
                              max_age_s=max_age if valid_age else None,
                              causal_stamp_s=causal[0] if causal else None,
                              causal_source_gap_s=gap,
                              selected_receive_age_s=selected_age_ns*1e-9 if causal else None,
                              latest_stamp_s=latest[0] if latest else None,
                              latest_receive_age_s=latest_age_ns*1e-9 if latest else None,
                              ready=False, reason='')
            if not valid_request:
                reason = 'invalid_requested_stamp'
            elif not valid_age:
                reason = 'invalid_max_age'
            elif latest is None:
                reason = 'no_samples'
            elif latest_age_ns < 0:
                reason = 'latest_receive_clock_negative'
            elif latest_age_ns > int(max_age*1e9):
                reason = 'latest_stream_stale'
            elif causal is None:
                reason = 'no_causal_sample'
            elif selected_age_ns < 0:
                reason = 'selected_receive_clock_negative'
            elif gap > max_age:
                reason = 'causal_source_gap_exceeded'
            else:
                reason = 'ready'
            diagnostic.update(ready=reason == 'ready', reason=reason)
            return causal, diagnostic

        def at(self, stamp, now_ns, max_age=default_age):
            causal, diagnostic = self._aligned_sample(stamp, now_ns, max_age)
            return causal[1] if diagnostic['ready'] else None

        def alignment_diagnostic(self, stamp, now_ns, max_age=default_age):
            """JSON-compatible timing metadata; never includes sensor values."""
            return self._aligned_sample(stamp, now_ns, max_age)[1]

    return AlignedSensorWindow
