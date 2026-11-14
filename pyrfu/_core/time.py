#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""The time conversions of pyrfu, in one place (they were written inline, in
several ways, with different rounding).

The times are numpy.datetime64 of any unit, converted to ns first; the
seconds come from the exact integer steps in ns divided by 1e9 (correctly
rounded; ``* 1e-9`` is not), and the durations in seconds go to ns by
rounding (truncating gave e.g. 15699999 ns for 0.0157 s). NaT gives NaN and
NaN gives NaT.
"""

# 3rd party imports
import numpy as np

_EPOCH = np.datetime64(0, "ns")


def _ns(time):
    r"""Times as int64 nanoseconds since 1970-01-01."""
    return np.asarray(time).astype("datetime64[ns]").astype(np.int64)


def _seconds(time, t_ref):
    r"""Seconds from `t_ref` to `time` (datetime64, str or array_like of them),
    from the exact steps in ns; NaN for NaT."""
    time = np.asarray(time).astype("datetime64[ns]")
    t_ref = np.asarray(t_ref).astype("datetime64[ns]")
    return (time - t_ref) / np.timedelta64(1, "s")


def _unix(time):
    r"""Unix times (seconds since 1970-01-01, without leap seconds); NaN for
    NaT."""
    return _seconds(time, _EPOCH)


def _timedelta(seconds):
    r"""Durations in seconds as timedelta64[ns], to the nearest ns; NaT for
    NaN."""
    seconds = np.asarray(seconds, dtype=np.float64)
    nan = np.isnan(seconds)
    t_ns = np.round(np.where(nan, 0.0, seconds) * 1e9).astype(np.int64)
    return np.where(nan, np.timedelta64("NaT", "ns"), t_ns.astype("timedelta64[ns]"))


def _datetime64(unix):
    r"""Unix times (s) as datetime64[ns], to the nearest ns; NaT for NaN."""
    return _EPOCH + _timedelta(unix)
