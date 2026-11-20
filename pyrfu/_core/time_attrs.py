#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Per-time global attributes, shared by time_clip, resample and ts_append.
# The tables follow the MMS attribute names of the distributions
# (pyrf.ts_skymap); any other attribute is per-time when its first dimension
# is the length of time.

# Per-time attributes of the distributions and their number of dimensions:
# step table, time widths, compression indicator and quality flags (time,),
# energy widths and FEEPS coverage (time, energy), azimuthal widths (time,
# phi); the EIS energy widths are per channel, (energy,)
_TIME_ATTRS = {
    "esteptable": 1,
    "compressionloss": 1,
    "errorflags": 1,
    "delta_time_minus": 1,
    "delta_time_plus": 1,
    "delta_energy_minus": 2,
    "delta_energy_plus": 2,
    "delta_phi_minus": 2,
    "delta_phi_plus": 2,
    "coverage": 2,
}

# Per-bin attributes of the distributions, never clipped even when their
# length is that of time (e.g. 32 energy channels and 32 times)
_BIN_ATTRS = (
    "energy0",
    "energy1",
    "delta_theta_minus",
    "delta_theta_plus",
    "delta_pitchangle_minus",
    "delta_pitchangle_plus",
    "delta_gyrophase_minus",
    "delta_gyrophase_plus",
)


def _is_time_attr(key, attr, n_t):
    r"""True if the global attribute `key` of a Dataset of `n_t` times has one
    value (or row) per time: the per-time attributes of the distributions
    with their number of dimensions, never their per-bin ones, else the
    array_like attributes whose first dimension is the length of time."""
    if key in _BIN_ATTRS or not isinstance(attr, np.ndarray) or attr.ndim == 0:
        return False

    if key in _TIME_ATTRS:
        return attr.ndim == _TIME_ATTRS[key] and attr.shape[0] == n_t

    return attr.shape[0] == n_t
