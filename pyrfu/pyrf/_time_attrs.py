#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.13"
__status__ = "Prototype"

# The rule shared by time_clip, resample and ts_append to tell the global
# attributes with one value (or row) per time from the others. The tables
# describe the particle distributions of pyrfu (pyrf.ts_skymap and the
# routines built on it), whose attribute names follow the MMS files; any
# other attribute is taken as per-time when its first dimension is the
# length of time.

# Attributes of the distributions (skymaps, pitch-angle and gyrophase
# distributions) with one value (or row) per time, and their number of
# dimensions when they do: the step table and the time widths (time,), the
# energy widths (time, energy) for FPI and FEEPS (per channel, (energy,), for
# EIS), the azimuthal widths (time, phi) and the FEEPS coverage (time, energy)
_TIME_ATTRS = {
    "esteptable": 1,
    "delta_time_minus": 1,
    "delta_time_plus": 1,
    "delta_energy_minus": 2,
    "delta_energy_plus": 2,
    "delta_phi_minus": 2,
    "delta_phi_plus": 2,
    "coverage": 2,
}

# Attributes of the distributions per channel or angular bin, never clipped
# (they were when their length was that of time, e.g. the energy tables of 32
# channels with 32 times)
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
