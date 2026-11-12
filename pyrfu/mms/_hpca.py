#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from .db_get_variable import db_get_variable

# Shared by get_hpca_dist and hpca_pad (hpca_pad imported it from
# get_hpca_dist, which imports hpca_pad: a cyclic import; get_hpca_dist
# imported _hpca_elevations from hpca_pad)


def _half_spins(inp, azimuth):
    r"""First sample of inp of each half-spin of azimuth, and whether the
    half-spin is complete in inp with non-zero azimuths."""
    t_inp = inp.time.data.astype("datetime64[ns]")
    t_azi = (azimuth.time.data.astype("datetime64[ns]") - t_inp[0]).astype(np.int64)
    t_inp = (t_inp - t_inp[0]).astype(np.int64)
    n_az = azimuth.shape[1]
    d_t = np.median(np.diff(t_inp))

    # The azimuth times are the centres of the half-spins, 7.5 samples after
    # their first samples (times shifted to the middle of the accumulations, as
    # by get_ts and get_data), or their starts (Epoch_Angles in the files)
    for offset in [(n_az - 1) / 2 * d_t, 0.0]:
        t_first = t_azi - offset
        i_start = np.searchsorted(t_inp, t_first - d_t / 4)
        i_first = np.minimum(i_start, len(t_inp) - 1)
        complete = np.abs(t_inp[i_first] - t_first) < d_t / 4

        if np.any(complete):
            break

    i_last = np.minimum(i_start + n_az - 1, len(t_inp) - 1)
    complete &= i_start + n_az <= len(t_inp)
    complete &= np.abs(t_inp[i_last] - t_inp[i_first] - (n_az - 1) * d_t) < d_t / 2

    # Half-spins without azimuths (zeros), from the first energy and anode
    complete &= np.any(azimuth.data[:, :, 0, 0] != 0, axis=1)

    return i_start, complete


def _hpca_elevations(vdf, source: str = "default", data_path: str = ""):
    r"""Polar angles (colatitudes) of the 16 HPCA anodes, read from
    mms?_hpca_centroid_elevation_angle in the file of `vdf`."""
    try:
        dataset = vdf.attrs["GLOBAL"]["Logical_source"]  # e.g., mms1_hpca_brst_l2_ion
    except (KeyError, TypeError) as err:
        raise ValueError(
            "Can't find the HPCA dataset of vdf (GLOBAL Logical_source attribute): "
            "give the anode elevations with elevation"
        ) from err

    cdf_name = f"{dataset.split('_')[0]}_hpca_centroid_elevation_angle"
    tint = list(datetime642iso8601(vdf.time.data[[0, -1]]))
    elevation = db_get_variable(
        dataset, cdf_name, tint, verbose=False, data_path=data_path, source=source
    )

    return np.asarray(elevation.data, dtype=np.float64).ravel()
