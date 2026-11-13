#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Helpers shared by the EPD routines (EIS, FEEPS, HPCA): spin averages, the
HPCA half-spins and anode elevations, and the attributes and units of the EIS
spectra."""

# Built-in imports
import warnings

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.mms.db_get_variable import db_get_variable
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601


def _spin_reduce(data, starts, stops, method: str = "mean"):
    r"""Mean or sum of `data` along its first axis (time) over each spin
    [starts[i], stops[i]), NaN-aware: NaN for a spin without data (also for
    the sum, which was 0, as data). The spins have the same number of samples,
    as the complete spins of EIS, FEEPS and HPCA (one array for all of them; it
    was a loop over the spins in each routine)."""
    data = np.asarray(data)
    starts, stops = np.asarray(starts), np.asarray(stops)

    if len(starts) == 0:
        return np.empty((0, *data.shape[1:]), dtype=np.float64)

    length = stops[0] - starts[0]

    if np.any(stops - starts != length):
        raise ValueError("the spins must have the same number of samples")

    # (spin, sample in the spin, ...)
    spins = data[starts[:, None] + np.arange(length)]

    if method == "sum" and spins.dtype.kind in "iub":
        return spins.sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method == "sum":
            return np.where(
                np.isfinite(spins).any(axis=1), np.nansum(spins, axis=1), np.nan
            )

        return np.nanmean(spins, axis=1)


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


def _is_nan(attr):
    r"""True for a NaN or NaT scalar, False for any other value."""
    try:
        return bool(np.isnan(attr))
    except (TypeError, ValueError):
        return False


def _equal(attr0, attr1):
    r"""True if the attribute values are equal, arrays (and dicts or lists of
    them) included: == compares arrays element by element, which raised
    ValueError in a condition. NaN and NaT (fill values) equal themselves."""
    if isinstance(attr0, dict) and isinstance(attr1, dict):
        return attr0.keys() == attr1.keys() and all(
            _equal(attr0[k], attr1[k]) for k in attr0
        )

    if isinstance(attr0, (list, tuple)) and isinstance(attr1, (list, tuple)):
        return len(attr0) == len(attr1) and all(
            _equal(a_0, a_1) for a_0, a_1 in zip(attr0, attr1)
        )

    if isinstance(attr0, np.ndarray) or isinstance(attr1, np.ndarray):
        try:
            return bool(np.array_equal(attr0, attr1, equal_nan=True))
        except TypeError:  # no NaN in strings and objects
            return bool(np.array_equal(attr0, attr1))

    return bool(attr0 == attr1) or (_is_nan(attr0) and _is_nan(attr1))


def _common_attrs(attrs_list, skip=()):
    r"""Attributes of the first dict of `attrs_list`, except those in `skip`,
    with the same value in all the others."""
    return {
        k: v
        for k, v in attrs_list[0].items()
        if k not in skip
        and all(k in attrs and _equal(attrs[k], v) for attrs in attrs_list[1:])
    }


# Shared by the EIS routines for fluxes only (eis_combine_proton_spec,
# eis_moments, eis_proton_correction) and by the combinations of spacecraft


def _check_flux(*inps):
    r"""ValueError unless the inputs (DataArrays) have the same units, a
    differential particle flux, as the flux of the EIS files,
    1/(cm^2 s sr keV) (counts are ' ', cps 1/s). Inputs without UNITS (built
    by hand) are not checked."""
    units = sorted({inp.attrs["UNITS"] for inp in inps if "UNITS" in inp.attrs})

    if len(units) > 1:
        raise ValueError(f"the inputs have different units: {units}")

    if units and "cm^2" not in units[0]:
        raise ValueError(
            f"the data must be a flux, 1/(cm^2 s sr keV), not {units[0]!r}"
        )


def _check_same_attrs(inps, keys, name):
    r"""ValueError unless the inputs (DataArrays) have the same attributes
    `keys` (e.g. the species, the method of the PADs), or none of them (they
    were combined whatever they were)."""
    for key in keys:
        values = sorted({str(inp.attrs.get(key)) for inp in inps})

        if len(values) > 1:
            raise ValueError(f"The {name} have different {key}: {values}")
