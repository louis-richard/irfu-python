#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contributed by Apostolos Kolokotronis."""

from collections.abc import Sequence

# 3rd party imports
import numpy as np
import xarray as xr
from xarray import DataArray

# Local imports
from pyrfu._core.time import _ns
from pyrfu._core.units import _check_same_units

from .calc_fs import calc_fs
from .resample import resample


def _valid_count(inp):
    r"""1 where `inp` (array_like or DataArray) is not NaN, 0 where it is, as a
    numpy.ndarray of int (xr.where returned a numpy.ndarray for an array,
    whose ``.data`` taken by feeps_avg_4sc was a memoryview)."""
    return (~np.isnan(np.asarray(inp))).astype(np.int64)


def _without_gaps(inp, inp_r):
    r"""`inp_r` (`inp` resampled) without the values invented by the
    resampling. A time is covered if a sample of `inp` is within half its
    sampling period (the median step); NaN otherwise (data gaps, before the
    start and after the end of `inp`). A covered time that is not between two
    samples of `inp` at most 1.5 periods apart (a gap edge, an end) takes the
    nearest sample, not a value interpolated across the gap."""
    time = _ns(inp.time.data).astype(np.float64)
    ref = _ns(inp_r.time.data).astype(np.float64)
    steps = np.diff(time) if len(time) > 1 else np.diff(ref)
    d_t = np.median(steps) if len(steps) else 0.0

    i_r = np.searchsorted(time, ref)
    i_l = np.clip(i_r - 1, 0, len(time) - 1)
    i_rc = np.clip(i_r, 0, len(time) - 1)
    dist_l, dist_r = np.abs(ref - time[i_l]), np.abs(time[i_rc] - ref)
    i_near = np.where(dist_l <= dist_r, i_l, i_rc)

    covered = np.minimum(dist_l, dist_r) <= d_t / 2.0
    between = (i_r > 0) & (i_r < len(time)) & (time[i_rc] - time[i_l] <= 1.5 * d_t)
    nearest = covered & ~between

    data = inp_r.data.astype(np.float64)
    data[nearest] = inp.data[i_near[nearest]]
    data[~covered] = np.nan

    return inp_r.copy(data=data)


def nanavg_4sc(b_list: Sequence[DataArray]) -> DataArray:
    r"""Average data from 4 spacecrafts while ignoring NaN values.
    Computes the input quantity at the center of mass of the MMS
    tetrahedron. When averaging, NaN values are ignored by counting the number of
    non-NaN values at each time step.

    Parameters
    ----------
    b_list : Sequence of DataArray or Dataset
        List of the time series of the quantity for each spacecraft.

    Returns
    -------
    b_avg : DataArray or Dataset
        Time series of the input quantity a the enter of mass of the
        MMS tetrahedron.

    Raises
    ------
    TypeError
        If b_list is not a list or a tuple of DataArray or Dataset (a tuple
        was rejected).
    ValueError
        If the spacecraft have different UNITS (they were averaged).

    Notes
    -----
    Each spacecraft is resampled to the times of the first one, NaN in its
    data gaps and outside its time range, so that the average there is that
    of the spacecraft with data, as `pyrfu.mms.feeps_avg_4sc` (the
    resampling filled the gaps and extrapolated past the ends, averaged as
    data). The result is labelled "4sc_avg" (probe, mms, MMS and mmsId).

    """
    # Check input type
    if not isinstance(b_list, (list, tuple)):
        raise TypeError("b_list must be a list or a tuple")

    b_list_r = []

    for b in b_list:
        if not isinstance(b, (xr.DataArray, xr.Dataset)):
            raise TypeError("elements of b_list must be DataArray or Dataset")

        # On the times of the first spacecraft, NaN where this one has no data
        b_r = resample(b, b_list[0], f_s=calc_fs(b_list[0]))

        if isinstance(b, xr.Dataset):
            b_list_r.append(b_r.assign(data=_without_gaps(b["data"], b_r["data"])))
        else:
            b_list_r.append(_without_gaps(b, b_r))

    _check_same_units(b_list, "spacecraft")

    # Values to average: the data of a DataArray, or the "data" variable of a
    # Dataset (e.g. a skymap, whose other variables are taken from the first)
    values = [
        np.asarray(b.data if isinstance(b, xr.DataArray) else b["data"].data)
        for b in b_list_r
    ]

    b_avg_data = np.zeros(values[0].shape)
    b_nan_denom = np.zeros(values[0].shape)

    for value in values:
        # Count the valid samples before replacing NaNs by zeros in the sum
        b_nan_denom += _valid_count(value)
        b_avg_data += np.nan_to_num(value, nan=0.0)

    # Copy so that the attributes of the caller's first time series are unchanged
    attrs = dict(b_list[0].attrs)

    for key in ["probe", "mms", "MMS", "mmsId"]:
        if key in attrs:
            attrs[key] = "4sc_avg"

    # NaN where no spacecraft has data
    with np.errstate(invalid="ignore", divide="ignore"):
        b_avg_data = b_avg_data / b_nan_denom

    if isinstance(b_list_r[0], xr.DataArray):
        b_avg = xr.DataArray(
            b_avg_data,
            coords=b_list_r[0].coords,
            dims=b_list_r[0].dims,
            attrs=attrs,
        )
    else:
        b_avg = b_list_r[0].assign(data=b_list_r[0]["data"].copy(data=b_avg_data))
        b_avg.attrs = attrs

    return b_avg
