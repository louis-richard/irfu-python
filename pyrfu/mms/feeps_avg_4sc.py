#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from copy import deepcopy
from typing import Sequence

import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.resample import resample
from ._eis_units import _check_same_units

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolos.kolokotronis@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.5.10"
__status__ = "Prototype"


def _nan_count(inp):
    r"""Counts the number of non-NaN values in the input array at each time step.

    Parameters
    ----------
    inp : DataArray or Dataset
        Input array.

    Returns
    -------
    inp_to_counts : DataArray or Dataset
        Array with the same shape as the input array, where each element
        is the number of non-NaN values at the corresponding time step.

    """
    inp_to_counts = xr.where(np.isnan(inp), 0, 1)

    return inp_to_counts


def _ns(time):
    r"""Times as float nanoseconds."""
    return time.astype("datetime64[ns]").astype(np.int64).astype(np.float64)


def _without_gaps(inp, inp_r):
    r"""`inp_r` (`inp` resampled) without the values invented by the
    resampling. A time is covered if a sample of `inp` is within half its
    sampling period (the median step); NaN otherwise (data gaps, before the
    start and after the end of `inp`). A covered time that is not between two
    samples of `inp` at most 1.5 periods apart (a gap edge, an end) takes the
    nearest sample, not a value interpolated across the gap."""
    time, ref = _ns(inp.time.data), _ns(inp_r.time.data)
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


def feeps_avg_4sc(
    b_list: Sequence[DataArray], flag: str = "omni", combined_energies: dict = None
) -> DataArray:
    r"""Average data from 4 spacecrafts while ignoring NaN values.
    Computes the input quantity at the center of mass of the MMS
    tetrahedron. When averaging, NaN values are ignored by counting the number of
    non-NaN values at each time step.

    Parameters
    ----------
    b_list : Sequence of DataArray
        List of the time series of the quantity for each spacecraft.

    flag : str, optional
        Type of data to be averaged (omnidirectional or
        pitch-angle distribution). The default is "omni".

    combined_energies : dict, optional
        Used only if flag is "omni". Energy channels of each spacecraft to combine
        into each output channel: ``{i: [ch_1, ch_2, ch_3, ch_4]}``, where ``i`` is
        the output channel (0-based, one key per channel of the first spacecraft)
        and ``ch_k`` the channel of MMS k (1-based, as in pyspedas), or None to
        leave MMS k out of that channel. The spacecraft are identified by their
        ``mmsId`` attribute. The energy of each output channel is the average of
        the combined channels. The default is None, which averages the channels
        with the same index on all spacecraft, even if their energies differ.

    Returns
    -------
    b_avg : DataArray
        Time series of the input quantity at the center of mass of the
        MMS tetrahedron.

    Raises
    ------
    TypeError
        If b_list is not a list (or tuple) of DataArray, or combined_energies
        is not a dict.
    ValueError
        If flag is not "omni" or "pad", or if the spacecraft have different
        ``UNITS``.

    Notes
    -----
    The spacecraft are resampled to the times of the first one. A spacecraft
    is left out of the average where it has no data: in its data gaps (two
    samples more than 1.5 sampling periods apart), and before the start and
    after the end of its data (beyond half a sampling period). The resampling
    filled the gaps by interpolation and extrapolated past the ends, which
    were averaged as data. Times where the quantity is NaN on all the
    spacecraft give NaN.

    """
    # Check input type
    if not isinstance(b_list, (list, tuple)):
        raise TypeError("b_list must be a list or a tuple")

    if flag not in ["omni", "pad"]:
        raise ValueError("flag must be 'omni' or 'pad'")

    b_list_r = []

    for b in b_list:
        if isinstance(b, xr.DataArray):
            # On the times of the first spacecraft, NaN where a spacecraft has
            # no data (the resampling filled its gaps and extrapolated past its
            # ends, averaged as data)
            b_r = resample(b, b_list[0], f_s=calc_fs(b_list[0]))
            b_list_r.append(_without_gaps(b, b_r))
        else:
            raise TypeError("elements of b_list must be DataArray")

    # The units of the first spacecraft were taken
    _check_same_units(b_list, "spacecraft")

    b_list_count_nans = deepcopy(b_list_r)
    b_list_r = [xr.where(np.isnan(b), 0, b) for b in b_list_r]
    b_avg_data = np.zeros(b_list_r[0].shape)
    b_nan_denom = np.zeros(b_list_r[0].shape)

    if flag == "omni" and combined_energies is not None:
        if not isinstance(combined_energies, dict):
            raise TypeError("combined_energies must be a dict")

        energy = np.zeros(len(b_list_r[0].energy.data))

        for ien_bin in range(len(b_list_r[0].energy.data)):
            non_none_sc_ch = 0
            for i, b, b_count_nan in zip(
                range(len(b_list_r)), b_list_r, b_list_count_nans
            ):
                mms_id = b_list[i].attrs["mmsId"] - 1
                if combined_energies[ien_bin][mms_id] is None:
                    continue
                non_none_sc_ch += 1
                b_avg_data[:, ien_bin] += b.data[
                    :, combined_energies[ien_bin][mms_id] - 1
                ]
                b_nan_denom[:, ien_bin] += _nan_count(
                    b_count_nan.data[:, combined_energies[ien_bin][mms_id] - 1]
                ).data
                energy[ien_bin] += b.energy.data[combined_energies[ien_bin][mms_id] - 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                energy[ien_bin] /= non_none_sc_ch
    else:

        for b, b_count_nan in zip(b_list_r, b_list_count_nans):

            b_avg_data += b.data
            b_nan_denom += _nan_count(b_count_nan).data

    coords = {}
    if flag == "omni":
        if combined_energies is None:
            energy = b_list[0].energy.data
        coords = {"time": b_list_r[0].time, "energy": energy}
    elif flag == "pad":
        coords = {"time": b_list_r[0].time, "theta": b_list_r[0].theta}

    # NaN (without warning) where no spacecraft has data
    with np.errstate(divide="ignore", invalid="ignore"):
        b_avg_data = b_avg_data / b_nan_denom

    b_avg = xr.DataArray(
        b_avg_data,
        coords=coords,
        dims=b_list_r[0].dims,
        attrs=b_list[0].attrs,
    )

    if "probe" in b_avg.attrs.keys():
        b_avg.attrs["probe"] = "4sc_avg"
    if "mms" in b_avg.attrs.keys():
        b_avg.attrs["mms"] = "4sc_avg"
    if "MMS" in b_avg.attrs.keys():
        b_avg.attrs["MMS"] = "4sc_avg"
    if "mmsId" in b_avg.attrs.keys():
        b_avg.attrs["mmsId"] = "4sc_avg"

    return b_avg
