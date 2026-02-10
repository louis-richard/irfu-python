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

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolos.kolokotronis@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.2"
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


def feeps_avg_4sc(
    b_list: Sequence[DataArray], flag: str = "omni", combined_energies: list = None
) -> DataArray:
    r"""Average data from 4 spacecrafts while ignoring NaN values.
    Computes the input quantity at the center of mass of the MMS
    tetrahedron. When averaging, NaN values are ignored by counting the number of
    non-NaN values at each time step.

    Parameters
    ----------
    b_list : Sequence of DataArray or Dataset
        List of the time series of the quantity for each spacecraft.

    flag : str, optional
        type of data to be averaged (omnidirectional or
        pitch-angle distribution). The default is "omni".

    combined_energies : list, optional
        Used only if flag is "omni".
        List of energy channel indices from each spacecraft to be combined into a common
        energy channel. The default is None, which means that the initial channels from
        the omnidirectional data are used. This will likely result to averaging fluxes
        for much different energies.

    Returns
    -------
    b_avg : DataArray or Dataset
        Time series of the input quantity at the center of mass of the
        MMS tetrahedron.

    Raises
    ------
    TypeError
        If b_list is not a list of DataArray or Dataset

    """
    # Check input type
    if not isinstance(b_list, list):
        raise TypeError("b_list must be a list")

    assert flag in ["omni", "pad"]

    b_list_r = []

    for b in b_list:
        if isinstance(b, (xr.DataArray, xr.Dataset)):
            b_list_r.append(resample(b, b_list[0], f_s=calc_fs(b_list[0])))
        else:
            raise TypeError("elements of b_list must be DataArray or Dataset")

    b_list_count_nans = deepcopy(b_list_r)
    b_list_r = [xr.where(np.isnan(b), 0, b) for b in b_list_r]
    b_avg_data = np.zeros(b_list_r[0].shape)
    b_nan_denom = np.zeros(b_list_r[0].shape)

    if flag == "omni" and combined_energies is not None:
        if not isinstance(combined_energies, dict):
            raise TypeError("combined_energies must be a dict")
        else:

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
                        b_count_nan.data[:, combined_energies[ien_bin][i] - 1]
                    ).data
                    energy[ien_bin] += b.energy.data[
                        combined_energies[ien_bin][mms_id] - 1
                    ]
                energy[ien_bin] /= non_none_sc_ch
    else:

        for b, b_count_nan in zip(b_list_r, b_list_count_nans):

            b_avg_data += b.data
            b_nan_denom += _nan_count(b_count_nan).data

    if flag == "omni":
        if combined_energies is None:
            energy = b_list[0].energy.data
        coords = {"time": b_list_r[0].time, "energy": energy}
    elif flag == "pad":
        coords = {"time": b_list_r[0].time, "theta": b_list_r[0].theta}

    b_avg = xr.DataArray(
        b_avg_data / b_nan_denom,
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
