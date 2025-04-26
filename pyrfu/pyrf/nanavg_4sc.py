#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence

# 3rd party imports
from copy import deepcopy
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from .calc_fs import calc_fs
from .resample import resample

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
    # inp_to_counts = inp.where(np.isnan(inp) == True, other=1)
    # inp_to_counts = inp_to_counts.where(np.isnan(inp) == False, other=0)

    return inp_to_counts


def nanavg_4sc(b_list: Sequence[DataArray], combined_energies:list = None) -> DataArray:
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
        If b_list is not a list of DataArray or Dataset

    """
    # Check input type
    if not isinstance(b_list, list):
        raise TypeError("b_list must be a list")

    b_list_r = []

    for b in b_list:
        if isinstance(b, (xr.DataArray, xr.Dataset)):
            b_list_r.append(resample(b, b_list[0], f_s=calc_fs(b_list[0])))
        else:
            raise TypeError("elements of b_list must be DataArray or Dataset")

    # b_list_r = [b.where(np.isnan(b) == False, other=0) for b in b_list_r]
    b_list_count_nans = deepcopy(b_list_r)
    b_list_r = [xr.where(np.isnan(b), 0, b) for b in b_list_r]
    b_avg_data = np.zeros(b_list_r[0].shape)
    b_nan_denom = np.zeros(b_list_r[0].shape)

    if combined_energies is not None:
        if not isinstance(combined_energies, list):
            raise TypeError("combined_energies must be a list")
        else:

            energy = np.zeros(len(b_list_r[0].energy.data))

            for ien_bin in range(len(b_list_r[0].energy.data)):
                non_none_sc_ch = 0
                for i, b, b_count_nan in zip(range(len(b_list_r)), b_list_r, b_list_count_nans):
                    if combined_energies[ien_bin][i] == None:
                        continue
                    non_none_sc_ch += 1
                    b_avg_data[:, ien_bin ] += b.data[:, combined_energies[ien_bin ][i] - 1]
                    b_nan_denom[:, ien_bin ] += _nan_count(b_count_nan.data[:, combined_energies[ien_bin][i] - 1]).data
                    energy[ien_bin] += b.energy.data[combined_energies[ien_bin][i] - 1]
                energy[ien_bin] /= non_none_sc_ch
    else:

        for b, b_count_nan in zip(b_list_r, b_list_count_nans):

            b_avg_data += b.data
            b_nan_denom += _nan_count(b_count_nan).data
        
    # return b_avg_data, b_nan_denom
    if "probe" in b_list[0].attrs.keys():
        b_list[0].attrs["probe"] = "4sc_avg"
    if "mms" in b_list[0].attrs.keys():
        b_list[0].attrs["mms"] = "4sc_avg"
    if "MMS" in b_list[0].attrs.keys():
        b_list[0].attrs["MMS"] = "4sc_avg"
    if "mmsId" in b_list[0].attrs.keys():
        b_list[0].attrs["mmsId"] = "4sc_avg"

    if combined_energies == None:
        energy = b_list[0].energy.data

    b_avg = xr.DataArray(
        b_avg_data / b_nan_denom,
        coords={"time": b_list_r[0].time, "energy": energy},
        dims=b_list_r[0].dims,
        attrs=b_list[0].attrs,
    )

    return b_avg
