#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def eis_pad_spinavg(inp, spin_nums):
    r"""Calculates spin-averaged Pitch-Angle Distribution (PAD) for the EIS
    instrument.

    Parameters
    ----------
    inp : xarray.DataArray
        Pitch-Angle Distribution.
    spin_nums : xarray.DataArray
        Spin #s associated with each measurement.

    Returns
    -------
    out : xarray.DataArray
        Spin-averaged PAD.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad

    """
    _, spin_starts = np.unique(spin_nums.data, return_index=True)

    spin_times = np.zeros(len(spin_starts), dtype="<M8[ns]")
    spin_sum_flux = np.zeros((len(spin_starts), len(inp.theta.data),
                              len(inp.energy.data)))

    current_start = 0
    # loop through the spins
    for i, spin_start in enumerate(spin_starts):
        idx_ = np.where(spin_nums.data == spin_nums.data[spin_start])[0]
        spin_sum_flux[i, :, :] = np.nanmean(inp.data[idx_, :, :], axis=0)
        spin_times[i] = inp.time.data[current_start]
        current_start = spin_start + 1

    out = xr.DataArray(spin_sum_flux,
                       coords=[spin_times, inp.theta.data, inp.energy.data],
                       dims=["time", "theta", "energy"])

    return out
