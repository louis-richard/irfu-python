#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .eis_spin_avg import _spin_segments

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
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
        Spin-averaged PAD, one per complete spin at the time of its first
        sample.

    Raises
    ------
    ValueError
        If `spin_nums` and `inp` don't have the same number of times, or if
        the interval has no complete spin.

    Notes
    -----
    The spins are those of :func:`pyrfu.mms.eis_spin_avg`: complete spins
    only, time stamped at their first sample (they were stamped one spin
    early, and the partial spins were kept).

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad

    """
    if len(spin_nums) != len(inp.time):
        raise ValueError("spin_nums must have one value per time of inp")

    spin_starts, spin_stops = _spin_segments(spin_nums.data)

    spin_sum_flux = np.zeros(
        (len(spin_starts), len(inp.theta.data), len(inp.energy.data)),
    )

    # loop through the spins
    for i, (spin_start, spin_stop) in enumerate(zip(spin_starts, spin_stops)):
        # NaN for the pitch-angle bins empty during the whole spin
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            spin_sum_flux[i, :, :] = np.nanmean(
                inp.data[spin_start:spin_stop, :, :], axis=0
            )

    out = xr.DataArray(
        spin_sum_flux,
        coords=[inp.time.data[spin_starts], inp.theta.data, inp.energy.data],
        dims=["time", "theta", "energy"],
    )

    return out
