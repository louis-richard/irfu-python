#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# #rd party imports
import numpy as np
import xarray as xr

from scipy import interpolate

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def feeps_pad_spinavg(pad, spin_sectors, bin_size: float = 16.3636):
    r"""Spin-average the FEEPS pitch angle distributions.

    Parameters
    ----------
    pad : xarray.DataArray
        Pitch angle distribution.
    spin_sectors : xarray.DataArray
        Time series of the spin sectors.
    bin_size : float, Optional
        Size of the pitch angle bins

    Returns
    -------
    out : xarray.DataArray
        Spin averaged pitch angle distribution.

    """

    n_pabins = int(180. / bin_size)
    new_bins = 180. * np.arange(int(n_pabins + 1)) / n_pabins

    # get the spin sectors
    # v5.5+ = mms1_epd_feeps_srvy_l1b_electron_spinsectnum
    spin_sectors = spin_sectors.data

    spin_starts = np.where(spin_sectors[:-1] >= spin_sectors[1:])[0] + 1

    times = pad.time.data
    data = pad.data
    angles = pad.theta.data

    n_spin = len(spin_starts)
    n_angs = len(angles)

    spin_avg_flux = np.zeros([n_spin, n_angs])
    rebinned_data = np.zeros([n_spin, int(n_pabins + 1)])
    spin_times = np.zeros(n_spin, dtype="<M8[ns]")

    # the following is for rebinning and interpolating to new_bins
    srx = n_angs / (n_pabins + 1) * (np.arange(int(n_pabins + 1)) + 0.5) - 0.5

    c_start = 0
    for i, spin in enumerate(spin_starts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spin_avg_flux[i, :] = np.nanmean(data[c_start:spin + 1, :], axis=0)
            spin_times[i] = times[c_start]

            # rebin and interpolate to new_bins
            spin_avg_interp = interpolate.interp1d(np.arange(n_angs),
                                                   spin_avg_flux[i, :],
                                                   fill_value='extrapolate')
            rebinned_data[i, :] = spin_avg_interp(srx)

            # we want to take the end values instead of extrapolating
            rebinned_data[i, 0] = spin_avg_flux[i, 0]
            rebinned_data[i, -1] = spin_avg_flux[i, -1]

        c_start = spin + 1

    out = xr.DataArray(rebinned_data,
                       coords=[spin_times, new_bins],
                       dims=["time", "theta"])

    return out
