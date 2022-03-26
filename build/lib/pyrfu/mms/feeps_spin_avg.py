#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def feeps_spin_avg(flux_omni, spin_sectors):
    r"""spin-average the omni-directional FEEPS energy spectra

    Parameters
    ----------
    flux_omni : xarray.DataArray
        Omni-direction flux.
    spin_sectors : xarray.DataArray
        Time series of the spin sectors.

    Returns
    -------
    spin_avg_flux : xarray.DataArray
        Spin averaged omni-directional flux.

    """

    spin_starts = np.where(spin_sectors[:-1] >= spin_sectors[1:])[0] + 1

    energies = flux_omni.energy.data
    data = flux_omni.data

    spin_avg = np.zeros([len(spin_starts), len(energies)])

    c_start = spin_starts[0]
    for i, spin_start in enumerate(spin_starts[1:-1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spin_avg[i, :] = np.nanmean(data[c_start:spin_start + 1, :],
                                        axis=0)
        c_start = spin_start + 1

    spin_avg_flux = xr.DataArray(spin_avg,
                                 coords=[flux_omni.time.data[spin_starts],
                                         energies],
                                 dims=["time", "energy"])
    return spin_avg_flux
