#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def feeps_spin_avg(flux_omni, spin_sectors):
    r"""Spin-average the omni-directional FEEPS energy spectra.

    Parameters
    ----------
    flux_omni : xarray.DataArray
        Omni-direction flux.
    spin_sectors : xarray.DataArray or numpy.ndarray
        Time series of the spin sectors, on the times of `flux_omni`.

    Returns
    -------
    spin_avg_flux : xarray.DataArray
        Spin averaged omni-directional flux.

    Raises
    ------
    ValueError
        If `spin_sectors` and `flux_omni` don't have the same number of times.

    Notes
    -----
    A spin starts at the first sample after the spin sector number wraps
    around. Each spin is averaged over its own samples, from its first sample
    up to (not including) the first sample of the next spin, and is time
    stamped at its first sample. The partial spins at the start and end of
    the interval are kept, and a spin without data is NaN.

    IDL SPEDAS and pyspedas instead average from the second sample of a spin
    to the first sample of the next one, drop the partial spins and set the
    last spin(s) to zero.

    """

    # NumPy array, a DataArray would be aligned on time in the comparison
    spin_sectors = np.asarray(spin_sectors)

    if len(spin_sectors) != len(flux_omni.time):
        raise ValueError("spin_sectors must have one value per time of flux_omni")

    spin_starts = np.where(spin_sectors[:-1] >= spin_sectors[1:])[0] + 1

    # spins [s_k, s_k+1), with the partial spins at both ends
    bounds = np.unique(np.hstack([0, spin_starts, len(spin_sectors)]))

    energies = flux_omni.energy.data
    data = flux_omni.data

    spin_avg = np.full([len(bounds) - 1, len(energies)], np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (i_start, i_stop) in enumerate(zip(bounds[:-1], bounds[1:])):
            spin_avg[i, :] = np.nanmean(data[i_start:i_stop, :], axis=0)

    spin_avg_flux = xr.DataArray(
        spin_avg,
        coords=[flux_omni.time.data[bounds[:-1]], energies],
        dims=["time", "energy"],
        attrs={**flux_omni.attrs},
    )
    return spin_avg_flux
