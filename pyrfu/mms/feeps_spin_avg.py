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


def _spin_bounds(spin_sectors):
    r"""Start indices of the spins [s_k, s_k+1), from the decreases of the spin
    sector, with the partial spins at both ends."""
    spin_starts = np.where(spin_sectors[:-1] >= spin_sectors[1:])[0] + 1
    return np.unique(np.hstack([0, spin_starts, len(spin_sectors)]))


def _complete_spins(spin_sectors):
    r"""Start and stop indices of the complete spins: the spins of
    `_spin_bounds` with all the sectors, i.e., as many samples as there are
    different sector numbers."""
    bounds = _spin_bounds(spin_sectors)
    lengths = np.diff(bounds)

    # The partial first and last spins and the spins cut by data gaps are left
    # out (they were kept)
    complete = lengths == len(np.unique(spin_sectors))

    if not np.any(complete):
        raise ValueError("no complete spin (with all the spin sectors) in the interval")

    return bounds[:-1][complete], bounds[1:][complete]


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
        Spin averaged omni-directional flux, one per complete spin at the time
        of its first sample.

    Raises
    ------
    ValueError
        If `spin_sectors` and `flux_omni` don't have the same number of times,
        or if the interval has no complete spin.

    Notes
    -----
    A spin starts at the first sample after the spin sector number wraps
    around. Each spin is averaged over its own samples, from its first sample
    up to (not including) the first sample of the next spin, and is time
    stamped at its first sample, as the EIS spin averages and
    :func:`pyrfu.mms.hpca_spin_sum`. Only the complete spins are averaged: the
    partial spins at the start and end of the interval and the spins cut by
    data gaps are left out (they were kept). A spin without data is NaN.

    IDL SPEDAS and pyspedas instead average from the second sample of a spin
    to the first sample of the next one, and set the last spin(s) to zero.

    """

    # NumPy array, a DataArray would be aligned on time in the comparison
    spin_sectors = np.asarray(spin_sectors)

    if len(spin_sectors) != len(flux_omni.time):
        raise ValueError("spin_sectors must have one value per time of flux_omni")

    spin_starts, spin_stops = _complete_spins(spin_sectors)

    energies = flux_omni.energy.data
    data = flux_omni.data

    spin_avg = np.full([len(spin_starts), len(energies)], np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (i_start, i_stop) in enumerate(zip(spin_starts, spin_stops)):
            spin_avg[i, :] = np.nanmean(data[i_start:i_stop, :], axis=0)

    spin_avg_flux = xr.DataArray(
        spin_avg,
        coords=[flux_omni.time.data[spin_starts], energies],
        dims=["time", "energy"],
        attrs={**flux_omni.attrs},
    )
    return spin_avg_flux
