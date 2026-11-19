#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Literal

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu._core.time import _ns
from pyrfu.mms._internal.epd import _spin_reduce
from pyrfu.mms.feeps_omni import _DEFAULT_METHODS


def _spin_bounds(spin_sectors, time):
    r"""Start indices of the spins [s_k, s_k+1), from the decreases of the spin
    sector and the data gaps (two samples more than 1.5 median sampling
    periods apart), with the partial spins at both ends."""

    spin_starts = np.where(spin_sectors[:-1] >= spin_sectors[1:])[0] + 1

    # Data gaps start spins too: a gap of a whole number of spins can leave
    # the sectors increasing
    if len(time) > 2:
        d_t = np.diff(_ns(time))
        gaps = np.where(d_t > 1.5 * np.median(d_t))[0] + 1
        spin_starts = np.hstack([spin_starts, gaps])

    return np.unique(np.hstack([0, spin_starts, len(spin_sectors)]))


def _feeps_complete_spins(spin_sectors, time):
    r"""Start and stop indices of the complete spins: the spins of
    `_spin_bounds` with all the sectors, i.e., as many samples as there are
    different sector numbers."""

    bounds = _spin_bounds(spin_sectors, time)
    lengths = np.diff(bounds)

    # The partial spins at the ends and those cut by data gaps are left out
    complete = lengths == len(np.unique(spin_sectors))

    if not np.any(complete):
        raise ValueError("no complete spin (with all the spin sectors) in the interval")

    return bounds[:-1][complete], bounds[1:][complete]


def feeps_spin_avg(
    flux_omni,
    spin_sectors,
    *,
    method: Literal["mean", "sum", "quadrature"] | None = None,
):
    r"""Spin-average the omni-directional FEEPS energy spectra.

    Parameters
    ----------
    flux_omni : xarray.DataArray
        Omni-direction flux.
    spin_sectors : xarray.DataArray or numpy.ndarray
        Time series of the spin sectors, on the times of `flux_omni`.
    method : {"mean", "sum", "quadrature"}, Optional
        Combination of the samples of a spin: the mean (intensities, count
        rates), the sum (counts) or the square root of the sum of the squares
        divided by the number of samples (errors of the mean). Default is from
        the units of `flux_omni` (``units_name`` of `pyrfu.mms.feeps_omni`):
        "sum" for "counts", "quadrature" for "fluxerr", "mean" otherwise.

    Returns
    -------
    spin_avg_flux : xarray.DataArray
        Spin averaged omni-directional flux, one per complete spin at the time
        of its first sample.

    Raises
    ------
    ValueError
        If `spin_sectors` and `flux_omni` don't have the same number of times,
        if the interval has no complete spin, or if `method` is not one of the
        above.

    Notes
    -----
    A spin starts at the first sample after the spin sector number wraps
    around. Each spin is averaged over its own samples, from its first sample
    up to (not including) the first sample of the next spin, and is time
    stamped at its first sample, as the EIS spin averages and
    :func:`pyrfu.mms.hpca_spin_sum`. Only the complete spins are averaged: the
    partial spins at the start and end of the interval and the spins cut by
    data gaps (two samples more than 1.5 median sampling periods apart) are
    left out. A spin without data is NaN.

    IDL SPEDAS and pyspedas instead average from the second sample of a spin
    to the first sample of the next one, and set the last spin(s) to zero.

    The spins come from the spin sector numbers, as in IDL SPEDAS and
    pyspedas, not from the spin numbers of the files (``spin``): in burst
    mode the spin number changes one sample after the sector number wraps
    around (at sector 1), and in survey mode it gives the last sectors of
    the spin before the start of a file to the first spin of the file (MMS1
    and MMS3, 2017-07-11 and 2019-09-14).

    """

    combine = method or _DEFAULT_METHODS.get(flux_omni.attrs.get("units_name"), "mean")

    if combine not in ["mean", "sum", "quadrature"]:
        raise ValueError(f"method must be mean, sum or quadrature, not {combine!r}")

    # NumPy array, a DataArray would be aligned on time in the comparison
    spin_sectors = np.asarray(spin_sectors)

    if len(spin_sectors) != len(flux_omni.time):
        raise ValueError("spin_sectors must have one value per time of flux_omni")

    spin_starts, spin_stops = _feeps_complete_spins(spin_sectors, flux_omni.time.data)

    energies = flux_omni.energy.data
    data = flux_omni.data

    spin_avg = _spin_reduce(data, spin_starts, spin_stops, combine).astype(np.float64)

    spin_avg_flux = xr.DataArray(
        spin_avg,
        coords=[flux_omni.time.data[spin_starts], energies],
        dims=["time", "energy"],
        attrs={**flux_omni.attrs},
    )
    return spin_avg_flux
