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


def feeps_pad_spinavg(pad, spin_sectors, bin_size: float = None):
    r"""Spin-average the FEEPS pitch angle distributions.

    Parameters
    ----------
    pad : xarray.DataArray
        Pitch angle distribution.
    spin_sectors : xarray.DataArray or numpy.ndarray
        Time series of the spin sectors, on the times of `pad`.
    bin_size : float, Optional
        Ignored, the spin average is on the pitch angle bins of `pad`.

        .. deprecated:: 2.6.0
            `bin_size` will be removed in a future version.

    Returns
    -------
    out : xarray.DataArray
        Spin averaged pitch angle distribution.

    Raises
    ------
    ValueError
        If `spin_sectors` and `pad` don't have the same number of times.

    Notes
    -----
    The spins are the same as in :func:`feeps_spin_avg`: each spin is averaged
    from its first sample up to (not including) the first sample of the next
    spin and is time stamped at its first sample, the partial spins at the
    start and end of the interval are kept, and a spin without data is NaN.

    IDL SPEDAS and pyspedas instead average from the second sample of a spin
    to the first sample of the next one, and interpolate the result from the
    bin centres onto the bin edges.

    """

    if bin_size is not None:
        warnings.warn(
            "bin_size is deprecated and ignored, and will be removed in a future "
            "version: the spin average is on the pitch angle bins of pad.",
            FutureWarning,
            stacklevel=2,
        )

    # NumPy array, a DataArray would be aligned on time in the comparison
    spin_sectors = np.asarray(spin_sectors)

    if len(spin_sectors) != len(pad.time):
        raise ValueError("spin_sectors must have one value per time of pad")

    spin_starts = np.where(spin_sectors[:-1] >= spin_sectors[1:])[0] + 1

    # spins [s_k, s_k+1), with the partial spins at both ends
    bounds = np.unique(np.hstack([0, spin_starts, len(spin_sectors)]))

    data = pad.data
    spin_avg_flux = np.full([len(bounds) - 1, len(pad.theta)], np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (i_start, i_stop) in enumerate(zip(bounds[:-1], bounds[1:])):
            spin_avg_flux[i, :] = np.nanmean(data[i_start:i_stop, :], axis=0)

    out = xr.DataArray(
        spin_avg_flux,
        coords=[pad.time.data[bounds[:-1]], pad.theta.data],
        dims=["time", "theta"],
        attrs={**pad.attrs},
    )

    return out
