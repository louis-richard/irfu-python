#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from typing import Literal

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.mms._internal.epd import _spin_reduce

from .feeps_omni import _DEFAULT_METHODS
from .feeps_spin_avg import _feeps_complete_spins


def feeps_pad_spinavg(
    pad,
    spin_sectors,
    bin_size: float | None = None,
    *,
    method: Literal["mean", "sum", "quadrature"] | None = None,
):
    r"""Spin-average the FEEPS pitch angle distributions.

    Parameters
    ----------
    pad : xarray.DataArray
        Pitch angle distribution.
    spin_sectors : xarray.DataArray or numpy.ndarray
        Time series of the spin sectors, on the times of `pad`.
    bin_size : float, Optional
        Ignored, the spin average is on the pitch angle bins of `pad`.

        .. deprecated:: 2.5.1
            `bin_size` will be removed in a future version.
    method : {"mean", "sum", "quadrature"}, Optional
        Combination of the samples of a spin: the mean (intensities, count
        rates), the sum (counts) or the square root of the sum of the squares
        divided by the number of samples (errors of the mean). Default is from
        the units of `pad` (``units_name`` of `pyrfu.mms.feeps_pad`): "sum" for
        "counts", "quadrature" for "fluxerr", "mean" otherwise.

    Returns
    -------
    out : xarray.DataArray
        Spin averaged pitch angle distribution, one per complete spin at the
        time of its first sample.

    Raises
    ------
    ValueError
        If `spin_sectors` and `pad` don't have the same number of times, if
        the interval has no complete spin, or if `method` is not one of the
        above.

    Notes
    -----
    The spins are the same as in :func:`pyrfu.mms.feeps_spin_avg`: each
    complete spin is averaged from its first sample up to (not including) the
    first sample of the next spin and is time stamped at its first sample, the
    partial spins at the start and end of the interval and the spins cut by
    data gaps are left out, and a spin without data is NaN.

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

    combine = method or _DEFAULT_METHODS.get(pad.attrs.get("units_name"), "mean")

    if combine not in ["mean", "sum", "quadrature"]:
        raise ValueError(f"method must be mean, sum or quadrature, not {combine!r}")

    # NumPy array, a DataArray would be aligned on time in the comparison
    spin_sectors = np.asarray(spin_sectors)

    if len(spin_sectors) != len(pad.time):
        raise ValueError("spin_sectors must have one value per time of pad")

    spin_starts, spin_stops = _feeps_complete_spins(spin_sectors, pad.time.data)

    data = pad.data
    spin_avg_flux = _spin_reduce(data, spin_starts, spin_stops, combine).astype(
        np.float64
    )

    out = xr.DataArray(
        spin_avg_flux,
        coords=[pad.time.data[spin_starts], pad.theta.data],
        dims=["time", "theta"],
        attrs={**pad.attrs},
    )

    return out
