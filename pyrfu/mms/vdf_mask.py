#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.resample import resample

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def _limit(vdf, limit):
    # Energy limit (eV) at the times of the distribution, (time, 1)
    if isinstance(limit, xr.DataArray):
        if len(limit.time) == len(vdf.time) and np.all(
            limit.time.data == vdf.time.data
        ):
            limit = limit.data
        else:
            limit = resample(limit, vdf.time).data

    limit = np.asarray(limit, dtype=np.float64)

    if limit.ndim == 0:
        limit = np.full(len(vdf.time), float(limit))

    if limit.shape not in [(len(vdf.time),), (len(vdf.time), 1)]:
        raise ValueError("The energy limits must be a scalar or (time,) values")

    return limit.reshape(-1, 1)


def vdf_mask(vdf, below=None, above=None, mask=None):
    r"""Replaces energy channels of a distribution by NaN, as irfu-matlab
    PDist.mask.

    Parameters
    ----------
    vdf : xarray.Dataset
        Distribution with (time, energy, ...) data and (time, energy) or
        (energy,) energies in eV: skymap, pitch-angle or omni-directional
        distribution.
    below : float or array_like or xarray.DataArray, Optional
        The channels whose energy is below this energy (eV) are masked
        (irfu-matlab ``mask("energy", "max", ...)``): a scalar, (time,) values
        or a time series, e.g. from `vdf_find_low_counts`.
    above : float or array_like or xarray.DataArray, Optional
        The channels whose energy is above this energy (eV) are masked
        (irfu-matlab ``mask("energy", "min", ...)``).
    mask : array_like or xarray.DataArray, Optional
        (time, energy) array of 0 and 1: the channels with 1 are masked
        (irfu-matlab ``mask("energy", "mat", ...)``), e.g. from
        `vdf_find_low_counts`.

    Returns
    -------
    out : xarray.Dataset
        Copy of `vdf` with the masked channels set to NaN.

    Raises
    ------
    ValueError
        If the limits or the mask have the wrong shape.

    Notes
    -----
    The energies of each time are used, so that the alternating energy tables
    of FPI are masked correctly (irfu-matlab uses the energies of the first
    time for all times).

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> vdf_i = mms.vdf_mask(vdf_i, below=100.0)

    """
    energy = np.broadcast_to(
        np.asarray(vdf.energy.data, dtype=np.float64), vdf.data.shape[:2]
    )
    masked = np.zeros(energy.shape, dtype=bool)

    if below is not None:
        masked |= energy < _limit(vdf, below)

    if above is not None:
        masked |= energy > _limit(vdf, above)

    if mask is not None:
        mask = np.asarray(mask)

        if mask.shape != energy.shape:
            raise ValueError("mask must be (time, energy)")

        masked |= mask.astype(bool)

    data = np.where(
        masked.reshape(masked.shape + (1,) * (vdf.data.ndim - 2)),
        np.nan,
        vdf.data.data,
    )

    out = vdf.copy()
    out["data"] = vdf.data.copy(data=data)
    return out
