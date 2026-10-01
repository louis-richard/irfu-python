#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .calc_fs import calc_fs
from .cross import cross
from .dot import dot
from .normalize import normalize
from .resample import resample
from .time_clip import time_clip

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def poynting_flux(e_xyz, b_xyz, b_hat=None):
    r"""Estimates Poynting flux at electric field sampling as

    .. math::

        S = \frac{E \times B}{\mu_0}

    if `b_hat` is given project the Poynting flux along `b_hat`

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field in mV/m.
    b_xyz : xarray.DataArray
        Time series of the magnetic field in nT.
    b_hat : xarray.DataArray, Optional
        Time series of the direction to project the Poynting flux. Default is
        None (no projection).

    Returns
    -------
    s : xarray.DataArray
        Time series of the Poynting flux in mW/m^2.
    s_z : xarray.DataArray
        Time series of the projection of the Poynting flux along `b_hat` in
        mW/m^2 (only if `b_hat` is given).
    int_s : xarray.DataArray
        Time series of the time integral of the Poynting flux, in mW s/m^2: of
        each component, or of the projection along `b_hat` if it is given. NaNs
        count as zero in the integral.

    Notes
    -----
    E and B are clipped to the interval where both exist, and the one with fewer
    samples is resampled to the times of the other one (B to E if they have the
    same number of samples but different times).

    """

    # interval where both E & B exist
    tint = [
        np.max([np.min(e_xyz.time.data), np.min(b_xyz.time.data)]),
        np.min([np.max(e_xyz.time.data), np.max(b_xyz.time.data)]),
    ]
    tint = [np.datetime_as_string(time, "ns") for time in tint]

    e_xyz, b_xyz = [time_clip(e_xyz, tint), time_clip(b_xyz, tint)]

    if len(e_xyz) < len(b_xyz):
        e_xyz = resample(e_xyz, b_xyz)
        f_spl = calc_fs(b_xyz)
    elif len(e_xyz) > len(b_xyz) or not np.array_equal(
        e_xyz.time.data, b_xyz.time.data
    ):
        b_xyz = resample(b_xyz, e_xyz)
        f_spl = calc_fs(e_xyz)
    else:
        f_spl = calc_fs(b_xyz)

    # Calculate Poynting flux
    s_xyz = cross(e_xyz, b_xyz) / (4 * np.pi / 1e7) * 1e-9
    s_xyz.attrs["UNITS"] = "mW/m^2"

    # Time integral of the Poynting flux (of each component, or along b_hat),
    # with NaNs counted as zero (in a copy, the returned fluxes keep them)
    if b_hat is not None:
        b_m = resample(b_hat, e_xyz)
        s_z = dot(normalize(b_m), s_xyz)
        s_z.attrs["UNITS"] = "mW/m^2"

        int_s_z = s_z.fillna(0.0).cumsum(dim="time") / f_spl
        int_s_z.attrs["UNITS"] = "mW s/m^2"

        return s_xyz, s_z, int_s_z

    int_s = s_xyz.fillna(0.0).cumsum(dim="time") / f_spl
    int_s.attrs["UNITS"] = "mW s/m^2"

    return s_xyz, int_s
