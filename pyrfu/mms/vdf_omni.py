#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.pyrf.ts_spectr import ts_spectr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def vdf_omni(vdf, method: str = "mean"):
    r"""Computes omni-directional distribution, without changing the units.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the 3D velocity distribution with :
            * time : Time samples.
            * data : 3D velocity distribution.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    method : {"mean", "sum"}, Optional
        Method of computation. Use "sum" for counts and "mean" for
        everything else. Default is "mean".

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the omnidirectional velocity distribution function: a
        DataArray (time, energy) with one energy table, or a Dataset with
        (time, energy) energies when the energy tables alternate. Both keep
        the units and the attributes of vdf (UNITS on the data variable of
        the Dataset, as for vdf).

    Raises
    ------
    ValueError
        If method is not "mean" or "sum".

    Notes
    -----
    As irfu-matlab PDist.omni, "mean" averages the distribution weighted by
    the solid angles over theta, then phi, and divides by the mean solid
    angle; without NaNs, this is the solid-angle weighted mean. A channel
    without data at all angles gives NaN ("mean") or 0 ("sum").

    """

    if method.lower() not in ["mean", "sum"]:
        raise ValueError(f"Invalid method {method!r}, use mean or sum")

    time = vdf.time.data
    energy = vdf.energy.data
    data = vdf.data.data

    with warnings.catch_warnings():
        # Channels without data at all angles give NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method.lower() == "mean":
            # Solid angles of the (phi, theta) bins: dphi dtheta sin(theta),
            # the constant dphi dtheta cancels with the mean solid angle
            solid_angles = np.sin(np.deg2rad(vdf.theta.data))
            omni = np.nanmean(np.nanmean(data * solid_angles, axis=3), axis=2)
            omni /= np.mean(solid_angles)
        else:
            omni = np.nansum(np.nansum(data, axis=3), axis=2)

    # Use global and zVariable attributes
    attrs = {**vdf.data.attrs, **vdf.attrs}
    attrs = {k: attrs[k] for k in sorted(attrs)}

    out = ts_spectr(time, energy, omni, attrs=attrs)

    # Time varying energies (alternating tables): units on the data variable
    # and the other attributes global, as for vdf
    if isinstance(out, xr.Dataset):
        out.data.attrs = dict(vdf.data.attrs)
        out.attrs = dict(vdf.attrs)

    return out
