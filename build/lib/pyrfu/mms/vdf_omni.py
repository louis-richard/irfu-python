#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
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
    out : xarray.DataArray
        Time series of the omnidirectional velocity distribution function.

    """

    time = vdf.time.data

    energy = vdf.energy.data
    thetas = vdf.theta.data
    dangle = np.pi / vdf.theta.shape[0]
    np_phi = vdf.phi.shape[1]

    sine_theta = np.ones((np_phi, 1)) * np.sin(np.deg2rad(thetas))
    solid_angles = dangle * dangle * sine_theta
    all_solid_angles = np.tile(solid_angles,
                               (len(time), energy.shape[1], 1, 1))

    if method.lower() == "mean":
        dist = vdf.data.data * all_solid_angles
        omni = np.squeeze(np.nanmean(np.nanmean(dist, axis=3), axis=2))
        omni /= np.mean(np.mean(solid_angles))
    elif method.lower() == "sum":
        dist = vdf.data.data
        omni = np.squeeze(np.nansum(np.nansum(dist, axis=3), axis=2))
    else:
        raise ValueError("invalid method!!")

    energy = np.mean(energy[:2, :], axis=0)

    out = xr.DataArray(omni, coords=[time, energy], dims=["time", "energy"],
                       attrs=vdf.attrs)

    return out
