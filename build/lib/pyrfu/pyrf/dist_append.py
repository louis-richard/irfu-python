#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dist_append.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr
from .ts_skymap import ts_skymap


def dist_append(inp0=None, inp1=None):
    """Concatenate two distribution skymaps along the time axis.

    Parameters
    ----------
    inp0 : xarray.DataArray
        3D skymap distribution at early times.

    inp1 : xarray.DataArray
        3D skymap distribution at late times.

    Returns
    -------
    out : xarray.DataArray
        3D skymap of the concatenated 3D skymaps.

    Notes
    -----
    The time series have to be in the correct time order.

    """

    assert inp0 is not None and isinstance(inp0, xr.Dataset)
    assert inp1 is not None and isinstance(inp1, xr.Dataset)

    # time
    time = np.hstack([inp0.time.data, inp1.time.data])

    # Azimuthal angle
    if inp0.phi.ndim == 2:
        phi = np.vstack([inp0.phi.data, inp1.phi.data])
    else:
        phi = inp0.phi.data

    # Elevation angle
    theta = inp0.theta.data

    # distribution
    data = np.vstack([inp0.data, inp1.data])

    if "delta_energy_plus" in inp0.attrs:
        delta_energy_plus = np.vstack(
            [inp0.attrs["delta_energy_plus"].data, inp1.attrs["delta_energy_plus"].data])
    else:
        delta_energy_plus = None

    if "delta_energy_minus" in inp0.attrs:
        delta_energy_minus = np.vstack(
            [inp0.attrs["delta_energy_minus"].data, inp1.attrs["delta_energy_minus"].data])
    else:
        delta_energy_minus = None

    # Energy
    if inp0.attrs["tmmode"] == "brst":
        step_table = np.hstack([inp0.attrs["esteptable"], inp1.attrs["esteptable"]])

        out = ts_skymap(time, data, None, phi, theta, energy0=inp0.energy0, energy1=inp0.energy1,
                        esteptable=step_table)
    else:
        energy = np.vstack([inp0.energy.data, inp1.energy.data])

        out = ts_skymap(time, data, energy, phi, theta)

    # attributes
    attrs = inp0.attrs

    if "esteptable" in attrs.keys():
        attrs.pop("esteptable")

    if "delta_energy_minus" in attrs.keys():
        attrs["delta_energy_minus"] = delta_energy_minus

    if "delta_energy_plus" in attrs.keys():
        attrs["delta_energy_plus"] = delta_energy_plus

    for k in attrs:
        out.attrs[k] = attrs[k]

    return out
