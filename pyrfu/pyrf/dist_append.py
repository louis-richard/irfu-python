#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def dist_append(inp0, inp1):
    r"""Concatenate two distribution skymaps along the time axis.

    Parameters
    ----------
    inp0 : xarray.Dataset
        3D skymap distribution at early times.
    inp1 : xarray.Dataset
        3D skymap distribution at late times.

    Returns
    -------
    out : xarray.Dataset
        3D skymap of the concatenated 3D skymaps.

    Notes
    -----
    The time series have to be in the correct time order.

    """

    if inp0 is None:
        return inp1

    # time
    time = np.hstack([inp0.time.data, inp1.time.data])

    # attributes
    attrs = inp0.attrs

    # Azimuthal angle
    if inp0.phi.ndim == 2:
        phi = np.vstack([inp0.phi.data, inp1.phi.data])
    else:
        phi = inp0.phi.data

    # Elevation angle
    theta = inp0.theta.data

    # distribution
    data = np.vstack([inp0.data, inp1.data])

    if "delta_energy_plus" in attrs:
        delta_energy_plus = np.vstack([inp0.attrs["delta_energy_plus"].data,
                                       inp1.attrs["delta_energy_plus"].data])
        attrs["delta_energy_plus"] = delta_energy_plus

    if "delta_energy_minus" in inp0.attrs:
        delta_energy_minus = np.vstack([inp0.attrs["delta_energy_minus"].data,
                                        inp1.attrs["delta_energy_minus"].data])
        attrs["delta_energy_minus"] = delta_energy_minus

    # Energy
    if inp0.attrs["tmmode"] == "brst":
        step_table = np.hstack([inp0.attrs["esteptable"],
                                inp1.attrs["esteptable"]])

        out = ts_skymap(time, data, None, phi, theta, energy0=inp0.energy0,
                        energy1=inp0.energy1, esteptable=step_table)

        attrs.pop("esteptable")
    else:
        energy = np.vstack([inp0.energy.data, inp1.energy.data])

        out = ts_skymap(time, data, energy, phi, theta)

    for k in attrs:
        out.attrs[k] = attrs[k]

    return out
