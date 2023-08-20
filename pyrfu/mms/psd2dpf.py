#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _mass_ratio(inp):
    if inp.attrs["species"] in ["ions", "ion", "protons", "proton"]:
        mass_ratio = 1
    elif inp.attrs["species"] in ["alphas", "alpha", "helium"]:
        mass_ratio = 4
    elif inp.attrs["species"] in ["electrons", "e"]:
        mass_ratio = constants.electron_mass / constants.proton_mass
    else:
        raise ValueError("Invalid specie")

    return mass_ratio


def _convert(inp, units, mass_ratio):
    fact = 1 / (1e6 * 0.53707 * mass_ratio**2)

    if units.lower() == "s^3/cm^6":
        out = inp * 1e30 * fact
    elif units.lower() == "s^3/m^6":
        out = inp * 1e18 * fact
    elif units.lower() == "s^3/km^6":
        out = inp * fact
    else:
        raise ValueError("Invalid unit")

    return out


def psd2dpf(inp):
    r"""Compute differential particle flux from phase density.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the velocity distribution function with :
            * time : Time samples.
            * data : 3D velocity distribution.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    Returns
    -------
    dpf : xarray.Dataset
        Time series of the 3D differential particle flux in 1/(cm^2 s sr keV) with :
            * time : Time samples.
            * data : 3D density particle flux.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    """

    assert isinstance(inp, (xr.DataArray, xr.Dataset)), "inp must be a xarray"

    if isinstance(inp, xr.Dataset):
        tmp_data = _convert(inp.data.data, inp.data.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        energy_mat = np.tile(energy[:, :, None, None], (1, 1, *tmp_data.shape[2:]))
        tmp_data *= energy_mat
        out = inp.copy()
        out.data.data = np.squeeze(tmp_data) * 1e3
        out.data.attrs["UNITS"] = "1/(cm^2 s sr keV)"
    else:
        tmp_data = _convert(inp.data, inp.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        energy_mat = np.tile(energy, (tmp_data.shape[0], 1))
        tmp_data *= energy_mat
        out = inp.copy()
        out.data = np.squeeze(tmp_data) * 1e3
        out.attrs["UNITS"] = "1/(cm^2 s sr keV)"

    return out
