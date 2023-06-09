#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
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


def _convert(psd, mass_ratio):
    if psd.data.attrs["UNITS"].lower() == "s^3/cm^6":
        tmp_data = psd.data.data * 1e30 / (1e6 * 0.53707 * mass_ratio**2)
    elif psd.data.attrs["UNITS"].lower() == "s^3/m^6":
        tmp_data = psd.data.data * 1e18 / (1e6 * 0.53707 * mass_ratio**2)
    elif psd.data.attrs["UNITS"].lower() == "s^3/km^6":
        tmp_data = psd.data.data / (1e6 * 0.53707 * mass_ratio**2)
    else:
        raise ValueError("Invalid unit")

    return tmp_data


def psd2dpf(vdf):
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
        Time series of the 3D differential particle flux with :
            * time : Time samples.
            * data : 3D density particle flux.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    """

    tmp_data = _convert(vdf, _mass_ratio(vdf))

    energy = vdf.energy.data

    if tmp_data.ndim == 2:
        tmp_data = tmp_data[:, :, None, None]

    data_r = np.reshape(
        tmp_data,
        (tmp_data.shape[0], tmp_data.shape[1], np.prod(tmp_data.shape[2:])),
    )

    if energy.ndim == 1:
        energy_mat = np.tile(
            energy,
            (len(vdf.time), np.prod(tmp_data.shape[2:]), 1),
        )
        energy_mat = np.transpose(energy_mat, [0, 2, 1])
    elif energy.ndim == 2:
        energy_mat = np.tile(energy, (np.prod(tmp_data.shape[2:]), 1, 1))
        energy_mat = np.transpose(energy_mat, [1, 2, 0])
    else:
        raise ValueError("Invalid energy shape")

    data_r *= energy_mat
    tmp_data = np.reshape(data_r, tmp_data.shape)

    dpf = vdf.copy()
    dpf.data.data = np.squeeze(tmp_data) * 1e3
    dpf.data.attrs["UNITS"] = "1/(cm^2 s sr keV)"

    return dpf
