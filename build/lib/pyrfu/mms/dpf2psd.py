#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import constants

# Local imports
from .spectr_to_dataset import spectr_to_dataset

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _mass_ratio(inp):
    if inp.attrs["species"] in ["ions", "i"]:
        mass_ratio = 1
    elif inp.attrs["species"] in ["electrons", "e"]:
        mass_ratio = constants.electron_mass / constants.proton_mass
    else:
        raise ValueError("Invalid specie")

    return mass_ratio


def _convert(inp, mass_ratio):
    if inp.attrs["UNITS"] == "1/(cm^2 s sr keV)":
        tmp_data = inp.data.data * 1e-3 / 1e12 * 0.53707 * mass_ratio ** 2
    else:
        raise ValueError("Invalid unit")

    return tmp_data


def dpf2psd(inp):
    r"""Computes phase space density from differential particle flux.

    Parameters
    ----------
    inp : xarray.Dataset or xarray.DataArray
        Time series of the differential particle flux in
        [(cm^{2} s sr keV)^{-1}].

    Returns
    -------
    psd : xarray.Dataset
        Time series of the phase space density in [s^{3} m^{-6}].

    """

    if isinstance(inp, xr.DataArray):
        inp = spectr_to_dataset(inp)

    tmp_data = _convert(inp, _mass_ratio(inp))

    energy = inp.energy.data

    if tmp_data.ndim == 2:
        tmp_data = tmp_data[:, :, None, None]

    data_r = np.reshape(tmp_data,
                        (tmp_data.shape[0], tmp_data.shape[1],
                         np.prod(tmp_data.shape[2:])))

    if energy.ndim == 1:
        energy_mat = np.tile(energy,
                             (len(inp.time),
                              np.prod(tmp_data.shape[2:]), 1))
        energy_mat = np.transpose(energy_mat, [0, 2, 1])
    elif energy.ndim == 2:
        energy_mat = np.tile(energy, (np.prod(tmp_data.shape[2:]), 1, 1))
        energy_mat = np.transpose(energy_mat, [1, 2, 0])
    else:
        raise ValueError("Invalid energy shape")

    data_r /= energy_mat
    tmp_data = np.reshape(data_r, tmp_data.shape)

    psd = inp.copy()
    psd.data.data = np.squeeze(tmp_data)
    psd.attrs["UNITS"] = "s^3/m^6"

    return psd
