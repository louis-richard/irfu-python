#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from scipy import constants
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["dpf2psd"]


def _mass_ratio(inp: Union[Dataset, DataArray]) -> float:
    r"""Compute mass ratio of the input species.

    Parameters
    ----------
    inp : Dataset or DataArray
        Input distribution function.

    Returns
    -------
    float
        Mass ratio of the species.

    Raises
    ------
    ValueError
        If the species is not supported.

    """
    if inp.attrs["species"].lower() in ["ions", "ion", "protons", "proton"]:
        mass_ratio = 1
    elif inp.attrs["species"].lower() in ["alphas", "alpha", "helium"]:
        mass_ratio = 4
    elif inp.attrs["species"].lower() in ["electrons", "e"]:
        mass_ratio = constants.electron_mass / constants.proton_mass
    else:
        raise ValueError("Invalid specie")

    return mass_ratio


def _convert(inp: np.ndarray, units: str, mass_ratio: float) -> np.ndarray:
    r"""Convert differential particle flux to phase space density.

    Parameters
    ----------
    inp : np.ndarray
        Input differential particle flux.
    units : str
        Units of the input differential particle flux.
    mass_ratio : float
        Mass ratio of the species.

    Returns
    -------
    tmp_data : np.ndarray
        Phase space density data.

    Raises
    ------
    ValueError
        If the input unit is not supported.

    """
    fact = 1e6 * 0.53707 * mass_ratio**2

    if units.lower() == "1/(cm^2 s sr kev)":
        tmp_data = inp * 1e-3 / 1e18 * fact
    elif units.lower() == "1/(cm^2 s sr ev)":
        tmp_data = inp / 1e18 * fact
    else:
        raise ValueError("Invalid unit")

    return tmp_data


def dpf2psd(inp: Union[Dataset, DataArray]) -> Union[Dataset, DataArray]:
    r"""Compute phase space density from differential particle flux.

    Parameters
    ----------
    inp : DataArray or Dataset
        Time series of the differential particle flux in
        [(cm^{2} s sr keV)^{-1}].

    Returns
    -------
    psd : DataArray or Dataset
        Time series of the phase space density in [s^{3} m^{-6}].

    Raises
    ------
    TypeError
        If the input type is not supported.

    """
    if isinstance(inp, Dataset):
        tmp_data = _convert(inp.data.data, inp.data.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        energy_mat = np.tile(energy[:, :, None, None], (1, 1, *tmp_data.shape[2:]))
        tmp_data /= energy_mat
        out = inp.copy()
        out.data.data = np.squeeze(tmp_data)
        out.data.attrs["UNITS"] = "s^3/m^6"
    elif isinstance(inp, DataArray):
        tmp_data = _convert(inp.data, inp.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        energy_mat = np.tile(energy, (tmp_data.shape[0], 1))
        tmp_data /= energy_mat
        out = inp.copy()
        out.data = np.squeeze(tmp_data)
        out.attrs["UNITS"] = "s^3/m^6"
    else:
        raise TypeError("Invalid input type")

    return out
