#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.4.21"
__status__ = "Prototype"

# Shared by the PSD, DEF and DPF unit conversions (def2psd, dpf2psd, psd2def,
# psd2dpf)

_MASS_RATIOS = {
    **dict.fromkeys(["ions", "ion", "protons", "proton"], 1.0),
    **dict.fromkeys(["alphas", "alpha", "helium"], 4.0),
    **dict.fromkeys(
        ["electrons", "electron", "e"], constants.electron_mass / constants.proton_mass
    ),
}


def _mass_ratio(inp):
    r"""Mass of the species of the distribution in proton masses."""
    species = str(inp.attrs["species"])

    if species.lower() not in _MASS_RATIOS:
        raise ValueError(f"Invalid species {species!r}")

    return _MASS_RATIOS[species.lower()]


def _data_and_units(inp):
    r"""Data and units of a distribution (Dataset with a data variable) or of
    a spectrum (DataArray)."""
    if isinstance(inp, Dataset):
        return inp.data.data, inp.data.attrs["UNITS"]

    if isinstance(inp, DataArray):
        return inp.data, inp.attrs["UNITS"]

    raise TypeError("inp must be a xarray.Dataset or xarray.DataArray")


def _energy(inp, data):
    r"""Energies of the distribution, shaped to broadcast against its data:
    (energy,) or (time, energy) energies, with any number of angle dimensions
    after the energy dimension."""
    energy = np.asarray(inp.energy.data, dtype=np.float64)

    if energy.ndim == 1:
        shape = (1, energy.size) + (1,) * (data.ndim - 2)
    else:
        shape = energy.shape + (1,) * (data.ndim - energy.ndim)

    return energy.reshape(shape)


def _output(inp, data, units):
    r"""Copy of the input with the converted data and units, without changing
    the input or its attributes."""
    if isinstance(inp, Dataset):
        out = inp.copy()
        out["data"] = inp.data.copy(data=data)
        out.data.attrs = {**inp.data.attrs, "UNITS": units}
    else:
        out = inp.copy(data=data)
        out.attrs = {**inp.attrs, "UNITS": units}

    return out
