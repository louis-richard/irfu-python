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

# Masses in proton masses; the alphas (helium) are He2+ ions, 3.9726 (they
# were 4, 0.7 % heavier)
_MASS_RATIOS = {
    **dict.fromkeys(["ions", "ion", "protons", "proton"], 1.0),
    **dict.fromkeys(
        ["alphas", "alpha", "helium"],
        constants.physical_constants["alpha particle mass"][0] / constants.proton_mass,
    ),
    **dict.fromkeys(
        ["electrons", "electron", "e"], constants.electron_mass / constants.proton_mass
    ),
}

# m_p^2 / (2 e^2) in the units of the conversions, 0.54494: the masses are in
# proton masses, as the FPI products. The constant was 0.53707, with the atomic
# mass unit as irfu-matlab: the fluxes were 1.46 % too large.
_MASS_FACTOR = 1e16 * constants.proton_mass**2 / (2 * constants.elementary_charge**2)


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


def _relativistic_factor(inp, data):
    r"""Ratio of the relativistic to the non-relativistic phase-space density
    in velocity space for the same intensity, 2 gamma^5 / (gamma + 1), shaped
    as the energies of `_energy`.

    The non-relativistic conversions use f = m J / v^2 with v^2 = 2 E / m. In
    velocity space, the phase-space density whose integral over d^3v is the
    density is f = gamma^3 m J / v^2 with the relativistic speed v."""
    rest_energy = _mass_ratio(inp) * constants.proton_mass * constants.c**2
    gamma = 1.0 + constants.elementary_charge * _energy(inp, data) / rest_energy
    return 2.0 * gamma**5 / (gamma + 1.0)


def _attrs(attrs, units):
    r"""Attributes with the new units, without the label of the old ones
    (LABLAXIS, e.g. "Intensity" for a phase-space density) if they changed."""
    if attrs.get("UNITS") != units:
        attrs = {k: v for k, v in attrs.items() if k != "LABLAXIS"}

    return {**attrs, "UNITS": units}


def _output(inp, data, units):
    r"""Copy of the input with the converted data and units, without changing
    the input or its attributes."""
    if isinstance(inp, Dataset):
        out = inp.copy()
        out["data"] = inp.data.copy(data=data)
        out.data.attrs = _attrs(inp.data.attrs, units)
    else:
        out = inp.copy(data=data)
        out.attrs = _attrs(inp.attrs, units)

    return out
