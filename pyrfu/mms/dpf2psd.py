#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.mms._psd_units import (
    _MASS_FACTOR,
    _data_and_units,
    _energy,
    _is_relativistic,
    _mass_ratio,
    _output,
    _relativistic_factor,
)

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["dpf2psd"]


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
    fact = 1e6 * _MASS_FACTOR * mass_ratio**2

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

    Notes
    -----
    The masses are in proton masses, as the FPI products: the constant
    m_p^2 / (2 e^2) was 0.53707, with the atomic mass unit as irfu-matlab, and
    the phase-space densities were 1.46 % too small.

    The phase-space density is in the convention of its ``relativistic``
    attribute: the relativistic one, f = gamma^3 m J / v^2 with the
    relativistic speed (the FEEPS skymaps), or without it the
    non-relativistic one, f = m J / v^2 with v^2 = 2 E / m, as in the FPI,
    HPCA and EIS files and irfu-matlab. The attribute is kept, so that the
    conversions are inverse of each other.
    For electrons, the relativistic f is larger by 4.5 % at 5 keV, 52 % at
    50 keV and a factor 2.2 at 100 keV (the moments much less, as the speeds
    are smaller). The ``relativistic`` argument was removed: the convention
    is that of the data.

    """

    data, units = _data_and_units(inp)
    data = _convert(data, units, _mass_ratio(inp)) / _energy(inp, data)

    if _is_relativistic(inp):
        data = data * _relativistic_factor(inp, data)

    return _output(inp, data, "s^3/m^6")
