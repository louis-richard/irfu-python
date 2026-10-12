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
    _data_and_units,
    _energy,
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
    fact = 1e6 * 0.53707 * mass_ratio**2

    if units.lower() == "1/(cm^2 s sr kev)":
        tmp_data = inp * 1e-3 / 1e18 * fact
    elif units.lower() == "1/(cm^2 s sr ev)":
        tmp_data = inp / 1e18 * fact
    else:
        raise ValueError("Invalid unit")

    return tmp_data


def dpf2psd(
    inp: Union[Dataset, DataArray], relativistic: bool = False
) -> Union[Dataset, DataArray]:
    r"""Compute phase space density from differential particle flux.

    Parameters
    ----------
    inp : DataArray or Dataset
        Time series of the differential particle flux in
        [(cm^{2} s sr keV)^{-1}].
    relativistic : bool, Optional
        Phase-space density in velocity space with the relativistic speeds,
        f = gamma^3 m J / v^2, whose integral over the velocities (with
        relativistic speeds, e.g. `psd_moments` with ``relativistic=True`` or
        `reduce`) is the density. Default is False: f = m J / v^2 with the
        non-relativistic speed, as FPI and irfu-matlab. For electrons, f is
        larger by 4.5 % at 5 keV, 52 % at 50 keV and a factor 2.2 at 100 keV
        (the moments much less, as the speeds are smaller).

    Returns
    -------
    psd : DataArray or Dataset
        Time series of the phase space density in [s^{3} m^{-6}].

    Raises
    ------
    TypeError
        If the input type is not supported.

    """

    data, units = _data_and_units(inp)
    data = _convert(data, units, _mass_ratio(inp)) / _energy(inp, data)

    if relativistic:
        data = data * _relativistic_factor(inp, data)

    return _output(inp, data, "s^3/m^6")
