#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.mms._psd_units import _data_and_units, _energy, _mass_ratio, _output

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def _convert(inp, units, mass_ratio):
    fact = 1e6 * 0.53707 * mass_ratio**2

    if units.lower() in ["kev/(cm^2 s sr kev)", "ev/(cm^2 s sr ev)", "1/(cm^2 s sr)"]:
        tmp_data = inp / 1e18 * fact
    else:
        raise ValueError("Invalid unit")

    return tmp_data


def def2psd(inp: Union[DataArray, Dataset]) -> Union[DataArray, Dataset]:
    r"""Compute phase space density from differential energy flux.

    The phase-space density is given by:

    .. math:

        f(E) = m^2 \frac{DEF}{E^2} * 0.53707,

    where :math:`m` is the particle mass in atomic mass unit, :math:`DEF` is
    the differential energy flux in 1/(cm sr s) and :math:`E` is the energy
    in eV.

    Parameters
    ----------
    inp : xarray.Dataset or xarray.DataArray
        Time series of the differential energy flux in [(cm^{2} s sr)^{-1}].

    Returns
    -------
    psd : xarray.Dataset or xarray.DataArray
        Time series of the phase space density in [s^{3} m^{-6}]

    Raises
    ------
    TypeError
        If inp is not a xarray.Dataset or xarray.DataArray.

    """

    data, units = _data_and_units(inp)
    data = _convert(data, units, _mass_ratio(inp)) / _energy(inp, data) ** 2

    return _output(inp, data, "s^3/m^6")
