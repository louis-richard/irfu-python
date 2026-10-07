#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from pyrfu.mms._psd_units import _data_and_units, _energy, _mass_ratio, _output

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


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
    inp : xarray.Dataset or xarray.DataArray
        Phase-space density in s^3/cm^6, s^3/m^6 or s^3/km^6: a distribution
        (Dataset with a data variable, e.g. a skymap or a pitch-angle
        distribution) or a spectrum (DataArray), with (energy,) or
        (time, energy) energies in eV.

    Returns
    -------
    dpf : xarray.Dataset or xarray.DataArray
        Differential particle flux in 1/(cm^2 s sr keV), with the shape of
        inp.

    Raises
    ------
    TypeError
        If inp is not a xarray.Dataset or xarray.DataArray.
    ValueError
        If the species or the units are not supported.

    """

    data, units = _data_and_units(inp)
    data = _convert(data, units, _mass_ratio(inp)) * _energy(inp, data) * 1e3

    return _output(inp, data, "1/(cm^2 s sr keV)")
