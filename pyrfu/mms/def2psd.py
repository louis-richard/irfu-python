#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports

# 3rd party imports
from xarray import DataArray, Dataset

# Local imports
from pyrfu.mms._internal.psd_units import _data_and_units
from pyrfu.mms.vdf_convert import _DEF, vdf_convert


def def2psd(inp: DataArray | Dataset) -> DataArray | Dataset:
    r"""Compute phase space density from differential energy flux.

    The phase-space density is given by:

    .. math::

        f(E) = m^2 \frac{DEF}{E^2} * 0.54494,

    where :math:`m` is the particle mass in proton masses, :math:`DEF` is
    the differential energy flux in keV/(cm^2 s sr keV) and :math:`E` is the energy
    in eV. The constant is m_p^2 / (2 e^2) with the proton mass, as the FPI
    products; irfu-matlab uses the atomic mass unit (0.53707), which gives
    phase-space densities 1.46 % too small.

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

    Notes
    -----
    The phase-space density is in the convention of the ``relativistic``
    attribute of `inp`: the relativistic one, f = gamma^3 m J / v^2 with the
    relativistic speed (the FEEPS skymaps), or without it the
    non-relativistic one above, as in the FPI, HPCA and EIS files and
    irfu-matlab.

    """

    _, units = _data_and_units(inp)

    if units.lower() not in _DEF:
        raise ValueError("Invalid unit")

    return vdf_convert(inp, "s^3/m^6")
