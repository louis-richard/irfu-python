#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
from xarray import DataArray, Dataset

# Local imports
from pyrfu.mms._internal.psd_units import _data_and_units
from pyrfu.mms.vdf_convert import _DPF, vdf_convert

__all__ = ["dpf2psd"]


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

    _, units = _data_and_units(inp)

    if units.lower() not in _DPF:
        raise ValueError("Invalid unit")

    return vdf_convert(inp, "s^3/m^6")
