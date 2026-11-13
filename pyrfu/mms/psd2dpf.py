#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from pyrfu.mms._internal.psd_units import _data_and_units
from pyrfu.mms.vdf_convert import _PSD, vdf_convert


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

    Notes
    -----
    The masses are in proton masses, as the FPI products: the constant
    m_p^2 / (2 e^2) was 0.53707, with the atomic mass unit as irfu-matlab, and
    the fluxes were 1.46 % too large.

    The phase-space density is in the convention of its ``relativistic``
    attribute: the relativistic one, f = gamma^3 m J / v^2 with the
    relativistic speed (the FEEPS skymaps), or without it the
    non-relativistic one, f = m J / v^2 with v^2 = 2 E / m, as in the FPI,
    HPCA and EIS files and irfu-matlab. The attribute is kept, so that the
    conversions are inverse of each other.
    The ``relativistic`` argument was removed: the convention is that of the
    data.

    """

    _, units = _data_and_units(inp)

    if units.lower() not in _PSD:
        raise ValueError("Invalid unit")

    return vdf_convert(inp, "1/(cm^2 s sr keV)")
