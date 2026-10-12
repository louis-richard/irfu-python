#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from ._psd_units import _data_and_units, _output, _relativistic_factor
from .def2psd import def2psd
from .dpf2psd import dpf2psd
from .psd2def import psd2def
from .psd2dpf import psd2dpf

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"

# Phase-space densities: factor to s^3/m^6
_PSD = {"s^3/cm^6": 1e12, "s^3/m^6": 1.0, "s^3/km^6": 1e-18}

# Differential energy fluxes (all the same quantity)
_DEF = {
    "kev/(cm^2 s sr kev)": "keV/(cm^2 s sr keV)",
    "ev/(cm^2 s sr ev)": "eV/(cm^2 s sr eV)",
}

# Differential particle fluxes: factor to 1/(cm^2 s sr keV)
_DPF = {"1/(cm^2 s sr kev)": 1.0, "1/(cm^2 s sr ev)": 1e3}

_NAMES = {
    **{k: k for k in _PSD},
    **_DEF,
    "1/(cm^2 s sr kev)": "1/(cm^2 s sr keV)",
    "1/(cm^2 s sr ev)": "1/(cm^2 s sr eV)",
}


def _check(units):
    if units.lower() not in _NAMES:
        raise ValueError(f"Units not supported: {units}")

    return units.lower()


def _to_psd_si(inp, units, relativistic):
    # Phase-space density in s^3/m^6
    data, _ = _data_and_units(inp)

    if units in _PSD:
        return _output(inp, data * _PSD[units], "s^3/m^6")

    if units in _DEF:
        out = def2psd(inp)

        if relativistic:
            out = _output(
                out, out.data * _relativistic_factor(inp, out.data), "s^3/m^6"
            )

        return out

    return dpf2psd(inp, relativistic=relativistic)


def _from_psd_si(psd, units, relativistic):
    # From the phase-space density in s^3/m^6 to units
    data, _ = _data_and_units(psd)

    if units in _PSD:
        return _output(psd, data / _PSD[units], _NAMES[units])

    if units in _DEF:
        if relativistic:
            psd = _output(psd, data / _relativistic_factor(psd, data), "s^3/m^6")

        out = psd2def(psd)
        return _output(out, _data_and_units(out)[0], _NAMES[units])

    out = psd2dpf(psd, relativistic=relativistic)
    return _output(out, _data_and_units(out)[0] / _DPF[units], _NAMES[units])


def vdf_convert(vdf, units: str, relativistic: bool = None):
    r"""Converts a distribution or a spectrum to other units, as irfu-matlab
    PDist.convertto.

    Parameters
    ----------
    vdf : xarray.Dataset or xarray.DataArray
        Distribution (skymap, pitch-angle distribution, omni-directional
        Dataset) or spectrum (DataArray), with the ``species`` attribute and
        the energies in eV.
    units : str
        New units: a phase-space density ("s^3/cm^6", "s^3/m^6",
        "s^3/km^6"), a differential energy flux ("keV/(cm^2 s sr keV)" or
        "eV/(cm^2 s sr eV)") or a differential particle flux
        ("1/(cm^2 s sr keV)", "1/(cm^2 s sr eV)"). The input units are taken
        from ``UNITS`` and are one of these (case-insensitive).
    relativistic : bool, Optional
        The phase-space density is the relativistic one (see `dpf2psd`).
        Default is the ``relativistic`` attribute of `vdf` if any (True for the
        FEEPS skymaps), else False. It changes only the conversions between
        phase-space densities and fluxes.

    Returns
    -------
    out : xarray.Dataset or xarray.DataArray
        Copy of `vdf` in the new units.

    Raises
    ------
    ValueError
        If the input or new units, or the species, are not supported.

    See Also
    --------
    pyrfu.mms.dpf2psd, pyrfu.mms.def2psd, pyrfu.mms.psd2dpf, pyrfu.mms.psd2def

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> def_e = mms.vdf_convert(vdf_e, "keV/(cm^2 s sr keV)")

    """
    _, old_units = _data_and_units(vdf)
    old_units, new_units = _check(old_units), _check(units)

    if relativistic is None:
        relativistic = vdf.attrs.get("relativistic", False)

    if old_units == new_units:
        return _output(vdf, _data_and_units(vdf)[0].copy(), _NAMES[new_units])

    psd = _to_psd_si(vdf, old_units, bool(relativistic))
    return _from_psd_si(psd, new_units, bool(relativistic))
