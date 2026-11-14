#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from pyrfu.mms._internal.psd_units import (
    _MASS_FACTOR,
    _data_and_units,
    _energy,
    _is_relativistic,
    _mass_ratio,
    _output,
    _relativistic_factor,
)

# The units of the distributions, in one place (each of psd2def, psd2dpf,
# def2psd and dpf2psd had its own conversion, two of them the same copy).
# Phase-space densities: factor to s^3/m^6
_PSD = {"s^3/cm^6": 1e12, "s^3/m^6": 1.0, "s^3/km^6": 1e-18}

# Differential energy fluxes (all the same quantity; "1/(cm^2 s sr)" is the
# name def2psd accepted)
_DEF = {
    "kev/(cm^2 s sr kev)": "keV/(cm^2 s sr keV)",
    "ev/(cm^2 s sr ev)": "eV/(cm^2 s sr eV)",
    "1/(cm^2 s sr)": "1/(cm^2 s sr)",
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
    # Lower-case units, if supported
    if units.lower() not in _NAMES:
        raise ValueError(f"Units not supported: {units}")

    return units.lower()


def _mass_term(inp):
    # 1e6 m^2 / (2 e^2) (masses in proton masses) of the fluxes
    return 1e6 * _MASS_FACTOR * _mass_ratio(inp) ** 2


def _to_psd_si(inp, units):
    # Phase-space density in s^3/m^6, in the convention of inp (relativistic
    # from a flux if inp is)
    data, _ = _data_and_units(inp)

    if units in _PSD:
        return _output(inp, data * _PSD[units], "s^3/m^6")

    if units in _DEF:
        data = data / 1e18 * _mass_term(inp) / _energy(inp, data) ** 2
    else:
        scale = 1e-3 if units == "1/(cm^2 s sr kev)" else 1.0
        data = data * scale / 1e18 * _mass_term(inp) / _energy(inp, data)

    if _is_relativistic(inp):
        data = data * _relativistic_factor(inp, data)

    return _output(inp, data, "s^3/m^6")


def _from_psd_si(psd, units):
    # From the phase-space density in s^3/m^6 to units (from the
    # non-relativistic one to a flux)
    data, _ = _data_and_units(psd)

    if units in _PSD:
        return _output(psd, data / _PSD[units], _NAMES[units])

    if units in _DEF:
        if _is_relativistic(psd):
            data = data / _relativistic_factor(psd, data)

        data = data * 1e18 * (1 / _mass_term(psd)) * _energy(psd, data) ** 2
    else:
        data = data * 1e18 * (1 / _mass_term(psd)) * _energy(psd, data) * 1e3

        if _is_relativistic(psd):
            data = data / _relativistic_factor(psd, data)

        data = data / _DPF[units]

    return _output(psd, data, _NAMES[units])


def _relativistic_psd(vdf):
    r"""Relativistic phase-space density (s^3/m^6, array) of a distribution:
    its data in s^3/m^6, times 2 gamma^5 / (gamma + 1) at the energies of its
    bins if they are the non-relativistic one (FPI, HPCA, EIS), for the
    integrals over the velocities with the relativistic speeds."""
    data = _data_and_units(vdf_convert(vdf, "s^3/m^6"))[0]

    if _is_relativistic(vdf):
        return data

    return data * _relativistic_factor(vdf, data)


def _in_convention(out, relativistic: bool):
    r"""Distribution `out` whose data are a relativistic phase-space density,
    in the convention `relativistic` (divided by 2 gamma^5 / (gamma + 1) at
    its energies for the non-relativistic one), with the attribute."""
    data, units = _data_and_units(out)
    out = out.copy()

    if relativistic:
        out.attrs["relativistic"] = True
    else:
        out = _output(out, data / _relativistic_factor(out, data), units)
        out.attrs.pop("relativistic", None)

    return out


def vdf_convert(vdf, units: str):
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

    Returns
    -------
    out : xarray.Dataset or xarray.DataArray
        Copy of `vdf` in the new units.

    Raises
    ------
    ValueError
        If the input or new units, or the species, are not supported.

    Notes
    -----
    The phase-space density is in the convention of the ``relativistic``
    attribute of `vdf` (see `dpf2psd`), kept by the conversion: the
    relativistic one for the FEEPS skymaps, the non-relativistic one for the
    FPI, HPCA and EIS files.

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

    if old_units == new_units:
        return _output(vdf, _data_and_units(vdf)[0].copy(), _NAMES[new_units])

    psd = _to_psd_si(vdf, old_units)
    return _from_psd_si(psd, new_units)
