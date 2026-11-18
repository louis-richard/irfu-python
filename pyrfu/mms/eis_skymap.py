#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.ts_skymap import ts_skymap
from .vdf_convert import _DPF, vdf_convert


def eis_skymap(inp_ang_ang, to_psd: bool = True):
    r"""Construct skymap distribution from angle-angle distribution.

    Parameters
    ----------
    inp_ang_ang : xarray.DataArray
        EIS angle-angle distribution.
    to_psd : bool, Optional
        Flag to convert differential particle flux to phase space density, in
        s^3/cm^6 as the FPI distributions and :func:`pyrfu.mms.feeps_skymap`.
        Default is True.

    Returns
    -------
    out : xarray.Dataset
        EIS skymap distribution, with the energies and their widths
        (``delta_energy_minus``, ``delta_energy_plus``) in eV.

    Raises
    ------
    ValueError
        With `to_psd`, if the ``UNITS`` of `inp_ang_ang` are not those of a
        differential particle flux, or missing, as
        :func:`pyrfu.mms.feeps_skymap`.

    See Also
    --------
    pyrfu.mms.eis_ang_ang

    """

    # Units of the distribution, None without
    units = inp_ang_ang.attrs.get("UNITS")

    if to_psd and str(units).lower() not in _DPF:
        raise ValueError(
            f"Phase-space density needs differential particle fluxes, not {units}"
        )

    time_data = inp_ang_ang.time.data
    energy = np.tile(inp_ang_ang.energy.data, (len(time_data), 1))
    phi = np.tile(inp_ang_ang.phi.data, (len(time_data), 1))
    theta = inp_ang_ang.theta.data

    a = inp_ang_ang.attrs
    a = list(
        filter(
            lambda k: k not in ["delta_energy_plus", "delta_energy_minus"],
            a,
        ),
    )
    attrs = {k: inp_ang_ang.attrs[k] for k in a}
    coords_attrs = {
        k: {**inp_ang_ang[k].attrs} for k in ["time", "energy", "phi", "theta"]
    }
    coords_attrs["energy"]["UNITS"] = "eV"

    # The global attributes of the files, if any
    glob_attrs = inp_ang_ang.attrs.get("GLOBAL", {})
    # Energy widths in eV, as the energies
    glob_attrs = {
        "delta_energy_plus": 1e3 * np.asarray(inp_ang_ang.attrs["delta_energy_plus"]),
        "delta_energy_minus": 1e3 * np.asarray(inp_ang_ang.attrs["delta_energy_minus"]),
        "species": inp_ang_ang.attrs["species"],
        **glob_attrs,
    }

    # Same energy table at all times
    energy0 = energy[0, :]
    energy1 = energy[0, :]
    e_step_table = np.zeros(len(time_data))

    out = ts_skymap(
        time_data,
        inp_ang_ang.data,
        energy,
        phi,
        theta,
        energy0=energy0,
        energy1=energy1,
        esteptable=e_step_table,
        attrs=attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )
    # out.attrs["species"] = "ions"
    # out.attrs["UNITS"] = "1/(cm^2 s sr keV)"

    out.energy.data *= 1e3

    # Phase-space density in s^3/cm^6, as FPI and feeps_skymap
    if to_psd:
        out = vdf_convert(out, "s^3/cm^6")

    return out
