#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from ..pyrf.ts_skymap import ts_skymap

# Local imports
from .dpf2psd import dpf2psd

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def eis_skymap(inp_ang_ang, to_psd: bool = True):
    r"""Construct skymap distribution from angle-angle distribution.

    Parameters
    ----------
    inp_ang_ang : xarray.DataArray
        EIS angle-angle distribution.
    to_psd : bool, Optional
        Flag to convert differential particle flux to phase space density.

    Returns
    -------
    out : xarray.Dataset
        EIS skymap distribution.

    See Also
    --------
    pyrfu.mms.eis_ang_ang

    """

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
    coords_attrs = {k: inp_ang_ang[k].attrs for k in ["time", "energy", "phi", "theta"]}

    glob_attrs = inp_ang_ang.attrs["GLOBAL"]
    glob_attrs = {
        "delta_energy_plus": inp_ang_ang.attrs["delta_energy_plus"],
        "delta_energy_minus": inp_ang_ang.attrs["delta_energy_minus"],
        "species": inp_ang_ang.attrs["species"],
        **glob_attrs,
    }

    energy0, energy1 = [energy[i, :] for i in range(2)]
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

    if to_psd:
        out = dpf2psd(out)

    return out
