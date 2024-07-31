#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from ..pyrf.ts_skymap import ts_skymap

# Local imports
from .psd_rebin import psd_rebin

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def vdf_to_e64(vdf_e32):
    r"""Recompile data into 64 energy channels. Time resolution is halved.
    Only applies to skymap.

    Parameters
    ----------
    vdf_e32 : xarray.Dataset
        Time series of the particle distribution with 32 energy channels.

    Returns
    -------
    vdf_e64 : xarray.Dataset
        Time series of the particle distribution with 32 energy channels.

    """

    time_r, vdf_r, energy_r, phi_r = psd_rebin(
        vdf_e32,
        vdf_e32.phi.data,
        vdf_e32.attrs.get("energy0"),
        vdf_e32.attrs.get("energy1"),
        vdf_e32.attrs.get("esteptable"),
    )

    energy_r = np.tile(energy_r, (len(vdf_r), 1))

    # Data attributes
    data_attrs = vdf_e32.data.attrs

    # Coordinates attributes
    coords_attrs = {k: vdf_e32[k].attrs for k in ["time", "energy", "phi", "theta"]}

    # Global attributes
    glob_attrs = vdf_e32.attrs

    # update delta_energy
    if "delta_energy_plus" in glob_attrs and "delta_energy_minus" in glob_attrs:
        log_energy = np.log10(energy_r[0, :])
        log10_energy = np.diff(log_energy)
        log10_energy_plus = log_energy + 0.5 * np.hstack(
            [log10_energy, log10_energy[-1]],
        )
        log10_energy_minus = log_energy - 0.5 * np.hstack(
            [log10_energy[0], log10_energy],
        )

        energy_plus = 10**log10_energy_plus
        energy_minus = 10**log10_energy_minus
        delta_energy_plus = energy_plus - energy_r
        delta_energy_minus = abs(energy_minus - energy_r)
        delta_energy_plus[-1] = np.max(vdf_e32.attrs["delta_energy_minus"][:, -1])
        delta_energy_minus[0] = np.min(vdf_e32.attrs["delta_energy_minus"][:, 0])

        glob_attrs["delta_energy_plus"] = delta_energy_plus
        glob_attrs["delta_energy_minus"] = delta_energy_minus

    vdf_e64 = ts_skymap(
        time_r,
        vdf_r,
        energy_r,
        phi_r,
        vdf_e32.theta.data,
        energy0=energy_r[0, :],
        energy1=energy_r[0, :],
        esteptable=np.zeros(len(time_r)),
        attrs=data_attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )

    return vdf_e64
