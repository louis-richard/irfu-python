#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf import ts_skymap

from .psd_rebin import psd_rebin

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
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

    time_r, vdf_r, energy_r, phi_r = psd_rebin(vdf_e32, vdf_e32.phi,
                                               vdf_e32.attrs.get("energy0"),
                                               vdf_e32.attrs.get("energy1"),
                                               vdf_e32.attrs.get("esteptable"))

    energy_r = np.tile(energy_r, (len(vdf_r), 1))

    vdf_e64 = ts_skymap(time_r, vdf_r, energy_r, phi_r, vdf_e32.theta.data,
                        energy0=energy_r[0, :], energy1=energy_r[0, :],
                        esteptable=vdf_e32.attrs.get("esteptable"))

    # update delta_energy
    log_energy = np.log10(energy_r[0, :])
    log10_energy = np.diff(log_energy)
    log10_energy_plus = log_energy + 0.5 * np.hstack([log10_energy,
                                                      log10_energy[-1]])
    log10_energy_minus = log_energy - 0.5 * np.hstack([log10_energy[0],
                                                       log10_energy])

    energy_plus = 10 ** log10_energy_plus
    energy_minus = 10 ** log10_energy_minus
    delta_energy_plus = energy_plus - energy_r
    delta_energy_minus = abs(energy_minus - energy_r)
    delta_energy_plus[-1] = np.max(vdf_e32.attrs["delta_energy_minus"][:, -1])
    delta_energy_minus[0] = np.min(vdf_e32.attrs["delta_energy_minus"][:, 0])
    vdf_e64.attrs["delta_energy_plus"] = delta_energy_plus
    vdf_e64.attrs["delta_energy_minus"] = delta_energy_minus
    vdf_e64.attrs["esteptable"] = np.zeros(len(time_r))
    vdf_e64.attrs["species"] = vdf_e32.attrs["species"]
    vdf_e64.attrs["UNITS"] = vdf_e32.attrs["UNITS"]

    return vdf_e64
