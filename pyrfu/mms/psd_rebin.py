#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party
import numpy as np

# Local imports
from ..pyrf.calc_dt import calc_dt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def psd_rebin(vdf, phi, energy0, energy1, esteptable):
    r"""Converts burst mode distribution into 64 energy channel distribution.
    Functions takes the burst mode distribution sampled in two energy tables
    and converts to a single energy table with 64 energy channels. Time
    resolution is halved and phi angles are averaged over adjacent times.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the particle distribution.
    phi : xarray.DataArray
        Time series of the phi angles.
    energy0 : numpy.ndarray
        Energy table 0.
    energy1 : numpy.ndarray
        Energy table 1.
    esteptable : numpy.ndarray
        Time series of the stepping table between energies (burst).

    Returns
    -------
    time_r : numpy.ndarray
        Revised time steps.
    vdf_r : numpy.ndarray
        Rebinned particle distribution.
    energy_r : numpy.ndarray
        Revised energy table.
    phi_r : numpy.ndarray
        Time series of the recalculated phi angle.

    Notes
    -----
    I'm assuming no gaps in the burst data interval. If there is a gap use
    time_clip before running. To be updated later.

    """

    assert isinstance(energy0, np.ndarray), "energy0 must be a numpy.ndarray"
    assert isinstance(energy1, np.ndarray), "energy1 must be a numpy.ndarray"
    assert isinstance(esteptable, np.ndarray), "step_table must be a numpy.ndarray"

    # Sort energy levels
    energy_r = np.sort(np.hstack([energy0, energy1]))

    # Define new times
    delta_t = calc_dt(vdf.data)
    time_r = vdf.time.data[:-1:2] + int(delta_t * 1e9 / 2)

    vdf_r = np.zeros((len(time_r), 64, *vdf.data.shape[2:]))
    phi_r = np.zeros((len(time_r), vdf.data.shape[2]))

    phi_s = np.roll(phi.data, 2, axis=1)
    phi_s[:, 0] = phi_s[:, 0] - 360

    time_indices = np.arange(0, len(vdf.time) - 1, 2)

    for new_el_num, idx in enumerate(time_indices[:-1]):
        if phi.data[idx, 0] > phi.data[idx + 1, 0]:
            phi_r[new_el_num, :] = (phi.data[idx, :] + phi_s[idx + 1, :]) / 2

            vdf_temp = np.roll(
                np.squeeze(vdf.data.data[idx + 1, ...]),
                2,
                axis=1,
            )

            if esteptable[idx]:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[idx, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf_temp
            else:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[idx, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf_temp

        else:
            phi_r[new_el_num, :] = phi.data[idx, :] + phi.data[idx + 1, :]
            phi_r[new_el_num, :] /= 2

            if esteptable[idx]:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[idx, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf.data.data[idx + 1, ...]
            else:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[idx + 1, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf.data.data[idx, ...]

    return time_r, vdf_r, energy_r, phi_r
