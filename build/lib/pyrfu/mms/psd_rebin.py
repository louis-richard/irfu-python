#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from ..pyrf import calc_dt


def psd_rebin(vdf=None, phi=None, energy0=None, energy1=None, step_table=None):
    """Convert burst mode distribution into 64 energy channel distribution.
    Functions takes the burst mode distribution sampled in two energy tables and converts to a
    single energy table with 64 energy channels. Time resolution is halved and phi angles are
    averaged over adjacent times.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the particle distribution.

    phi : xarray.DataArray
        Time series of the phi angles.

    energy0 : xarray.DataArray or numpy.ndarray
        Energy table 0.

    energy1 : xarray.DataArray or numpy.ndarray
        Energy table 1.

    step_table : xarray.DataArray
        Time series of the stepping table between energies (burst).

    Returns
    -------
    vdf_r : numpy.ndarray
        Rebinned particle distribution.

    phi_r : numpy.ndarray
        Time series of the recalculated phi angle.

    energy_r : numpy.ndarray
        Revised energy table.

    Notes
    -----
    I'm assuming no gaps in the burst data interval. If there is a gap use time_clip before running.
    To be updated later.
    
    """

    assert vdf is not None
    assert phi is not None
    assert energy0 is not None and isinstance(energy0, (xr.DataArray, np.ndarray))
    assert energy1 is not None and isinstance(energy1, (xr.DataArray, np.ndarray))
    assert step_table is not None

    if isinstance(energy0, xr.DataArray):
        energy0 = energy0.data
    else:
        pass

    if isinstance(energy1, xr.DataArray):
        energy1 = energy1.data
    else:
        pass

    step_table = step_table.data

    # Sort energy levels
    energy_r = np.sort(np.hstack([energy0, energy1]))

    # Define new times
    dt = calc_dt(vdf.data)
    new_time = vdf.time.data[:-1:2] + int(dt * 1e9 / 2)

    vdf_r, phi_r = [np.zeros((len(new_time), 64, 32, 16)), np.zeros((len(new_time), 32))]

    phi_s = np.roll(phi.data, 2, axis=1)
    phi_s[:, 0] = phi_s[:, 0] - 360

    time_indices = np.arange(0, len(vdf.time), 2)

    for new_el_num, ii in enumerate(time_indices):
        if phi.data[ii, 0] > phi.data[ii + 1, 0]:
            phi_r[new_el_num, :] = (phi.data[ii, :] + phi_s[ii + 1, :]) / 2

            vdf_temp = np.roll(np.squeeze(vdf.data.data[ii + 1, ...]), 2, axis=1)

            if step_table[ii]:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[ii, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf_temp
            else:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[ii, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf_temp

        else:
            phi_r[new_el_num, :] = (phi.data[ii, :] + phi.data[ii + 1, :]) / 2

            if step_table[ii]:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[ii, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf.data.data[ii + 1, ...]
            else:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[ii + 1, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf.data.data[ii, ...]

    return vdf_r, phi_r, energy_r
