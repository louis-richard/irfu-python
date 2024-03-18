#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party
import numpy as np
from xarray.core.dataset import Dataset

# Local imports
from ..pyrf.calc_dt import calc_dt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def psd_rebin(
    vdf: Dataset,
    phi: np.ndarray,
    energy0: np.ndarray,
    energy1: np.ndarray,
    esteptable: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    r"""Convert burst mode distribution into 64 energy channel distribution.

    Takes the burst mode distribution sampled in two energy tables and converts to a
    single energy table with 64 energy channels. Time resolution is halved and phi
    angles are averaged over adjacent times.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the particle distribution.
    phi : numpy.ndarray
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

    Raises
    ------
    TypeError
        If vdf is not a xarray.Dataset.
    TypeError
        If phi is not a numpy.ndarray.
    TypeError
        If energy0 is not a numpy.ndarray.
    TypeError
        If energy1 is not a numpy.ndarray.
    TypeError
        If esteptable is not a numpy.ndarray.

    Notes
    -----
    I'm assuming no gaps in the burst data interval. If there is a gap use
    time_clip before running. To be updated later.

    """
    if not isinstance(vdf, Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    if not isinstance(phi, np.ndarray):
        raise TypeError("phi must be a numpy.ndarray")

    if not isinstance(energy0, np.ndarray):
        raise TypeError("energy0 must be a numpy.ndarray")

    if not isinstance(energy1, np.ndarray):
        raise TypeError("energy1 must be a numpy.ndarray")

    if not isinstance(esteptable, np.ndarray):
        raise TypeError("step_table must be a numpy.ndarray")

    # Sort energy levels
    energy_r = np.sort(np.hstack([energy0, energy1]))

    # Define new times
    delta_t = calc_dt(vdf.data)
    time_r = vdf.time.data[:-1:2] + int(delta_t * 1e9 / 2)

    vdf_r = np.zeros((len(time_r), 64, *vdf.data.shape[2:]))
    phi_r = np.zeros((len(time_r), vdf.data.shape[2]))

    phi_s = np.roll(phi, 2, axis=1)
    phi_s[:, 0] = phi_s[:, 0] - 360

    time_indices = np.arange(0, len(vdf.time) - 1, 2)

    for new_el_num, idx in enumerate(time_indices[:-1]):
        if phi[idx, 0] > phi[idx + 1, 0]:
            phi_r[new_el_num, :] = (phi[idx, :] + phi_s[idx + 1, :]) / 2

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
            phi_r[new_el_num, :] = phi[idx, :] + phi[idx + 1, :]
            phi_r[new_el_num, :] /= 2

            if esteptable[idx]:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[idx, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf.data.data[idx + 1, ...]
            else:
                vdf_r[new_el_num, 1:64:2, ...] = vdf.data.data[idx + 1, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf.data.data[idx, ...]

    return time_r, vdf_r, energy_r, phi_r
