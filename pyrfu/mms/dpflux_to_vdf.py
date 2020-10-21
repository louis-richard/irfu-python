#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dpflux_to_vdf.py

@author : Louis RICHARD
"""


import numpy as np
import xarray as xr

from astropy import constants

from .spectr_to_dataset import spectr_to_dataset


def dpflux_to_vdf(dpflux=None):
    """Compute density particle flux from velocity distribution function.

    Parameters
    ----------
    dpflux : xarray.Dataset or xarray.DataArray
        Time series of the density particle flux in 1/(cm^2 s sr keV).

    Returns
    -------
    vdf : xarray.Dataset
        Time series of the velocity distribution function. Units must be either s^3/m^6.

    """

    assert dpflux is not None and isinstance(dpflux, (xr.Dataset, xr.DataArray))

    if isinstance(dpflux, xr.DataArray):
        dpflux = spectr_to_dataset(dpflux)

    if dpflux.attrs["species"] in ["ions", "i"]:
        mm = 1
    elif dpflux.attrs["species"] in ["electrons", "e"]:
        mm = constants.m_e.value / constants.m_p.value
    else:
        raise ValueError("Invalid specie")

    if dpflux.attrs["UNITS"] == "1/(cm^2 s sr keV)":
        tmp_data = dpflux.data.data / 1e12 * 0.53707 * mm ** 2
    else:
        raise ValueError("Invalid unit")

    energy = dpflux.energy.data

    if tmp_data.ndim == 2:
        tmp_data = tmp_data[:, :, None, None]

    data_r = np.reshape(tmp_data,
                        (tmp_data.shape[0], tmp_data.shape[1], np.prod(tmp_data.shape[2:])))

    if energy.ndim == 1:
        energy_mat = np.tile(energy, (len(dpflux.time), np.prod(tmp_data.shape[2:]), 1))
        energy_mat = np.transpose(energy_mat, [0, 2, 1])
    elif energy.ndim == 2:
        energy_mat = np.tile(energy, (np.prod(tmp_data.shape[2:]), 1, 1))
        energy_mat = np.transpose(energy_mat, [1, 2, 0])
    else:
        raise ValueError("Invalid energy shape")

    data_r /= energy_mat
    tmp_data = np.reshape(data_r, tmp_data.shape)

    dpflux = dpflux.copy()
    dpflux.data.data = np.squeeze(tmp_data)
    dpflux.attrs["UNITS"] = "s^3/m^6"

    return dpflux
