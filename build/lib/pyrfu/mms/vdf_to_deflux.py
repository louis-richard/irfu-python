#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vdf_to_deflux.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from astropy import constants


def vdf_to_deflux(vdf=None):
    """Changes units to differential energy flux.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the 3D velocity distribution with :
            * time : Time samples.
            * data : 3D velocity distribution.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    Returns
    -------
    out : xarray.Dataset
        Time series of the 3D density energy flux with :
            * time : Time samples.
            * data : 3D density energy flux.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    """

    assert vdf is not None and isinstance(vdf, xr.Dataset)

    if vdf.attrs["species"] in ["ions", "i"]:
        mm = 1
    elif vdf.attrs["species"] in ["electrons", "e"]:
        mm = constants.m_e.value / constants.m_p.value
    else:
        raise ValueError("Invalid specie")

    if vdf.attrs["UNITS"].lower() == "s^3/cm^6":
        tmp_data = vdf.data.data * 1e30 / (1e6 * 0.53707 * mm ** 2)
    elif vdf.attrs["UNITS"].lower() == "s^3/m^6":
        tmp_data = vdf.data.data * 1e18 / (1e6 * 0.53707 * mm ** 2)
    elif vdf.attrs["UNITS"].lower() == "s^3/km^6":
        tmp_data = vdf.data.data / (1e6 * 0.53707 * mm ** 2)
    else:
        raise ValueError("Invalid unit")

    energy = vdf.energy.data

    if tmp_data.ndim == 2:
        tmp_data = tmp_data[:, :, None, None]

    data_r = np.reshape(tmp_data,
                        (tmp_data.shape[0], tmp_data.shape[1], np.prod(tmp_data.shape[2:])))

    if energy.ndim == 1:
        energy_mat = np.tile(energy, (len(vdf.time), np.prod(tmp_data.shape[2:]), 1))
        energy_mat = np.transpose(energy_mat, [0, 2, 1])
    elif energy.ndim == 2:
        energy_mat = np.tile(energy, (np.prod(tmp_data.shape[2:]), 1, 1))
        energy_mat = np.transpose(energy_mat, [1, 2, 0])
    else:
        raise ValueError("Invalid energy shape")

    data_r *= energy_mat ** 2
    tmp_data = np.reshape(data_r, tmp_data.shape)

    out = vdf.copy()
    out.data.data = np.squeeze(tmp_data)
    out.attrs["UNITS"] = "keV/(cm^2 s sr keV)"

    return out
