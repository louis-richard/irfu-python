#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.


import numpy as np

from scipy import constants


def vdf_to_dpflux(vdf):
    """Compute differential particle flux from velocity distribution function.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the velocity distribution function with :
            * time : Time samples.
            * data : 3D velocity distribution.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    Returns
    -------
    dpflux : xarray.Dataset
        Time series of the 3D differential particle flux with :
            * time : Time samples.
            * data : 3D density particle flux.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    """

    if vdf.attrs["species"] in ["ions", "i"]:
        mass_ratio = 1
    elif vdf.attrs["species"] in ["electrons", "e"]:
        mass_ratio = constants.electron_mass / constants.proton_mass
    else:
        raise ValueError("Invalid specie")

    if vdf.attrs["UNITS"].lower() == "s^3/cm^6":
        tmp_data = vdf.data.data * 1e30 / (1e6 * 0.53707 * mass_ratio ** 2)
    elif vdf.attrs["UNITS"].lower() == "s^3/m^6":
        tmp_data = vdf.data.data * 1e18 / (1e6 * 0.53707 * mass_ratio ** 2)
    elif vdf.attrs["UNITS"].lower() == "s^3/km^6":
        tmp_data = vdf.data.data / (1e6 * 0.53707 * mass_ratio ** 2)
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

    data_r *= energy_mat
    tmp_data = np.reshape(data_r, tmp_data.shape)

    dpflux = vdf.copy()
    dpflux.data.data = np.squeeze(tmp_data)
    dpflux.attrs["UNITS"] = "1/(cm^2 s sr eV)"

    return dpflux
