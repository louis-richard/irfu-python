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
import xarray as xr


def vdf_omni(vdf):
    """Computes omnidirectional distribution, conserving unit.

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
        Time series of the omnidirectional velocity distribution function with :
            * time : Time samples.
            * data : Omnidirectional velocity distribution.
            * energy : Energy levels.

    """

    time = vdf.time.data

    energy = vdf.energy.data
    thetas = vdf.theta.data
    dangle = np.pi / 16
    np_phi = 32

    sine_theta = np.ones((np_phi, 1)) * np.sin(thetas * np.pi / 180)
    solid_angles = dangle * dangle * sine_theta
    all_solid_angles = np.tile(solid_angles, (len(time), energy.shape[1], 1, 1))

    dist = vdf.data.data * all_solid_angles
    omni = np.squeeze(np.nanmean(np.nanmean(dist, axis=3), axis=2))
    omni /= np.mean(np.mean(solid_angles))

    energy = np.mean(energy[:2, :], axis=0)

    out = xr.DataArray(omni, coords=[time, energy], dims=["time", "energy"], attrs=vdf.attrs)

    return out
