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

"""cart2sph.py
@author: Louis Richard
"""

import numpy as np

from .ts_vec_xyz import ts_vec_xyz


def cart2sph(inp, direction_flag=1):
    """Computes magnitude, theta and phi angle from column vector xyz
    (first column is x ....) theta is 0 at equator.
    direction_flag = -1  -> to make transformation in opposite direction

    Parameters
    ----------
    inp : xarray.DataArray
        Time series to convert.

    direction_flag : int {1, -1}
        Set to 1 (default) to transform from cartesian to spherical
        coordinates. Set to -1 to transform from spherical to cartesian
        coordinates.

    Returns
    -------
    out : xarray.DataArray
        Input field in spherical/cartesian coordinate system.

    """

    if inp.attrs["TENSOR_ORDER"] != 1 or inp.data.ndim != 2:
        raise TypeError("Input must be vector field")

    if direction_flag == -1:
        r_data = inp.data[:, 0]

        sin_the = np.sin(np.deg2rad(inp.data[:, 1]))
        cos_the = np.cos(np.deg2rad(inp.data[:, 1]))
        sin_phi = np.sin(np.deg2rad(inp.data[:, 2]))
        cos_phi = np.cos(np.deg2rad(inp.data[:, 2]))

        z_data = r_data * sin_the
        x_data = r_data * cos_the * cos_phi
        y_data = r_data * cos_the * sin_phi

        out_data = np.hstack([x_data, y_data, z_data])

    else:
        xy2 = inp.data[:, 0] ** 2 + inp.data[:, 1] ** 2

        r_data = np.sqrt(xy2 + inp.data[:, 2] ** 2)
        theta = np.rad2deg(np.arctan2(inp.data[:, 2], np.sqrt(xy2)))
        phi = np.rad2deg(np.arctan2(inp.data[:, 1], inp.data[:, 0]))

        out_data = np.hstack([r_data, theta, phi])

    out = ts_vec_xyz(inp.time.data, out_data, inp.attrs)

    return out
