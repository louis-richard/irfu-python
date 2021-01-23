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

from . import ts_vec_xyz


def cart2sph(inp, direction_flag=1):
    """Computes magnitude, theta and phi angle from column vector xyz (first column is x ....)
    theta is 0 at equator. direction_flag = -1  -> to make transformation in opposite direction

    Parameters
    ----------
    inp : xarray.DataArray
        Time series to convert.

    direction_flag : int {1, -1}
        Set to 1 (default) to transform from cartesian to spherical coordinates.
        Set to -1 to transform from spherical to cartesian coordinates.

    Returns
    -------
    out : xarray.DataArray
        Input field in spherical/cartesian coordinate system.

    """

    if inp.attrs["TENSOR_ORDER"] != 1 or inp.data.ndim != 2:
        raise TypeError("Input must be vector field")

    xyz = inp.data

    if direction_flag == -1:
        r = xyz[:, 0]

        st = np.sin(xyz[:, 1] * np.pi / 180)
        ct = np.cos(xyz[:, 1] * np.pi / 180)
        sp = np.sin(xyz[:, 2] * np.pi / 180)
        cp = np.cos(xyz[:, 2] * np.pi / 180)

        z = r * st
        x = r * ct * cp
        y = r * ct * sp

        out_data = np.hstack([x, y, z])

    else:
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2

        r = np.sqrt(xy + xyz[:, 2] ** 2)
        t = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
        p = np.arctan2(xyz[:, 1], xyz[:, 0]) * 180 / np.pi

        out_data = np.hstack([r, t, p])

    out = ts_vec_xyz(inp.time.data, out_data, inp.attrs)

    return out
