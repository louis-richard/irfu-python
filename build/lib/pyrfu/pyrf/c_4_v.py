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

from scipy import interpolate
from astropy.time import Time


def get_vol_ten(r, t):
    """to fill."""

    if len(t) == 1:
        t = np.array([t, t, t, t])

    tckr_x, tckr_y, tckr_z = [[], [], []]

    for i in range(4):
        tckr_x.append(interpolate.interp1d(r[i].time.data, r[i].data[:, 0]))
        tckr_y.append(interpolate.interp1d(r[i].time.data, r[i].data[:, 1]))
        tckr_z.append(interpolate.interp1d(r[i].time.data, r[i].data[:, 2]))

        r[i] = np.array([tckr_x[i](t[0]), tckr_y[i](t[0]), tckr_z[i](t[0])])

    # Volumetric tensor with SC1 as center.
    dr_mat = (np.vstack(r[1:]) - np.tile(r[0], (3, 1))).T

    return dr_mat


def c_4_v(r, x):
    """Calculates velocity or time shift of discontinuity as in [6]_.

    Parameters
    ----------
    r : list of xarray.DataArray
        Time series of the positions of the spacecraft.

    x : list
        Crossing times or time and velocity.

    Returns
    -------
    out : ndarray
        Discontinuity velocity or time shift with respect to mms1.

    References
    ----------
    .. [6]	Vogt, J., Haaland, S., and Paschmann, G. (2011) Accuracy of multi-point boundary
            crossing time analysis, Ann. Geophys., 29, 2239-2252,
            doi : https://doi.org/10.5194/angeo-29-2239-2011

    """

    if isinstance(x, np.ndarray) and x.dtype == np.datetime64:
        flag = "v_from_t"

        x = Time(x, format="datetime64").unix
    elif x[1] > 299792.458:
        flag = "v_from_t"
    else:
        flag = "dt_from_v"

    if flag.lower() == "v_from_t":
        # Time input, velocity output
        t = x
        dr_mat = get_vol_ten(r, t)
        tau = np.array(t[1:]) - t[0]
        m = np.linalg.solve(dr_mat, tau)

        # "1/v vector"
        out = m / np.linalg.norm(m) ** 2

    elif flag.lower() == "dt_from_v":
        # Time and velocity input, time output
        tc = x[0]  # center time
        v = np.array(x[1:])  # Input velocity
        m = v / np.linalg.norm(v) ** 2

        dr_mat = get_vol_ten(r, tc)

        dt = np.matmul(dr_mat, m)
        out = np.hstack([0, dt])

    else:
        raise ValueError("Invalid flag")

    return out
