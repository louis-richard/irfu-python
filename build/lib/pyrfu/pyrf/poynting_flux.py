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

from astropy.time import Time

from . import calc_fs, cross, dot, normalize, resample, time_clip


def poynting_flux(e_xyz, b_xyz, b0):
    """
    Estimates Poynting flux at electric field sampling as

    .. math::

        \\mathbf{S} = \\frac{\\mathbf{E}\\times\\mathbf{B}}{\\mu_0}

    if `b0` is given project the Poynting flux along `b0`


    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field.

    b_xyz : xarray.DataArray
        Time series of the magnetic field.

    b0 : xarray.DataArray, optional
        Time series of the direction to project the Pointing flux.

    Returns
    -------
    s : xarray.DataArray
        Time series of the Pointing flux.

    s_z : xarray.DataArray
        Time series of the projection of the Pointing flux (only if b0).

    int_s : xarray.DataArray
        Time series of the time integral of the Pointing flux (if b0 integral along b0).

    """

    # check which Poynting flux to calculate
    flag_s_z, flag_int_s_z, flag_int_s = [False, False, False]

    if b0 is None:
        flag_int_s = True
    else:
        flag_s_z, flag_int_s_z = [True, True]

    # resample if necessary
    fs_b, fs_e = [calc_fs(b_xyz), calc_fs(e_xyz)]

    # interval where both E & B exist
    tmin = Time(max([min(e_xyz.time.data), min(b_xyz.time.data)]), format="datetime64").iso
    tmax = Time(min([max(e_xyz.time.data), max(b_xyz.time.data)]), format="datetime64").iso
    tint = [tmin, tmax]
    ee, bb = [time_clip(e_xyz, tint), time_clip(b_xyz, tint)]
    
    if fs_e < fs_b:
        e = resample(ee, bb)
        b = bb
        fs = fs_b
    elif fs_e > fs_b:
        b = resample(bb, ee)
        e = ee
        fs = fs_e
    else:
        e = ee
        b = bb
        print("assuming the same sampling. Interpolating B and E to 2x E sampling.")

    """
    else
      disp('assuming the same sampling. Interpolating B and E to 2x E sampling.');
      t=sort([ee(:,1);ee(:,1)+0.5/fs_e])
      e=irf_resamp(ee,t)
      b=irf_resamp(bb,t);Fs=2*fs_e;
    end
    """

    # Calculate Poynting flux
    s = cross(e, b) / (4 * np.pi / 1e7) * 1e-9

    if flag_s_z:
        b_m = resample(b0, e)
        s_z = dot(normalize(b_m), s)

    # time integral of Poynting flux along ambient magnetic field
    if flag_int_s_z:
        ss_z = s_z

        idx = np.isnan(s_z.data)

        ss_z[idx] = 0  # set to zero points where Sz=NaN

        int_s_z = np.cumsum(ss_z) / fs
        return s, s_z, int_s_z

    if flag_int_s:  # time integral of all Poynting flux components
        ss = s
        idx = np.isnan(s[:, 2].data)
        ss[idx] = 0  # set to zero points where Sz=NaN

        int_s = np.cumsum(ss) / fs
        return s, int_s
