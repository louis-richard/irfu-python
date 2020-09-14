#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
poynting_flux.py

@author : Louis RICHARD
"""

import numpy as np

from astropy.time import Time

from . import calc_fs, cross, dot, normalize, resample, tlim


def poynting_flux(e_xyz=None, b_xyz=None, b0=None):
    """
    Estimates Poynting flux S and Poynting flux along Bo from electric field E and magnetic field B.

    If E and B have different sampling then the lowest sampling is resampled at the highest sampling

    Parameters :
        e_xyz : DataArray
            Time series of the electric field

        b_xyz : DataArray
            Time series of the magnetic field

    Option :
        b0 : DataArray
            Time series of the direction to project the Pointing flux (optional)

    Returns :
        s : DataArray
            Time series of the Pointing flux

        s_z : DataArray
            Time series of the projection of the Pointing flux (only if b0)

        int_s : DataArray
            Time series of the time integral of the Pointing flux (if b0 integral along b0)

    """

    if e_xyz is None:
        raise ValueError("Pointing_flux requires at least two inputs")

    if b_xyz is None:
        raise ValueError("Pointing_flux requires at least two inputs")

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
    ee, bb = [tlim(e_xyz, tint), tlim(b_xyz, tint)]
    
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

        int_s_z = ss_z
        int_s_z = np.cumsum(ss_z) / fs
        return s, s_z, int_s_z

    if flag_int_s:  # time integral of all Poynting flux components
        ss = s
        idx = np.isnan(s[:, 2].data)
        ss[idx] = 0  # set to zero points where Sz=NaN
        int_s = ss
        int_s = np.cumsum(ss) / fs
        s_z = int_s
        return s, int_s
