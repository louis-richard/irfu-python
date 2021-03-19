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

"""get_hpca_pad.py
@author: Louis Richard
"""

import warnings
import numpy as np
import xarray as xr

from astropy.time import Time
from scipy import interpolate
from ..pyrf import time_clip, resample, normalize, ts_scalar


def _hpca_elevations():
    anode_theta = [123.75000, 101.25000, 78.750000, 56.250000, 33.750000,
                   11.250000, 11.250000, 33.750000, 56.250000, 78.750000,
                   101.25000, 123.75000, 146.25000, 168.75000, 168.75000,
                   146.25000]

    anode_theta[6:14] = [anode_val + 180. for anode_val in anode_theta[6:14]]

    return anode_theta


def get_hpca_pad(vdf, saz, aze, b_xyz, elim=None):
    r"""Computes HPCA pitch angle distribution.

    Parameters
    ----------
    vdf : xarray.DataArray
        Ion PSD or flux; [nt, npo16, ner63], looking direction

    saz : xarray.DataArray
        Start index of azimuthal angle; [nt, 1], (0 - 15)

    aze : xarray.DataArray
        Azimuthal angle per energy; [nT, naz16, npo16, ner63]

    b_xyz : xarray.DataArray
        B in dmpa coordinate

    elim : list, optional
        [emin, emax], energy range for PAD

    Returns
    -------
    pad_spec : xarray.DataArray
        PAD spectrum


    Examples
    --------
    >>> pad_ = get_hpca_pad(vdf, saz, aze, elev, ien, b_xyz, elim=[500, 3000])

    """

    # 1. get data
    ien = vdf.ccomp
    elev = _hpca_elevations()

    if elim is None:
        elim = [ien.data[0], ien.data[-1]]

    # azimuthal angle #
    n_az = 16

    # 2. set tint and TLIM(dist & startaz)
    n_start_az = np.where(saz.data == 0)[0][0]
    start_time_match = aze.time.data[0].view("i8") - vdf.time.data[
        n_start_az].view("i8")

    if start_time_match < 1e6:
        warnings.warn("start times of aze and vdf match.", UserWarning)
    else:
        raise ValueError("start times of aze and vdf don't match!")

    n_stop_az = np.where(saz.data == 15)[0][-1]

    if (n_stop_az - n_start_az + 1) // n_az == len(aze):
        warnings.warn("stop times of aze and vdf match.", UserWarning)
    else:
        raise ValueError("stop times of aze and vdf don't match!")

    tint_ = [saz.time.data[n_start_az - 1], saz.time.data[n_stop_az]]
    tint_ = [Time(time, format="datetime64").isot for time in tint_]
    vdf = time_clip(vdf, tint_)

    # 3. compute PAD
    # 3.1. pitchangle
    # Default pitch angles. 15 degree angle widths
    angle_vec = np.linspace(15, 180, 12)

    d_angle = np.median(np.diff(angle_vec)) * np.ones(len(angle_vec))
    pitch_a = angle_vec - d_angle / 2

    # 3.2.data dimension
    n_po, n_en, n_ta, n_ti = len(elev), len(ien), len(aze), len(vdf)

    assert n_ti == n_ta * n_az, "aze and vdf don't match!"

    tt_ = vdf.time.data

    # 3.3.reshape aze to aze_mat
    aze_mat = np.transpose(aze.data, [1, 0, 2, 3])  # [nt, npo, ner]
    aze_mat = np.reshape(aze_mat, [n_ti, n_po, n_en])

    elev_mat = np.tile(elev, (n_ti, n_en, 1))  # [nt, npo, ner]
    elev_mat = np.transpose(elev_mat, [0, 2, 1])
    xx_ = np.sin(np.deg2rad(elev_mat)) * np.cos(np.deg2rad(aze_mat))
    yy_ = np.sin(np.deg2rad(elev_mat)) * np.sin(np.deg2rad(aze_mat))
    zz_ = np.cos(np.deg2rad(elev_mat))

    t0_ = Time(vdf.time.data, format="datetime64").unix
    t0_start = t0_[0]
    t0_ -= t0_start
    tck_ = interpolate.interp1d(np.arange(0, n_en * len(t0_), n_en), t0_,
                                fill_value="extrapolate")
    t1_tt = Time(tck_(np.arange(0, n_en * len(t0_))) + t0_start,
                 format="unix").datetime64
    b_xyz_r = resample(b_xyz, ts_scalar(t1_tt, np.zeros(len(t1_tt))))
    b_xyz_r = normalize(b_xyz_r)

    b_x = np.transpose(np.reshape(b_xyz_r.data[:, 0], [n_en, n_ti]))
    b_x = np.transpose(np.tile(b_x, [n_po, 1, 1]), [1, 0, 2])  # [nt, npo, ner]

    b_y = np.transpose(np.reshape(b_xyz_r.data[:, 1], [n_en, n_ti]))
    b_y = np.transpose(np.tile(b_y, [n_po, 1, 1]), [1, 0, 2])  # [nt, npo, ner]

    b_z = np.transpose(np.reshape(b_xyz_r.data[:, 2], [n_en, n_ti]))
    b_z = np.transpose(np.tile(b_z, [n_po, 1, 1]), [1, 0, 2])  # [nt, npo, ner]

    theta_b = np.arccos(xx_ * b_x + yy_ * b_y + zz_ * b_z)
    theta_b = np.rad2deg(theta_b)

    # 3.5.select dist for PAD
    vdfs_ = [vdf.data.copy() for _ in range(len(angle_vec))]

    vdfs_[0][theta_b > angle_vec[0]] = np.nan

    for i, vdf_ in enumerate(vdfs_):
        vdf_[theta_b < (angle_vec[i] - d_angle[i])] = np.nan
        vdf_[theta_b > angle_vec[i]] = np.nan

    vdfs_[-1][theta_b < (angle_vec[-1] - d_angle[len(angle_vec) - 1])] = np.nan

    # [n_ti, n_po, n_en] --> [n_ti, n_en]
    vdfs_ = [np.squeeze(np.nanmean(vdf_, axis=1)) for vdf_ in vdfs_]

    # average among energy dimension
    i_elim = np.argmin(abs(ien.data - elim[0]))
    e_min = ien.data[i_elim]
    j_elim = np.argmin(abs(ien.data - elim[1]))
    e_max = ien.data[j_elim]
    messsage = f"PSD/pflux pitch angle dist. from {e_min} [eV] to {e_max} [eV]"
    warnings.warn(messsage, UserWarning)

    vdfs_ = [np.nanmean(vdf_[:, i_elim:j_elim], axis=1) for vdf_ in vdfs_]
    vdfs_ = [np.squeeze(vdf_) for vdf_ in vdfs_]  	# [n_ti, n_en] --> [n_ti]
    padd_ = np.transpose(np.stack(vdfs_))  			# [nt, ner, npitcha12]

    # 3.6.make spectrum
    coords = [tt_, pitch_a]
    dims = ["time", "theta"]
    pad_spec = xr.DataArray(padd_, coords=coords, dims=dims)

    return pad_spec
