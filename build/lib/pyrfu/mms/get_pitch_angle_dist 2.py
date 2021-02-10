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

import warnings
import tqdm
import numpy as np
import xarray as xr

from ..pyrf import ts_scalar, time_clip, resample, normalize


def get_pitch_angle_dist(vdf, b_xyz, tint, **kwargs):
    """
    Computes the pitch angle distributions from l1b brst particle data.

    Parameters
    ----------
    vdf : xarray.Dataset
        to fill

    b_xyz : xarray.DataArray
        to fill

    tint : list of str, optional
        Time interval for closeup.

    **kwargs : dict
        Hash table of keyword arguments with :
            * angles : int or float or list of ndarray
                User defined angles

            * meanorsum : str
                Method :
                    * mean : to fill
                    * sum : to fill
                    * sum_weighted : to fill


    Returns
    -------
    pad : xarray.DataArray
        Particle pitch angle distribution


    Examples
    --------
    >>> from pyrfu import mms

    Define time intervals

    >>> tint_long = ["2017-07-24T12:48:34.000", "2017-07-24T12:58:20.000"]
    >>> tint_zoom = ["2017-07-24T12:49:18.000", "2017-07-24T12:49:30.000"]

    Load ions velocity distribution for MMS1

    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint_long, 1)

    Load magnetic field in the spacecraft coordinates system.

    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint_long, 1)

    Compute pitch angle distribution

    >>> options = dict(angles=24)
    >>> pad_i = mms.get_pitch_angle_dist(vdf, b_dmpa, tint_zoom, **options)

    """

    # Default pitch angles. 15 degree angle widths
    angles_v = np.linspace(15, 180, int(180 / 15))
    d_angles = np.median(np.diff(angles_v)) * np.ones(len(angles_v))

    # Method
    mean_or_sum = "mean"

    if "angles" in kwargs:
        if isinstance(kwargs["angles"], (int, float)):
            n_angles = int(kwargs["angles"])  # Make sure input is integer
            d_angles = 180 / n_angles
            angles_v = np.linspace(d_angles, 180, n_angles)
            d_angles = d_angles * np.ones(n_angles)
            print("notice : User defined number of pitch angles.")

        elif isinstance(kwargs["angles"], (list, np.ndarray)):
            angles_v = kwargs["angles"]
            d_angles = np.diff(angles_v)
            angles_v = angles_v[1:]
            print("notice : User defined pitch angle limits.")

        else:
            raise ValueError("angles parameter not understood.")

    if "meanorsum" in kwargs:
        if isinstance(kwargs["meanorsum"], str) and kwargs["meanorsum"] in ["mean", "sum"]:
            mean_or_sum = kwargs["meanorsum"]
        else:
            raise ValueError("meanorsum parameter not understood.")

    pitch_angles = angles_v - d_angles / 2
    n_angles = len(angles_v)

    time = vdf.time.data

    vdf0 = vdf.data.copy()

    if vdf.phi.data.ndim == 1:
        phi = np.tile(vdf.phi.data, (len(time), 1))
        phi = xr.DataArray(phi, coords=[time, np.arange(len(phi))], dims=["time", "idx1"])
    else:
        phi = vdf.phi

    theta = vdf.theta

    if "esteptable" in vdf.attrs.keys():
        step_table = ts_scalar(time, vdf.attrs["esteptable"])
    else:
        step_table = ts_scalar(time, np.zeros(len(time)))

    if "energy0" in vdf.attrs.keys() and "energy1" in vdf.attrs.keys():
        energy0, _ = [vdf.attrs[f"energy{i}"] for i in range(2)]
    else:
        energy0, _ = vdf.energy.data[:2, :]

    if tint is not None:
        b_xyz = time_clip(b_xyz, tint)
        vdf0, phi, step_table = [time_clip(dat, tint) for dat in [vdf0, phi, step_table]]

    time = vdf0.time.data

    # Check size of energy
    n_en, n_phi, n_theta = [len(energy0), len(phi.data[0, :]), len(theta)]

    b_xyz = resample(b_xyz, vdf0)
    b_vec = normalize(b_xyz)

    b_vec_x = np.transpose(np.tile(b_vec.data[:, 0], [n_en, n_phi, n_theta, 1]), [3, 0, 1, 2])
    b_vec_y = np.transpose(np.tile(b_vec.data[:, 1], [n_en, n_phi, n_theta, 1]), [3, 0, 1, 2])
    b_vec_z = np.transpose(np.tile(b_vec.data[:, 2], [n_en, n_phi, n_theta, 1]), [3, 0, 1, 2])

    x_vec = np.zeros((len(time), n_phi, n_theta))
    y_vec = np.zeros((len(time), n_phi, n_theta))
    z_vec = np.zeros((len(time), n_phi, n_theta))

    for i in range(len(time)):
        x_vec[i, ...] = np.dot(-np.cos(np.deg2rad(phi.data[i, None])).T,
                               np.sin(np.deg2rad(theta.data[:, None])).T)
        y_vec[i, ...] = np.dot(-np.sin(np.deg2rad(phi.data[i, None])).T,
                               np.sin(np.deg2rad(theta.data[:, None])).T)
        z_vec[i, ...] = np.dot(-np.ones((n_phi, 1)), np.cos(np.deg2rad(theta.data[:, None])).T)

    if tint is not None:
        energy = time_clip(vdf.energy, tint).data
    else:
        energy = vdf.energy.data

    temp = [np.tile(mat, [n_en, 1, 1, 1]) for mat in [x_vec, y_vec, z_vec]]
    x_mat, y_mat, z_mat = [np.squeeze(np.transpose(mat, [1, 0, 2, 3])) for mat in temp]

    theta_b = np.arccos(x_mat * np.squeeze(b_vec_x) + y_mat * np.squeeze(b_vec_y)
                        + z_mat * np.squeeze(b_vec_z))
    theta_b = np.rad2deg(theta_b)

    dists = [vdf0.data.copy() for _ in range(n_angles)]

    pad_arr = [None] * n_angles

    for i in tqdm.tqdm(np.arange(n_angles)):
        dists[i][theta_b < angles_v[i] - d_angles[i]] = np.nan
        dists[i][theta_b > angles_v[i]] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if mean_or_sum == "mean":
                pad_arr[i] = np.squeeze(np.nanmean(np.nanmean(dists[i], axis=3), axis=2))
            elif mean_or_sum == "sum":
                pad_arr[i] = np.squeeze(np.nansum(np.nansum(dists[i], axis=3), axis=2))
            else:
                raise ValueError("Invalid method")

    pad_arr = np.transpose(np.stack(pad_arr), [1, 0, 2])

    energy = np.mean(energy[:2, :], axis=0)

    pad = xr.Dataset({"data": (["time", "idx0", "idx1"], np.transpose(pad_arr, [0, 2, 1])),
                      "energy": (["time", "idx0"], np.tile(energy, (len(pad_arr), 1))),
                      "theta": (["time", "idx1"], np.tile(pitch_angles, (len(pad_arr), 1))),
                      "time": time,
                      "idx0": np.arange(len(energy)),
                      "idx1": np.arange(len(pitch_angles))})

    pad.attrs = vdf.attrs
    pad.attrs["mean_or_sum"] = mean_or_sum
    pad.attrs["delta_pitchangle_minus"] = d_angles * .5
    pad.attrs["delta_pitchangle_plus"] = d_angles * .5

    return pad
