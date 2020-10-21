#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_pitch_angle_dist.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from pyrfu.pyrf import ts_scalar, time_clip, resample, normalize


def get_pitch_angle_dist(vdf=None, b_xyz=None, tint=None, **kwargs):
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
            * angles : int or float or list of numpy.ndarray
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
    >>> # Define time interval
    >>> tint_long = ["2017-07-24T12:48:34.000", "2017-07-24T12:58:20.000"]
    >>> # Load ions velocity distribution for MMS1
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint_long, 1)
    >>> # Load magnetic field in the spacecraft coordinates system.
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint_long, 1)
    >>> # Define closeup time interval
    >>> tint_zoom = ["2017-07-24T12:49:18.000", "2017-07-24T12:49:30.000"]
    >>> # Compute pitch angle distribution
    >>> options = dict(anggles=24)
    >>> pad_i = mms.get_pitch_angle_dist(vdf, b_dmpa, tint_zoom, **options)

    """

    assert vdf is not None and isinstance(vdf, xr.Dataset)
    assert b_xyz is not None and isinstance(b_xyz, xr.Dataset)
    assert tint is not None and isinstance(tint, list)
    assert isinstance(tint[0], str) and isinstance(tint[1], str)

    # Default pitch angles. 15 degree angle widths
    angles_v = np.linspace(15, 180, int(180 / 15))
    d_angles = np.median(np.diff(angles_v)) * np.ones(len(angles_v))

    # Method
    mean_or_sum = "mean"

    if "angles" in kwargs:
        if isinstance(kwargs["angles"], (int, float)):
            n_angles = np.floor(kwargs["angles"])  # Make sure input is integer
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

    time = vdf.time.data

    vdf0 = vdf.data

    if vdf.phi.data.ndim == 1:
        phi = np.tile(vdf.phi.data, (len(time), 1))
        phi = xr.DataArray(phi, coords=[time, np.arange(len(phi))], dims=["time", "idx1"])
    else:
        phi = vdf.phi

    theta = vdf.theta.data

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

    x = np.zeros((len(time), n_phi, n_theta))
    y = np.zeros((len(time), n_phi, n_theta))
    z = np.zeros((len(time), n_phi, n_theta))

    for ii in range(len(time)):
        x[ii, ...] = np.dot(-np.cos(phi.data[ii, None] * np.pi / 180).T,
                            np.sin(theta.data[:, None] * np.pi / 180).T)
        y[ii, ...] = np.dot(-np.sin(phi.data[ii, None] * np.pi / 180).T,
                            np.sin(theta.data[:, None] * np.pi / 180).T)
        z[ii, ...] = np.dot(-np.ones((n_phi, 1)), np.cos(theta.data[:, None] * np.pi / 180).T)

    if tint is not None:
        energy = time_clip(vdf.energy, tint).data
    else:
        energy = vdf.energy.data

    xt, yt, zt = [np.tile(mat, [n_en, 1, 1, 1]) for mat in [x, y, z]]
    xt, yt, zt = [np.squeeze(np.transpose(mat, [1, 0, 2, 3])) for mat in [xt, yt, zt]]

    theta_b = np.arccos(xt * np.squeeze(b_vec_x) + yt * np.squeeze(b_vec_y) + zt * np.squeeze(
        b_vec_z)) * 180 / np.pi

    dists = [vdf0.data for _ in range(len(angles_v))]

    pad_arr = []

    for i, dist in enumerate(dists):
        dist[theta_b < angles_v[i] - d_angles[i]] = np.nan
        dist[theta_b > angles_v[i]] = np.nan
        if mean_or_sum == "mean":
            pad_arr.append(np.squeeze(np.nanmean(np.nanmean(dist, axis=3), axis=2)))
        elif mean_or_sum == "sum":
            pad_arr.append(np.squeeze(np.nansum(np.nansum(dist, axis=3), axis=2)))
        else:
            raise ValueError("Invalid method")

    pad_arr = np.stack(pad_arr)
    pad_arr = np.transpose(pad_arr, [1, 0, 2])

    energy = np.mean(energy[:2, :], axis=0)

    pad = xr.DataArray(pad_arr, coords=[time, pitch_angles, energy],
                       dims=["time", "theta", "energy"])

    pad.attrs = vdf.attrs
    pad.attrs["mean_or_sum"] = mean_or_sum
    pad.attrs["delta_pitchangle_minus"] = d_angles * .5
    pad.attrs["delta_pitchangle_plus"] = d_angles * .5

    return pad
