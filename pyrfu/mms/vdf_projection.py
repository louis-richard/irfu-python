#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
import itertools

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import constants

# Local imports
from ..pyrf import iso86012datetime64, time_clip, ts_scalar, ts_skymap

from .psd_rebin import psd_rebin

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.10"
__status__ = "Prototype"


def _coord_sys(coord_sys):
    x_vec = coord_sys[0, :] / np.linalg.norm(coord_sys[0, :])
    y_vec = coord_sys[1, :] / np.linalg.norm(coord_sys[1, :])

    z_vec = np.cross(x_vec, y_vec) / np.linalg.norm(np.cross(x_vec, y_vec))
    y_vec = np.cross(z_vec, x_vec) / np.linalg.norm(np.cross(z_vec, x_vec))

    changed_xyz = [False, False, False]

    for i, vec, c in zip([0, 1, 2], [x_vec, y_vec, z_vec], ["x", "y", "z"]):
        if abs(np.rad2deg(np.arccos(np.dot(vec, coord_sys[:, i])))) > 1.:
            msg = " ".join([f"In making 'xyz' a right handed orthogonal",
                            f"coordinate system, {c} (in-plane {i:d}) was",
                            "changed from",
                            np.array2string(coord_sys[:, i]),
                            "to",
                            np.array2string(x_vec),
                            "Please verify that this is according to your",
                            "intentions."])
            warnings.warn(msg, UserWarning)
            changed_xyz[i] = True

    return x_vec, y_vec, z_vec, changed_xyz


def _init(vdf, tint):
    assert isinstance(vdf, xr.Dataset), "vdf must be a xarray.Dataset"

    len_e = 32

    if vdf.phi.data.ndim == 1:
        phi = xr.DataArray(np.tile(vdf.phi.data, (len(vdf.data), 1)),
                           coords=[vdf.time.data,
                                   np.arange(len(vdf.phi.data))],
                           dims=["time", "idx"])
    else:
        phi = vdf.phi

    theta = vdf.theta
    polar = np.deg2rad(theta)
    azimuthal = np.deg2rad(phi)
    step_table = vdf.attrs.get("esteptable", np.zeros(len(vdf.time)))

    energy0 = vdf.attrs.get("energy0", vdf.energy.data[0, :])
    energy1 = vdf.attrs.get("energy1", vdf.energy.data[1, :])

    diff_energ = np.median(np.diff(np.log10(energy0))) / 2

    energy0_edges = np.hstack([10 ** (np.log10(energy0) - diff_energ),
                               10 ** (np.log10(energy0[-1]) + diff_energ)])
    energy1_edges = np.hstack([10 ** (np.log10(energy1) - diff_energ),
                               10 ** (np.log10(energy1[-1]) + diff_energ)])

    if tint is not None and len(tint) == 1:
        t_id = np.argmin(
            np.abs(vdf.time.data - iso86012datetime64(np.array(tint))[0]))

        dist = vdf.data.data[t_id, ...]
        dist = dist[None, ...]
        step_table = step_table[t_id]
        azimuthal = azimuthal[t_id, ...]

        if step_table.data:
            energy_edges = energy1_edges
        else:
            energy_edges = energy1_edges

    elif tint is not None and len(tint) == 2:
        dist = time_clip(vdf.data, tint)
        step_table = ts_scalar(vdf.time.data, step_table)
        step_table = time_clip(step_table, tint)
        azimuthal = time_clip(azimuthal, tint)

        if len(dist.time) > 1 and list(energy0) != list(energy1):
            print("notice: Rebinning distribution.")
            temp = ts_skymap(dist.time.data, dist, time_clip(vdf.energy, tint),
                             np.rad2deg(azimuthal), theta)
            newt, dist, energy, phi = psd_rebin(temp, phi, energy0, energy1,
                                                step_table)
            dist = ts_skymap(newt, dist, np.tile(energy, (len(newt), 1)), phi,
                             theta)
            dist = time_clip(dist.data, tint).data
            azimuthal = xr.DataArray(phi,
                                     coords=[newt, np.arange(phi.shape[1])],
                                     dims=["time", "odx"])
            len_e = dist.shape[1]
            energy_edges = np.hstack(
                [10 ** (np.log10(energy) - diff_energ / 2),
                 10 ** (np.log10(energy[-1]) + diff_energ / 2)])
        else:
            if all(step_table.data):
                energy_edges = energy1_edges
            else:
                energy_edges = energy0_edges
    else:
        raise ValueError("Invalid time interval")

    return dist, polar.data, azimuthal.data, energy_edges, len_e


def _cotrans(dist, polar, azimuthal, x_vec, y_vec, z_vec, e_lim,
             bin_corr):
    # Construct polar and azimuthal angle matrices
    polar = np.ones((len(dist), 1)) * polar

    f_mat = np.zeros((len(dist), dist.shape[2], dist.shape[1]))  #
    # azimuthal, energy
    edges_az = np.linspace(0, 2 * np.pi, azimuthal.shape[1] + 1)

    for i in range(len(dist)):
        pol_mat, azm_mat = np.meshgrid(polar[i, :], azimuthal[i, :])

        # '-' because the data shows which direction the particles were
        # coming from
        x_mat = -np.sin(pol_mat) * np.cos(azm_mat)
        y_mat = -np.sin(pol_mat) * np.sin(azm_mat)
        z_mat = -np.cos(pol_mat)

        # Transform into different coordinate system
        xx_mat = np.reshape(x_mat, (x_mat.shape[0] * x_mat.shape[1], 1))
        yy_mat = np.reshape(y_mat, (y_mat.shape[0] * y_mat.shape[1], 1))
        zz_mat = np.reshape(z_mat, (z_mat.shape[0] * z_mat.shape[1], 1))

        new_tmp_x = np.dot(np.hstack([xx_mat, yy_mat, zz_mat]), x_vec)
        new_tmp_y = np.dot(np.hstack([xx_mat, yy_mat, zz_mat]), y_vec)
        new_tmp_z = np.dot(np.hstack([xx_mat, yy_mat, zz_mat]), z_vec)

        new_x_mat = np.reshape(new_tmp_x, (x_mat.shape[0], x_mat.shape[1]))
        new_y_mat = np.reshape(new_tmp_y, (x_mat.shape[0], x_mat.shape[1]))
        new_z_mat = np.reshape(new_tmp_z, (x_mat.shape[0], x_mat.shape[1]))

        elevation_angle = np.arctan(
            new_z_mat / np.sqrt(new_x_mat ** 2 + new_y_mat ** 2))
        plane_az = np.arctan2(new_y_mat, new_x_mat) + np.pi

        # gets velocity in direction normal to 'z'-axis
        geo_factor_elev = np.cos(elevation_angle)

        # geoFactorBinSize - detector bins in 'equator' plane are bigger and
        # get a larger weight. I think this is not good for the
        # implementation in this function
        if bin_corr:
            geo_factor_bin_size = np.sin(pol_mat)
        else:
            geo_factor_bin_size = np.ones(pol_mat.shape)

        f_mat[i, ...] = _cotrans_jit(dist[i, ...], elevation_angle, e_lim,
                                     plane_az, edges_az, geo_factor_elev,
                                     geo_factor_bin_size)

    return f_mat


def _cotrans_jit(dist, elevation_angle, elevation_lim, plane_az, edges_az,
                 geo_factor_elev, geo_factor_bin_size):
    out = np.zeros((dist.shape[1], dist.shape[0]))  # azimuthal, energy

    for ie, iaz in itertools.product(range(dist.shape[0]),
                                     range(dist.shape[1])):
        # dist.data has dimensions nT x nE x nAz x nPol
        c_mat = dist[ie, ...].copy()
        c_mat = c_mat * geo_factor_elev * geo_factor_bin_size
        c_mat[np.abs(elevation_angle) > np.deg2rad(elevation_lim)] = np.nan
        # use 0.1 deg to fix Az angle edges bug
        c_mat[plane_az < edges_az[iaz] - np.deg2rad(.1)] = np.nan
        # use 0.1 deg to fix Az angle edges bug
        c_mat[plane_az > edges_az[iaz + 1] + np.deg2rad(.1)] = np.nan

        out[iaz, ie] = np.nanmean(c_mat)

    return out


def vdf_projection(vdf, tint, coord_sys: np.ndarray = np.eye(3),
                   sc_pot: xr.DataArray = None, e_lim: float = 20,
                   bins_correction: bool = False):
    r"""Computes projection of the velocity distribution onto a specified
    plane.

    Parameters
    ----------
    vdf : xarray.Dataset
        Electron or ion 3D skymap velocity distribution function.
    tint : list of str
        Computes data for time interval if len(tint) = 2 or closest time if
        len(tint) = 1. For tint includes two or more distributions the
        energies are rebinned into 64 channels.
    coord_sys : ndarray, Optional
        3x3 matrix with 1st column is x, 2nd column is y and 3rd column is z.
        z is normal to the projection plane and x and y are made orthogonal to
        z and each other if they are not already. Default is np.eye(3)
        (project onto spacecraft spin plane).
    sc_pot : xarray.DataArray, Optional
        Spacecraft potential to correct velocities. For a single value of tint
        the closest potential is used. For an interval the spacecraft
        potential is average over that interval. Default is None (no
        correction).
    e_lim : float, Optional
        Elevation angle limit in degrees above/below projection plane to
        include in projection. Default is e_lim = 20.
    bins_correction : bool, Optional
        Flag to correction elevation bins. Default is False.

    Returns
    -------
    v_x : ndarray
        2D grid of the velocities in the x direction.
    v_y : ndarray
        2D grid of the velocities in the y direction.
    f_mat : ndarray
        2D projection of the velocity distribution onto the specified plane

    """

    specie = vdf.attrs.get("species", "electrons")
    is_des = specie.lower() == "electrons"

    dist, polar, azimuthal, energy_edges, len_e = _init(vdf, tint)
    x_vec, y_vec, z_vec, changed_xyz = _coord_sys(coord_sys)

    if azimuthal.ndim == 1:
        azimuthal = np.ones((len(dist), 1)) * azimuthal

    f_mat = _cotrans(dist, polar, azimuthal, x_vec, y_vec, z_vec, e_lim,
                     bins_correction)
    if len(dist) == 1:
        f_mat = np.squeeze(f_mat)
    else:
        f_mat = np.squeeze(np.nanmean(f_mat, axis=0))

    if sc_pot is not None:
        if len(tint) == 1:
            time_datetime64 = iso86012datetime64(np.array(tint))[0]
            t_id = np.argmin(np.abs(sc_pot.time.data - time_datetime64))
            sc_pot = sc_pot.data[t_id]
        else:
            sc_pot = time_clip(sc_pot, tint)
            sc_pot = np.nanmean(sc_pot.data)
    else:
        sc_pot = 0.

    if is_des:
        mass = constants.electron_mass
    else:
        mass = constants.proton_mass
        sc_pot *= -1

    q_e = constants.elementary_charge

    speed_table = np.sqrt((energy_edges - sc_pot) * q_e * 2 / mass)
    speed_table = np.real(speed_table * 1e-3)  # km/s

    r_en = speed_table
    v_x = np.matmul(r_en[:, None],
                    np.cos(np.linspace(0, 2 * np.pi, azimuthal.shape[1] + 1)
                           + np.pi)[None, :])
    v_y = np.matmul(r_en[:, None],
                    np.sin(np.linspace(0, 2 * np.pi, azimuthal.shape[1] + 1)
                           + np.pi)[None, :])

    f_mat[f_mat <= 0] = np.nan

    return v_x, v_y, f_mat
