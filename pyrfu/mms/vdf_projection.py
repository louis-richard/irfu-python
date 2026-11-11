#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import warnings

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.mms._psd_units import _is_relativistic, _output, _relativistic_factor
from pyrfu.mms._vdf_bins import (
    _charge,
    _charge_sign,
    _energy_edges,
    _mass,
    _speed,
)
from pyrfu.mms.psd_rebin import psd_rebin
from pyrfu.mms.vdf_convert import _PSD
from pyrfu.pyrf.iso86012datetime64 import iso86012datetime64
from pyrfu.pyrf.time_clip import time_clip
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def _coord_sys(coord_sys: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, list):
    x_vec = coord_sys[0, :] / np.linalg.norm(coord_sys[0, :])
    y_vec = coord_sys[1, :] / np.linalg.norm(coord_sys[1, :])

    z_vec = np.cross(x_vec, y_vec) / np.linalg.norm(np.cross(x_vec, y_vec))
    y_vec = np.cross(z_vec, x_vec) / np.linalg.norm(np.cross(z_vec, x_vec))

    changed_xyz = [False, False, False]

    for i, vec, comp in zip([0, 1, 2], [x_vec, y_vec, z_vec], ["x", "y", "z"]):
        # x, y and z are the rows of coord_sys
        old_vec = coord_sys[i, :] / np.linalg.norm(coord_sys[i, :])
        cos_angle = np.clip(np.dot(vec, old_vec), -1.0, 1.0)
        if np.rad2deg(np.arccos(cos_angle)) > 1.0:
            logger.warning(
                "In making xyz a right handed orthogonal coordinate system, %(comp)s "
                "(in-plane %(i)d) was changed from %(x_old)s to %(x_new)s. Please "
                "verify that this is according to your intentions.",
                {
                    "comp": comp,
                    "i": i,
                    "x_old": np.array2string(coord_sys[i, :]),
                    "x_new": np.array2string(vec),
                },
            )
            changed_xyz[i] = True

    return x_vec, y_vec, z_vec, changed_xyz


def _width_edges(vdf: Dataset, i_t: int):
    r"""Energy edges (eV) from the channel widths (delta_energy_minus/plus) at
    the time index i_t: the lower edge of the first channel, the geometric
    means of the upper edges and the lower edges of the next channels, and
    the upper edge of the last channel. None without the widths."""
    if not {"delta_energy_minus", "delta_energy_plus"} <= set(vdf.attrs):
        return None

    lower, upper = (edges[i_t] for edges in _energy_edges(vdf))

    return np.hstack([lower[0], np.sqrt(upper[:-1] * lower[1:]), upper[-1]])


def _angle_widths(vdf: Dataset, i_times: np.ndarray, n_t: int):
    r"""Angular widths (radians) of the bins from the delta_phi_* (time, phi)
    and delta_theta_* (theta) attributes at the times i_times (those of the
    first one if the distribution was rebinned to n_t other times), or None
    without them."""
    keys = ["delta_phi_minus", "delta_phi_plus"]
    keys += ["delta_theta_minus", "delta_theta_plus"]

    if not all(k in vdf.attrs for k in keys):
        return None

    n_phi = vdf.phi.shape[-1]
    phi_widths = []

    for key in keys[:2]:
        width = np.broadcast_to(vdf.attrs[key], (len(vdf.time), n_phi))[i_times]
        width = width if len(width) == n_t else np.tile(width[:1], (n_t, 1))
        phi_widths.append(np.deg2rad(width))

    theta_widths = [np.deg2rad(np.asarray(vdf.attrs[k], float)) for k in keys[2:]]

    return (*phi_widths, *theta_widths)


def _sub_angles(centres, minus, plus, n_sub):
    r"""n_sub angles evenly spaced across each bin [centre - minus, centre +
    plus], bin after bin."""
    frac = (np.arange(n_sub) + 0.5) / n_sub
    sub = (centres - minus)[:, None] + (minus + plus)[:, None] * frac[None, :]
    return sub.ravel()


def _log_edges(energy, half_width):
    r"""Energy edges `half_width` (in log10(E)) below each channel and above
    the last one."""
    log_energy = np.log10(energy)
    return np.hstack(
        [10 ** (log_energy - half_width), 10 ** (log_energy[-1] + half_width)]
    )


def _table_edges(step_table, energy0_edges, energy1_edges):
    r"""Edges of the energy table of the distributions: table 1 if they all
    use it (step table), else table 0."""
    return energy1_edges if np.all(step_table) else energy0_edges


def _channel_edges(vdf: Dataset, i_t: int, energy_edges):
    r"""Edges from the channel widths at the time index i_t if the skymap has
    them, as many as `energy_edges` (e.g. EIS, FEEPS), else `energy_edges`."""
    width_edges = _width_edges(vdf, i_t)

    if width_edges is None or len(width_edges) != len(energy_edges):
        return energy_edges

    return width_edges


def _rebin_tables(
    vdf: Dataset, dist, azimuthal, tint, theta, energy0, energy1, step_table
):
    r"""Distributions of `tint` with alternating energy tables rebinned to one
    table of 64 channels (psd_rebin): data, azimuthal angles (radians) of the
    look directions and energies (eV)."""
    logger.info("Rebinning distribution.")
    temp = ts_skymap(
        dist.time.data,
        dist.data,
        time_clip(vdf.energy, tint).data,
        np.rad2deg(azimuthal),
        theta,
    )
    # phi clipped to tint, like the distribution
    newt, dist, energy, phi = psd_rebin(
        temp,
        temp.phi.data,
        energy0,
        energy1,
        step_table,
    )
    dist = ts_skymap(
        newt,
        dist,
        np.tile(energy, (len(newt), 1)),
        phi,
        theta,
    )
    dist = time_clip(dist.data, tint)
    # psd_rebin returns phi in degrees; azimuthal is in radians
    azimuthal = xr.DataArray(
        np.deg2rad(phi),
        coords=[newt, np.arange(phi.shape[1])],
        dims=["time", "idx"],
    )

    return dist, time_clip(azimuthal, tint).data, energy


def _init(vdf: Dataset, tint: list):
    len_e, rebinned = 32, False

    if vdf.phi.data.ndim == 1:
        phi = xr.DataArray(
            np.tile(vdf.phi.data, (len(vdf.data), 1)),
            coords=[vdf.time.data, np.arange(len(vdf.phi.data))],
            dims=["time", "idx"],
        )
    else:
        phi = vdf.phi

    theta = vdf.theta.data
    polar = np.deg2rad(theta)
    azimuthal = np.deg2rad(phi)
    step_table = vdf.attrs.get("esteptable", np.zeros(len(vdf.time)))

    # The energy tables, else the energies of the first two times (the default
    # of attrs.get was evaluated even with the attribute: a single time raised
    # IndexError)
    energy = np.atleast_2d(vdf.energy.data)
    energy0 = vdf.attrs["energy0"] if "energy0" in vdf.attrs else energy[0]
    energy1 = (
        vdf.attrs["energy1"] if "energy1" in vdf.attrs else energy[1 % len(energy)]
    )

    diff_energ = np.median(np.diff(np.log10(energy0))) / 2

    energy0_edges = _log_edges(energy0, diff_energ)
    energy1_edges = _log_edges(energy1, diff_energ)

    if tint is not None and len(tint) == 1:
        t_id = np.argmin(
            np.abs(vdf.time.data - iso86012datetime64(np.array(tint))[0]),
        )

        dist = vdf.data.data[t_id, ...]
        dist = dist[None, ...]
        step_table = step_table[t_id]
        azimuthal = azimuthal.data[t_id, ...]
        energy_edges = _table_edges(step_table, energy0_edges, energy1_edges)
        i_times = np.array([t_id])

    elif tint is not None and len(tint) == 2:
        dist = time_clip(vdf.data, tint)
        i_times = np.flatnonzero(np.isin(vdf.time.data, dist.time.data))
        step_table = time_clip(ts_scalar(vdf.time.data, step_table), tint)
        step_table = step_table.data
        azimuthal = time_clip(azimuthal, tint)
        azimuthal = azimuthal.data

        if len(dist.time) > 1 and list(energy0) != list(energy1):
            dist, azimuthal, energy = _rebin_tables(
                vdf, dist, azimuthal, tint, theta, energy0, energy1, step_table
            )
            len_e, rebinned = dist.shape[1], True
            energy_edges = _log_edges(energy, diff_energ / 2)
        else:
            energy_edges = _table_edges(step_table, energy0_edges, energy1_edges)

        dist = dist.data
    else:
        raise ValueError("Invalid time interval")

    # Edges from the channel widths if any (e.g. EIS, FEEPS); not for the
    # rebinned alternating tables
    if not rebinned:
        energy_edges = _channel_edges(vdf, i_times[0], energy_edges)

    return dist, polar, azimuthal, energy_edges, len_e, i_times


def _cotrans(dist, polar, azimuthal, x_vec, y_vec, z_vec, e_lim, bin_corr, widths=None):
    # Construct polar and azimuthal angle matrices
    polar = np.ones((len(dist), 1)) * polar

    f_mat = np.zeros((len(dist), dist.shape[2], dist.shape[1]))  #
    # azimuthal, energy
    edges_az = np.linspace(0, 2 * np.pi, azimuthal.shape[1] + 1)

    # With angular widths, each bin is sampled at n_sub x n_sub directions
    # across its extent, so that a bin crossing the elevation limit or two
    # azimuthal sectors counts in proportion (its centre only without)
    n_sub = 5

    for i in range(len(dist)):
        pol_i, azm_i, dist_i = polar[i, :], azimuthal[i, :], dist[i, ...]

        if widths is not None:
            phi_m, phi_p, theta_m, theta_p = widths
            pol_i = _sub_angles(pol_i, theta_m, theta_p, n_sub)
            azm_i = _sub_angles(azm_i, phi_m[i], phi_p[i], n_sub)
            dist_i = np.repeat(np.repeat(dist_i, n_sub, axis=1), n_sub, axis=2)

        pol_mat, azm_mat = np.meshgrid(pol_i, azm_i)

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
            new_z_mat / np.sqrt(new_x_mat**2 + new_y_mat**2),
        )
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

        f_mat[i, ...] = _cotrans_jit(
            dist_i,
            elevation_angle,
            e_lim,
            plane_az,
            edges_az,
            geo_factor_elev,
            geo_factor_bin_size,
        )

    return f_mat


def _cotrans_jit(
    dist,
    elevation_angle,
    elevation_lim,
    plane_az,
    edges_az,
    geo_factor_elev,
    geo_factor_bin_size,
):
    # dist has dimensions nE x nAz x nPol; output is azimuthal bin x energy
    weighted = dist * (geo_factor_elev * geo_factor_bin_size)[None, ...]
    in_elevation = np.abs(elevation_angle) <= np.deg2rad(elevation_lim)

    out = np.zeros((len(edges_az) - 1, dist.shape[0]))

    for i_az in range(len(edges_az) - 1):
        # use 0.1 deg to fix Az angle edges bug
        in_bin = in_elevation & (plane_az >= edges_az[i_az] - np.deg2rad(0.1))
        in_bin &= plane_az <= edges_az[i_az + 1] + np.deg2rad(0.1)

        with warnings.catch_warnings():
            # empty bins give NaN, as before
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[i_az, :] = np.nanmean(
                np.where(in_bin[None, ...], weighted, np.nan), axis=(1, 2)
            )

    return out


def vdf_projection(
    vdf,
    tint,
    coord_sys: np.ndarray = np.eye(3),
    sc_pot: xr.DataArray = None,
    e_lim: float = 20,
    bins_correction: bool = False,
):
    r"""Compute the projection of the velocity distribution onto a specified plane.

    Parameters
    ----------
    vdf : xarray.Dataset
        Electron or ion 3D skymap velocity distribution function. The
        ``species`` attribute gives the mass (e.g. "ions", "electrons",
        "oxygen+", "helium++"; electrons without it) and the optional
        ``charge`` attribute the charge of the ions (1 by default).
    tint : list of str
        Computes data for time interval if len(tint) = 2 or closest time if
        len(tint) = 1. For tint includes two or more distributions the
        energies are rebinned into 64 channels.
    coord_sys : ndarray, Optional
        3x3 matrix with x, y and z as rows, e.g., ``np.vstack([x, y, z])``.
        z is normal to the projection plane, and x and y are made orthogonal to
        z and each other if they are not already. Default is np.eye(3)
        (project onto spacecraft spin plane).
    sc_pot : xarray.DataArray, Optional
        Spacecraft potential to correct velocities (the ions gain q V and the
        electrons lose e V). For a single value of tint the closest potential
        is used. For an interval the spacecraft potential is average over
        that interval. Default is None (no correction).
    e_lim : float, Optional
        Elevation angle limit in degrees above/below projection plane to
        include in projection. Default is e_lim = 20.
    bins_correction : bool, Optional
        Flag to correction elevation bins. Default is False.

    Returns
    -------
    v_x : numpy.ndarray
        2D grid of the velocities in the x direction.
    v_y : numpy.ndarray
        2D grid of the velocities in the y direction.
    f_mat : numpy.ndarray
        2D projection of the velocity distribution onto the specified plane,
        in the units of `vdf`: of the relativistic phase-space density, at the
        relativistic speeds. For the FPI, HPCA and EIS skymaps, whose
        phase-space density is the non-relativistic one, it is that of the
        files times 2 gamma^5 / (gamma + 1) (+5 % for electrons at 5 keV,
        +27 % at 27 keV); the speeds were non-relativistic (1.47 c for FEEPS
        electrons of 500 keV).

    Notes
    -----
    As irfu-matlab mms.plot_projection, each in-plane azimuthal sector and
    energy is the mean of f cos(elevation) over the bins within `e_lim` of
    the plane. The energy edges are those of the channel widths
    (``delta_energy_minus`` and ``delta_energy_plus``) when `vdf` has them
    (e.g. EIS, FEEPS, FPI burst), else halfway between the channels in
    log(E), as irfu-matlab (which uses them for all the instruments; within
    1 % of the FPI burst widths, apart from the first ion channel). With the
    angular widths of the bins (``delta_phi_*`` and ``delta_theta_*``, e.g.
    FEEPS, HPCA), each bin is sampled at 5 x 5 directions across its extent,
    so that a bin crossing the elevation limit or two sectors counts in
    proportion; without them (e.g. FPI), the bins are placed by their
    centres.

    """
    # Mass and charge of the species (electrons without species attribute).
    # All the species not starting with "e" were protons.
    if "species" not in vdf.attrs:
        vdf = vdf.assign_attrs(species="electrons")

    # (a DataArray raised AttributeError before the check, in _init)
    if not isinstance(vdf, Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    mass, is_des = _mass(vdf), _charge_sign(vdf) < 0.0
    charge = 1.0 if is_des else _charge(vdf)

    # The relativistic phase-space density, projected with the relativistic
    # speeds: that of the FPI, HPCA and EIS files times 2 gamma^5 / (gamma + 1)
    data, units = vdf.data.data, str(vdf.data.attrs.get("UNITS", ""))

    if not _is_relativistic(vdf) and units.lower() in _PSD:
        vdf = _output(vdf, data * _relativistic_factor(vdf, data), units)

    dist, polar, azimuthal, energy_edges, _, i_times = _init(vdf, tint)
    x_vec, y_vec, z_vec, _ = _coord_sys(coord_sys)

    if azimuthal.ndim == 1:
        azimuthal = np.ones((len(dist), 1)) * azimuthal

    widths = _angle_widths(vdf, i_times, len(dist))

    f_mat = _cotrans(
        dist,
        polar,
        azimuthal,
        x_vec,
        y_vec,
        z_vec,
        e_lim,
        bins_correction,
        widths,
    )
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
        sc_pot = 0.0

    if not is_des:
        # Ions of charge q gain q V (eV)
        sc_pot *= -charge

    # Speeds of the energy edges after correction for the spacecraft potential
    # (0 below it, rather than NaN which breaks pcolormesh), in km/s
    energy_corr = np.clip(energy_edges - sc_pot, 0.0, None)
    speed_table = _speed(energy_corr, mass) * 1e-3

    r_en = speed_table
    v_x = np.matmul(
        r_en[:, None],
        np.cos(np.linspace(0, 2 * np.pi, azimuthal.shape[1] + 1) + np.pi)[None, :],
    )
    v_y = np.matmul(
        r_en[:, None],
        np.sin(np.linspace(0, 2 * np.pi, azimuthal.shape[1] + 1) + np.pi)[None, :],
    )

    f_mat[f_mat <= 0] = np.nan

    return v_x, v_y, f_mat
