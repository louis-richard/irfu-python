#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import interpolate, constants

# Local imports
from pyrfu.pyrf import cart2sph, sph2cart, resample, time_clip

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.12"
__status__ = "Prototype"

__all__ = ["vdf_frame_transformation", "vdf_reduce"]


def _interp_skymap_sphe(vdf, energy, phi, theta, grid_sphe):
    r"""Interpolate the skymap distribution defined on the grid (`energy`,
    `phi`, `theta`) onto the spherical grid `grid_sphe`.

    Parameters
    ----------
    vdf : numpy.ndarray
        Values of the skymap distribution.
    energy : numpy.ndarray
        Energy level of skymap sampling.
    phi : numpy.ndarray
        Azimuthal angle of skymap sampling.
    theta : numpy.ndarray
        Elevation angle of skymap sampling.
    grid_sphe : numpy.ndarray
        Spherical velocity grid to interpolate on (3xlxmxn).

    Returns
    -------
    out_data : numpy.ndarray
        Values of the distribution interpolated onto `grdi_sphe`.

    Notes
    -----
    The values corresponding to energy levels below the instrument range are
    discarded.

    """

    phi_period = np.zeros(len(phi) + 2)
    phi_period[1:-1] = phi
    phi_period[0] = phi[-1] - 2 * 180.
    phi_period[-1] = phi[0] + 2 * 180.

    theta_period = np.zeros(len(theta) + 2)
    theta_period[1:-1] = theta
    theta_period[0] = theta[-1] - 180.
    theta_period[-1] = theta[0] + 180.

    vdf_period = np.zeros((len(energy), len(phi) + 2, len(theta) + 2))
    vdf_period[:, 1:-1, 1:-1] = vdf
    vdf_period[:, 1:-1, 0] = vdf[:, :, -1]
    vdf_period[:, 1:-1, -1] = vdf[:, :, 0]
    vdf_period[:, 0] = vdf_period[:, 1]
    vdf_period[:, -1] = vdf_period[:, -2]

    vdf_interp = interpolate.RegularGridInterpolator((energy, phi_period,
                                                      theta_period),
                                                     vdf_period,
                                                     method="linear",
                                                     bounds_error=False,
                                                     fill_value=None)

    out_data = vdf_interp(grid_sphe)

    return out_data


def _interp_skymap_cart(vdf, energy, phi, theta, grid_cart):
    r"""Interpolate the skymap distribution defined on the grid (`energy`,
    `phi`, `theta`) onto the cartesian grid `grid_cart`.

    Parameters
    ----------
    vdf : numpy.ndarray
        Values of the skymap distribution.
    energy : numpy.ndarray
        Energy level of skymap sampling.
    phi : numpy.ndarray
        Azimuthal angle of skymap sampling.
    theta : numpy.ndarray
        Elevation angle of skymap sampling.
    grid_cart : numpy.ndarray
        Cartesian velocity grid to interpolate on (3xlxmxn).

    Returns
    -------
    out_data : numpy.ndarray
        Values of the distribution interpolated onto `grid_cart`.

    Notes
    -----
    The values corresponding to energy levels below the instrument range are
    discarded.

    See Also
    --------
    _inter_skymap_sphe.py
    """

    # Unpack cartesian grid
    v_x, v_y, v_z = grid_cart

    # Transform cartesian velocity grid to spherical velocity grid
    az, el, r = cart2sph(v_x, v_y, v_z)
    en = .5 * constants.proton_mass * r ** 2 / constants.elementary_charge
    az = np.rad2deg(az) + 180.
    el = np.rad2deg(el)

    grid_sphe = np.transpose(np.stack([en, az, el]), [1, 2, 3, 0])

    # Interpolate the skymap distribution onto the spherical grid
    out_data = _interp_skymap_sphe(vdf, energy, phi, theta, grid_sphe)

    # Discard points with energy below the instrument energy range.
    v_min_2 = 2 * energy[0] * constants.electron_volt / constants.proton_mass
    out_data[v_x ** 2 + v_y ** 2 + v_z ** 2 < v_min_2] = np.nan

    return out_data


def vdf_frame_transformation(vdf, v_gse):
    r"""Move the skymap into the desired frame associated with the bulk
    velocity `v_gse`.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution in the initial frame.
    v_gse : xarray.DataArray
        Time series of the bulk velocity to shift.

    Returns
    -------
    out : xarray.Dataset
        Skymap distribution into the new frame.

    Notes
    -----
    The new skymap grid is identical to the original one. The bulk velocity
    must be in the same coordinates system as the skymap (i.e spacecraft for
    FPI and GSE for EIS)

    See Also
    --------
    _interp_skymap_cart.py, _interp_skymap_sphe.py

    """

    v_bulk = resample(v_gse, vdf.time)
    theta = vdf.theta.data

    out_data = np.zeros_like(vdf.data.data)

    for i in range(len(vdf.time.data)):
        vdf_data = vdf.data.data[i, :]
        energy = vdf.energy.data[i, :]
        phi = vdf.phi.data[i, :]

        phi_mat, en_mat, theta_mat = np.meshgrid(phi, energy, theta)
        v_mat = np.sqrt(2 * en_mat * constants.electron_volt
                        / constants.proton_mass)
        v_x, v_y, v_z = sph2cart(np.deg2rad(phi_mat),
                                 np.deg2rad(theta_mat), v_mat)

        grid_cart = np.stack([v_x - v_bulk.data[i, 0, None, None, None],
                              v_y - v_bulk.data[i, 1, None, None, None],
                              v_z - v_bulk.data[i, 2, None, None, None]])

        out_data[i, ...] = _interp_skymap_cart(vdf_data, energy, phi, theta,
                                               grid_cart)

    out = vdf.copy()
    out.data.data = out_data

    return out


def vdf_reduce(vdf, tint, dim, x_vec, z_vec: list = None, v_int: list = None,
               n_vpt: int = 100):
    r"""Interpolate the skymap distribution onto the velocity grid defined
    by the velocity interval `v_int` along the axes `x_vec` and `z_vec`,
    and reduce (integrate) it along 1 (if `dim` is "2d") or 2 (if `dim` is
    "1d").

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution to reduce.
    tint : list of strs
        Time interval over which the time series of the skymap distribution
        is averaged.
    dim : {"1d", "2d"}
        Dimension of the output reduced distribution.
    x_vec : array_like
        X axis. For the "1d" case, it is the axis on which the skymap is
        plotted. For the "2d" case, it is the first of the two axes on which
        the skymap is plotted.
    z_vec : array_like, Optional
        Axis along which the skymap is integrated. Needed only for the "2d"
        case.
    v_int : array_like, Optional
        Velocity interval.
    n_vpt : int, Optional
        Number of points along the plot direction(s).

    Returns
    -------
    out : xarray.DataArray
        Reduced distribution.

    """

    if v_int is None:
        v_int = [-1e6, 1e6]
    if z_vec is None:
        z_vec = [0, 0, 1]

    x_vec = x_vec / np.linalg.norm(x_vec, keepdims=True)
    y_vec = np.cross(z_vec, x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec, keepdims=True)
    z_vec = np.cross(x_vec, y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec, keepdims=True)

    v_x, v_y, v_z = [np.linspace(v_int[0], v_int[1], n_vpt) for _ in range(3)]

    m_vec = np.transpose([x_vec, y_vec, z_vec])
    v_x, v_y, v_z = np.matmul(np.linalg.inv(m_vec), np.array([v_x, v_y, v_z]))
    v_x_mat, v_y_mat, v_z_mat = np.meshgrid(v_x, v_y, v_z)
    grid_cart = np.stack([v_x_mat, v_y_mat, v_z_mat])

    vdf = time_clip(vdf, tint)

    vdf_data = np.mean(vdf.data.data, axis=0)
    energy = np.mean(np.atleast_2d(vdf.energy.data), axis=0)
    phi = np.mean(np.atleast_2d(vdf.phi.data), axis=0)
    theta = vdf.theta.data

    interp_vdf = _interp_skymap_cart(vdf_data, energy, phi, theta, grid_cart)

    if dim.lower() == "2d":
        dv_ = np.abs(np.diff(v_z)[0])
        red_vdf = np.nansum(interp_vdf, axis=-1) * dv_
        out = xr.DataArray(red_vdf, coords=[v_x / 1e6, v_y / 1e6],
                           dims=["vx", "vy"])
    elif dim.lower() == "1d":
        dv_ = np.abs(np.diff(v_y)[0] * np.diff(v_z)[0])
        red_vdf = np.nansum(np.sum(interp_vdf, axis=-1), axis=-1) * dv_
        out = xr.DataArray(red_vdf, coords=[v_x / 1e6], dims=["vx"])

    else:
        raise ValueError

    return out
