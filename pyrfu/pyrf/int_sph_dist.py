#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random

from math import cos, sin, asin, sqrt

# Third party imports
import numba
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.3.26"
__status__ = "Prototype"


def int_sph_dist(vdf, speed, phi, theta, speed_grid, **kwargs):
    r"""Integrate a spherical distribution function to a line/plane.

    Parameters
    ----------
    vdf : numpy.ndarray
        Phase-space density skymap.
    speed : numpy.ndarray
        Velocity of the instrument bins,
    phi : numpy.ndarray
        Azimuthal angle of the instrument bins.
    theta : numpy.ndarray
        Elevation angle of the instrument bins.
    speed_grid : numpy.ndarray
        Velocity grid for interpolation.
    **kwargs
        Keyw

    Returns
    -------

    """

    # Coordinates system transformation matrix
    xyz = kwargs.get("xyz", np.eye(3))

    # Number of Monte Carlo iterations and how number of MC points is
    # weighted to data.
    n_mc = kwargs.get("n_mc", 10)
    weight = kwargs.get("weight", None)

    # limit on out-of-plane velocity and azimuthal angle
    v_lim = np.array(kwargs.get("v_lim", [-np.inf, np.inf]), dtype=np.float64)
    a_lim = np.array(kwargs.get("a_lim", [-180.0, 180.0]), dtype=np.float64)
    a_lim = np.deg2rad(a_lim)

    # Projection dimension and base
    projection_base = kwargs.get("projection_base", "pol")
    projection_dim = kwargs.get("projection_dim", "1d")

    speed_edges = kwargs.get("speed_edges", None)
    speed_grid_edges = kwargs.get("speed_grid_edges", None)

    # Azimuthal and elevation angles steps. Assumed to be constant
    # if not provided.
    d_phi = np.abs(np.median(np.diff(phi))) * np.ones_like(phi)
    d_phi = kwargs.get("d_phi", d_phi)
    d_theta = np.abs(np.median(np.diff(theta))) * np.ones_like(theta)
    d_theta = kwargs.get("d_theta", d_theta)

    # azimuthal angle of projection plane
    n_az_g = len(phi)
    d_phi_g = 2 * np.pi / n_az_g
    phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, n_az_g) + d_phi_g / 2
    phi_grid = kwargs.get("phi_grid", phi_grid)

    # Overwrite projection dimension if azimuthal angle of projection
    # plane is not provided. Set the azimuthal angle grid width.
    if phi_grid is None or projection_dim == "1d":
        projection_dim = "1d"
        d_phi_grid = 1.0
    elif phi_grid is not None and projection_dim.lower() in ["2d", "3d"]:
        d_phi_grid = np.median(np.diff(phi_grid))
    else:
        raise RuntimeError(
            "1d projection with phi_grid provided doesn't make sense!!",
        )

    # Make sure the transformation matrix is orthonormal.
    x_phat = xyz[:, 0] / np.linalg.norm(xyz[:, 0])  # re-normalize
    y_phat = xyz[:, 1] / np.linalg.norm(xyz[:, 1])  # re-normalize

    z_phat = np.cross(x_phat, y_phat)
    z_phat /= np.linalg.norm(z_phat)
    y_phat = np.cross(z_phat, x_phat)

    r_mat = np.transpose(np.stack([x_phat, y_phat, z_phat]), [1, 0])

    if speed_edges is None:
        d_v = np.hstack([np.diff(speed[:2]), np.diff(speed)])
        d_v_m, d_v_p = [np.diff(speed) / 2.0] * 2
    else:
        d_v_m = speed - speed_edges[:-1]
        d_v_p = speed_edges[1:] - speed
        d_v = d_v_m + d_v_p

    # Speed grid bins edges
    if speed_grid_edges is None:
        speed_grid_edges = np.zeros(len(speed_grid) + 1)
        speed_grid_edges[0] = speed_grid[0] - np.diff(speed_grid[:2]) / 2.0
        speed_grid_edges[1:-1] = speed_grid[:-1] + np.diff(speed_grid) / 2.0
        speed_grid_edges[-1] = speed_grid[-1] + np.diff(speed_grid[-2:]) / 2.0
    else:
        speed_grid = speed_grid_edges[:-1] + np.diff(speed_grid_edges) / 2.0

    if projection_base == "pol":
        d_v_grid = np.diff(speed_grid_edges)
    else:
        mean_diff = np.mean(np.diff(speed_grid))
        msg = "For a cartesian grid, all velocity bins must be equal!!"
        assert (np.diff(speed_grid) / mean_diff - 1 < 1e-2).all(), msg

        d_v_grid = mean_diff

    # Weighting of number of Monte Carlo particles
    n_sum = n_mc * np.sum(vdf != 0)  # total number of Monte Carlo particles
    if weight == "lin":
        n_mc_mat = np.ceil(n_sum / np.sum(vdf) * vdf)
    elif weight == "log":
        n_mc_mat = np.ceil(
            n_sum / np.sum(np.log10(vdf + 1)) * np.log10(vdf + 1),
        )
    else:
        n_mc_mat = np.zeros_like(vdf)
        n_mc_mat[vdf != 0] = n_mc

    n_mc_mat = n_mc_mat.astype(int)

    if projection_base == "pol":
        # Area or line element (primed)
        d_a_grid = speed_grid ** (int(projection_dim[0]) - 1) * d_phi_grid * d_v_grid
        d_a_grid = d_a_grid.astype(np.float64)

        if projection_dim == "1d":
            f_g = mc_pol_1d(
                vdf,
                speed,
                phi,
                theta,
                d_v,
                d_v_m,
                d_phi,
                d_theta,
                speed_grid_edges,
                d_a_grid,
                v_lim,
                a_lim,
                n_mc_mat,
                r_mat,
            )
        else:
            raise NotImplementedError(
                "2d projection on polar grid is not ready yet!!",
            )

    elif projection_base == "cart" and projection_dim == "2d":
        d_a_grid = d_v_grid**2
        f_g = mc_cart_2d(
            vdf,
            speed,
            phi,
            theta,
            d_v,
            d_v_m,
            d_phi,
            d_theta,
            speed_grid_edges,
            d_a_grid,
            v_lim,
            a_lim,
            n_mc_mat,
            r_mat,
        )
    elif projection_base == "cart" and projection_dim == "3d":
        d_a_grid = d_v_grid**3
        f_g = mc_cart_3d(
            vdf,
            speed,
            phi,
            theta,
            d_v,
            d_v_m,
            d_phi,
            d_theta,
            speed_grid_edges,
            d_a_grid,
            v_lim,
            a_lim,
            n_mc_mat,
            r_mat,
        )
    else:
        raise ValueError("Invalid base!!")

    if projection_dim == "1d":
        pst = {"f": f_g, "vx": speed_grid, "vx_edges": speed_grid_edges}
    elif projection_dim == "2d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": speed_grid,
            "vy": speed_grid,
            "vx_edges": speed_grid_edges,
            "vy_edges": speed_grid_edges,
        }
    elif projection_dim == "3d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": speed_grid,
            "vy": speed_grid,
            "vz": speed_grid,
            "vx_edges": speed_grid_edges,
            "vy_edges": speed_grid_edges,
            "vz_edges": speed_grid_edges,
        }
    else:
        raise NotImplementedError(
            "2d projection on polar grid is not ready yet!!",
        )

    return pst


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def mc_pol_1d(
    vdf,
    v,
    phi,
    theta,
    d_v,
    d_v_m,
    d_phi,
    d_theta,
    vg_edges,
    d_a_grid,
    v_lim,
    a_lim,
    n_mc,
    r_mat,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : double
        3D skymap particle velocity distribution function.
    v : double
        1D array of instrument speed bins centers.
    phi : double
        1D array of instrument azimuthal angles bins centers.
    theta : double
        1D array of instrument elevation angles bins centers.
    d_v : double
        1D array of instrument speed bins widths.
    d_v_m : double
        1D array of minus velocity from bins centers.
    d_phi : double
        1D array of instrument azimuthal angles bins widths.
    d_theta : double
        1D array of instrument elevation angles bins widths.
    vg_egdes : double
        Bin centers of the velocity of the projection grid.
    d_a_grid : double
        Bin centers of the azimuthal angle of the projection in radians in
        the span [0,2*pi]. If this input is given, the projection will be 2D.
        If it is omitted, the projection will be 1D.
    v_lim : double
        Limits on the out-of-plane velocity interval in 2D and "transverse"
        velocity in 1D.
    a_lim : double
        Angular limit in degrees, can be combined with v_lim.
    n_mc : double
        Number of Monte-Carlo particle for the corresponding instrument bins.
    r_mat : double
        Frame transformation matrix.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros(n_vg)

    for i in numba.prange(n_v):
        for j in range(n_ph):
            for k in range(n_th):
                n_mc_ijk = n_mc[i, j, k]

                if vdf[i][j][k] == 0.0:
                    continue

                dtau_ijk = v[i] ** 2 * np.cos(theta[k]) * d_v[i] * d_phi[j] * d_theta[k]
                c_ijk = dtau_ijk / n_mc_ijk
                f_ijk = vdf[i, j, k]

                for _ in range(n_mc_ijk):
                    d_v_mc = -random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (random.random() - 0.5) * d_phi[j]
                    d_the_mc = (random.random() - 0.5) * d_theta[k]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = theta[k] + d_the_mc

                    v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                    v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                    v_z = v_mc * sin(theta_mc)

                    # Get velocities in primed coordinate system
                    # vxp = [vx, vy, vz] * xphat'; % all MC points
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    v_z_p = sqrt(pow(v_y_p, 2) + pow(v_z_p, 2))
                    alpha = asin(v_z_p / v_mc)

                    use_point = (v_z_p >= v_lim[0]) * (v_z_p < v_lim[1])
                    use_point = use_point * (alpha >= a_lim[0]) * (alpha < a_lim[1])

                    i_vxg = np.searchsorted(vg_edges[:-2], v_x_p)
                    d_a = d_a_grid[i_vxg]

                    if use_point * (i_vxg < n_vg):
                        f_g[i_vxg] += f_ijk * c_ijk / d_a

    return f_g


@numba.jit(fastmath=True)
def mc_cart_3d(
    vdf,
    v,
    phi,
    theta,
    d_v,
    d_v_m,
    d_phi,
    d_theta,
    vg_edges,
    d_a_grid,
    v_lim,
    a_lim,
    n_mc,
    r_mat,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : numpy.ndarray
        3D skymap particle velocity distribution function.
    v : numpy.ndarray
        1D array of instrument speed bins centers.
    phi : numpy.ndarray
        1D array of instrument azimuthal angles bins centers.
    theta : numpy.ndarray
        1D array of instrument elevation angles bins centers.
    d_v : numpy.ndarray
        1D array of instrument speed bins widths.
    d_v_m : numpy.ndarray
        1D array of minus velocity from bins centers.
    d_phi : numpy.ndarray
        1D array of instrument azimuthal angles bins widths.
    d_theta : numpy.ndarray
        1D array of instrument elevation angles bins widths.
    vg_egdes : double
        Bin centers of the velocity of the projection grid.
    d_a_grid : double
        Bin centers of the azimuthal angle of the projection in radians in
        the span [0,2*pi]. If this input is given, the projection will be 2D.
        If it is omitted, the projection will be 1D.
    v_lim : double
        Limits on the out-of-plane velocity interval in 2D and "transverse"
        velocity in 1D.
    a_lim : double
        Angular limit in degrees, can be combined with v_lim.
    n_mc : double
        Number of Monte-Carlo particle for the corresponding instrument bins.
    r_mat : double
        Frame transformation matrix.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    # Get dimension of the instrument and interpolation grid.
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros((n_vg, n_vg, n_vg))

    for i in numba.prange(n_v):
        for j in range(n_ph):
            for k in range(n_th):
                n_mc_ijk = n_mc[i, j, k]

                if vdf[i][j][k] == 0.0:
                    continue

                dtau_ijk = v[i] ** 2 * cos(theta[k]) * d_v[i] * d_phi[j] * d_theta[k]
                c_ijk = dtau_ijk / n_mc_ijk
                f_ijk = vdf[i, j, k]

                for _ in range(n_mc_ijk):
                    d_v_mc = -random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (random.random() - 0.5) * d_phi[j]
                    d_the_mc = (random.random() - 0.5) * d_theta[k]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = theta[k] + d_the_mc

                    v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                    v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                    v_z = v_mc * sin(theta_mc)

                    # Get velocities in primed coordinate system
                    # vxp = [vx, vy, vz] * xphat'; % all MC points
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z
                    # velocity within [-dVm, +dVp]

                    alpha = asin(v_z_p / v_mc)

                    use_point = v_z_p >= v_lim[0] * v_z_p < v_lim[1]
                    use_point = use_point * alpha >= a_lim[0] * alpha < a_lim[1]

                    i_vxg = np.searchsorted(vg_edges[:-2], v_x_p)
                    i_vyg = np.searchsorted(vg_edges[:-2], v_y_p)
                    i_vzg = np.searchsorted(vg_edges[:-2], v_z_p)

                    if use_point:
                        f_g[i_vxg, i_vyg, i_vzg] += f_ijk * c_ijk / d_a_grid

    return f_g


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def mc_cart_2d(
    vdf,
    v,
    phi,
    theta,
    d_v,
    d_v_m,
    d_phi,
    d_theta,
    vg_edges,
    d_a_grid,
    v_lim,
    a_lim,
    n_mc,
    r_mat,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : double
        3D skymap particle velocity distribution function.
    v : double
        1D array of instrument speed bins centers.
    phi : double
        1D array of instrument azimuthal angles bins centers.
    theta : double
        1D array of instrument elevation angles bins centers.
    d_v : double
        1D array of instrument speed bins widths.
    d_v_m : double
        1D array of minus velocity from bins centers.
    d_phi : double
        1D array of instrument azimuthal angles bins widths.
    d_theta : double
        1D array of instrument elevation angles bins widths.
    vg_egdes : double
        Bin centers of the velocity of the projection grid.
    d_a_grid : double
        Bin centers of the azimuthal angle of the projection in radians in
        the span [0,2*pi]. If this input is given, the projection will be 2D.
        If it is omitted, the projection will be 1D.
    v_lim : double
        Limits on the out-of-plane velocity interval in 2D and "transverse"
        velocity
        in 1D.
    a_lim : double
        Angular limit in degrees, can be combined with v_lim.
    n_mc : double
        Number of Monte-Carlo particle for the corresponding instrument bins.
    r_mat : double
        Frame transformation matrix.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    # Get dimension of the instrument and interpolation grid.
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros((n_vg, n_vg))

    for i in range(n_v):
        for j in range(n_ph):
            for k in range(n_th):
                n_mc_ijk = n_mc[i, j, k]

                if vdf[i][j][k] == 0.0:
                    continue

                dtau_ijk = v[i] ** 2 * cos(theta[k]) * d_v[i] * d_phi[j] * d_theta[k]
                c_ijk = dtau_ijk / n_mc_ijk
                f_ijk = vdf[i, j, k]

                for _ in range(n_mc_ijk):
                    d_v_mc = -random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (random.random() - 0.5) * d_phi[j]
                    d_the_mc = (random.random() - 0.5) * d_theta[k]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = theta[k] + d_the_mc

                    v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                    v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                    v_z = v_mc * sin(theta_mc)

                    # Get velocities in primed coordinate system
                    # vxp = [vx, vy, vz] * xphat'; % all MC points
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z
                    # velocity within [-dVm, +dVp]

                    alpha = asin(v_z_p / v_mc)

                    use_point = v_z_p >= v_lim[0] * v_z_p < v_lim[1]
                    use_point = use_point * alpha >= a_lim[0] * alpha < a_lim[1]

                    i_vxg = np.searchsorted(vg_edges, v_x_p)
                    i_vyg = np.searchsorted(vg_edges, v_y_p)

                    if use_point:
                        f_g[i_vxg, i_vyg] += f_ijk * c_ijk / d_a_grid

    return f_g
