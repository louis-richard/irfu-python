#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
from math import asin, cos, sin, sqrt

# Third party imports
import numba
import numpy as np

# Implementation notes (numba kernel _mc_project):
# - one accumulator per worker thread, (n_threads, bins), summed at the end:
#   a shared accumulator under numba.prange loses updates when two threads
#   add to the same bin; the number of threads is passed in from Python
#   (numba.get_num_threads() inside a cache=True function disables its disk
#   cache);
# - the bins of the projection grid follow MATLAB's discretize, as
#   irf_int_sph_dist: edges[k] <= v < edges[k + 1], the last bin closed on
#   both ends (np.searchsorted side="right", minus 1); on a grid uniform to
#   near machine precision (_uniform_step), the index is computed and then
#   checked against the edges, which the computed edges can miss by a few
#   ulps (e.g. the edge at v = 0 of a grid centred on zero);
# - the first Monte-Carlo particle of a bin is at its centre, the others
#   uniform in speed, azimuth and sin(elevation) (uniform in solid angle).


def int_sph_dist(
    vdf,
    velocity,
    phi,
    theta,
    velocity_grid,
    phi_grid,
    *,
    xyz=None,
    n_mc=10,
    weight=None,
    v_int=(-np.inf, np.inf),
    a_int=(-180.0, 180.0),
    projection_base="pol",
    projection_dim="1d",
    d_v_m=None,
    d_v_p=None,
    velocity_edges=None,
    velocity_grid_edges=None,
    d_phi=None,
    d_theta=None,
):
    r"""Integrate a spherical distribution function to a line/plane.

    Parameters
    ----------
    vdf : numpy.ndarray
        Phase-space density skymap. Non-finite values (NaN) are treated as
        empty bins.
    velocity : numpy.ndarray
        Velocity of the instrument bins,
    phi : numpy.ndarray
        Azimuthal angle of the instrument bins.
    theta : numpy.ndarray
        Elevation angle of the instrument bins.
    velocity_grid : numpy.ndarray
        Velocity grid for interpolation.
    phi_grid : numpy.ndarray
        Azimuthal angle grid for interpolation.

    Returns
    -------
    pst : dict
        Dictionary with the projected distribution and
        corresponding velocity grid information.

    Other Parameters
    ----------------
    xyz : numpy.ndarray
        Transformation matrix of the projection, the unit vectors x, y (and z)
        as columns. Default is the identity.
    n_mc : int
        Number of Monte-Carlo particles per instrument bin (see `weight`).
        Default is 10.
    v_int : array_like
        Limits [min, max] of the speed transverse to the projection axis (1d)
        or out of the projection plane (2d, 3d). Default is no limit.
    a_int : array_like
        Limits [min, max] of the angle from the projection axis or plane, in
        degrees. Default is [-180, 180].
    projection_base : {"pol", "cart"}
        Base of the projection grid (case-insensitive). Default is "pol".
    projection_dim : {"1d", "2d", "3d"}
        Dimension of the projection (case-insensitive). Default is "1d". A polar
        projection is 1d if `phi_grid` is None; `phi_grid` is not used by the
        cartesian projections.
    weight : {None, "lin", "log"}
        Number of Monte-Carlo particles per bin: `n_mc` in every non-empty bin
        (None), or `n_mc` on average, in proportion to vdf ("lin") or to
        1 + ln(vdf / min(vdf)) over the bins with data ("log", whatever the
        units of vdf). Default is None. irfu-matlab's "log", log10(1 + vdf),
        equals "lin" for phase-space densities in SI units.
    d_v_m, d_v_p : numpy.ndarray
        Speed widths below and above the speed of each instrument bin, i.e., the
        bin spans [velocity - d_v_m, velocity + d_v_p]. Both must be given, and
        take precedence over `velocity_edges`.
    velocity_edges : numpy.ndarray
        Edges of the instrument speed bins, shape (len(velocity) + 1,). If
        neither `d_v_m`/`d_v_p` nor `velocity_edges` is given, the edges are at
        the geometric mean of neighbouring speeds.
    velocity_grid_edges : numpy.ndarray
        Edges of the bins of the projection grid, instead of the midpoints of
        `velocity_grid`.
    d_phi, d_theta : numpy.ndarray
        Azimuth and elevation widths of the instrument bins (rad). Default is
        the median spacing of `phi` and `theta`.

    Raises
    ------
    ValueError
        If the speed widths are negative or extend below zero speed, if only
        one of `d_v_m` and `d_v_p` is given, if `weight` is not one of the
        above, or if a cartesian projection grid is not uniform.
    NotImplementedError
        If the projection is not a 1d polar, or a 2d or 3d cartesian one.

    Notes
    -----
    The Monte-Carlo particles are drawn uniformly within each speed bin. This
    differs from irfu-matlab's irf_int_sph_dist, which draws them in
    [v - 1.5 dV, v - 0.5 dV] (one bin too low) and uses the spacing to the lower
    neighbour as the bin width when no edges are given. For a drifting
    Maxwellian on FPI-like bins, that biases the reduced distribution to lower
    speeds (bulk speed ~7 % and temperature ~10 % too low, density ~6 % too low).

    """

    # Non-finite values are empty bins (otherwise NaNs spread to the projected
    # bins, and with a weighting make the whole projection 0)
    vdf = np.where(np.isfinite(vdf), vdf, 0.0)

    # Coordinates system transformation matrix
    if xyz is None:
        xyz = np.eye(3)

    # Make sure the transformation matrix is orthonormal.
    x_phat = xyz[:, 0] / np.linalg.norm(xyz[:, 0])  # re-normalize
    y_phat = xyz[:, 1] / np.linalg.norm(xyz[:, 1])  # re-normalize

    z_phat = np.cross(x_phat, y_phat)
    z_phat /= np.linalg.norm(z_phat)
    y_phat = np.cross(z_phat, x_phat)

    # Define the rotation matrix from original to primed frame
    r_mat = np.transpose(np.stack([x_phat, y_phat, z_phat]), [1, 0])

    # Number of Monte Carlo iterations and how number of MC points is
    # weighted to data.
    # Another weight (e.g. "linear") was taken as None
    if weight not in (None, "lin", "log"):
        raise ValueError(f'weight must be None, "lin" or "log", not {weight!r}')

    # limit on out-of-plane velocity
    v_lim = np.array(v_int, dtype=np.float64)

    # limit on azymuthal angle from projection plane
    a_lim = np.array(a_int, dtype=np.float64)
    a_lim = np.deg2rad(a_lim)

    # Projection dimension and base
    projection_base = str(projection_base).lower()
    projection_dim = str(projection_dim).lower()

    # Azimuthal and elevation angles steps. Assumed to be constant
    # if not provided.
    if d_phi is None:
        d_phi = np.abs(np.median(np.diff(phi))) * np.ones_like(phi)

    if d_theta is None:
        d_theta = np.abs(np.median(np.diff(theta))) * np.ones_like(theta)

    # Speed widths below (d_v_m) and above (d_v_p) the speed of each instrument
    # bin. The Monte-Carlo particles are drawn uniformly in [v - d_v_m, v + d_v_p].
    # One of them alone was ignored
    if (d_v_m is None) != (d_v_p is None):
        raise ValueError("d_v_m and d_v_p must be given together")

    if d_v_m is not None:
        d_v_m = np.asarray(d_v_m, dtype=np.float64)
        d_v_p = np.asarray(d_v_p, dtype=np.float64)
    else:
        if velocity_edges is None:
            velocity_edges = _speed_bin_edges(velocity)
        elif len(velocity_edges) != len(velocity) + 1:
            raise ValueError("velocity_edges must have len(velocity) + 1 values")

        d_v_m = velocity - velocity_edges[:-1]
        d_v_p = velocity_edges[1:] - velocity

    d_v_m = d_v_m.astype(np.float64)
    d_v_p = d_v_p.astype(np.float64)
    _check_speed_widths(velocity, d_v_m, d_v_p)
    d_v = d_v_m + d_v_p

    # A polar projection is 1d without the azimuthal grid of the projection
    # plane; the supported projections are 1d polar, 2d and 3d cartesian (the
    # azimuthal grid width of the 2d and 3d polar ones was computed, unused)
    if projection_base == "pol" and phi_grid is None:
        projection_dim = "1d"

    if (projection_base, projection_dim) not in _PROJECTIONS:
        raise NotImplementedError(
            f"{projection_dim} projection on {projection_base} grid is not ready yet!!",
        )

    n_dim = int(projection_dim[0])

    # Speed grid bins edges
    if velocity_grid_edges is None:
        velocity_grid_diff = np.diff(velocity_grid)
        velocity_grid_edges = np.zeros(len(velocity_grid) + 1)
        velocity_grid_edges[0] = velocity_grid[0] - velocity_grid_diff[0] / 2.0
        velocity_grid_edges[1:-1] = velocity_grid[:-1] + velocity_grid_diff / 2.0
        velocity_grid_edges[-1] = velocity_grid[-1] + velocity_grid_diff[-1] / 2.0
    else:
        velocity_grid = velocity_grid_edges[:-1] + np.diff(velocity_grid_edges) / 2.0

    if projection_base == "pol":
        d_v_grid = np.diff(velocity_grid_edges)
    else:
        # All bins must have the same width (to 1 %), which is used for all
        d_v_grid = np.diff(velocity_grid_edges)
        mean_diff = np.mean(d_v_grid)

        if not np.all(np.abs(d_v_grid / mean_diff - 1) < 1e-2):
            raise ValueError("For a cartesian grid, all velocity bins must be equal!!")

        d_v_grid = mean_diff

    # Weighting of number of Monte Carlo particles
    n_mc_mat = _mc_particles(vdf, n_mc, weight)

    # Element of the projection grid: the speed width of each bin of the 1d
    # polar grid, the volume of the bins of the cartesian grids
    if projection_base == "pol":
        d_a_grid = d_v_grid.astype(np.float64)
    else:
        d_a_grid = np.full(1, d_v_grid**n_dim, dtype=np.float64)

    # The grid indices computed on a cartesian grid uniform to near machine
    # precision, searched otherwise (see the implementation notes)
    v_step = _uniform_step(velocity_grid_edges) if n_dim > 1 else 0.0

    f_g = _mc_project(
        vdf,
        velocity,
        phi,
        theta,
        d_v,
        d_v_m,
        d_phi,
        d_theta,
        velocity_grid_edges,
        d_a_grid,
        v_lim,
        a_lim,
        n_mc_mat,
        r_mat,
        v_step,
        n_dim,
        numba.get_num_threads(),
    )
    f_g = f_g.reshape((len(velocity_grid),) * n_dim)

    if projection_dim == "2d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": velocity_grid,
            "vy": velocity_grid,
            "vx_edges": velocity_grid_edges,
            "vy_edges": velocity_grid_edges,
        }
    elif projection_dim == "3d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": velocity_grid,
            "vy": velocity_grid,
            "vz": velocity_grid,
            "vx_edges": velocity_grid_edges,
            "vy_edges": velocity_grid_edges,
            "vz_edges": velocity_grid_edges,
        }
    else:
        pst = {"f": f_g, "vx": velocity_grid, "vx_edges": velocity_grid_edges}

    return pst


# Supported projections: (base, dimension)
_PROJECTIONS = [("pol", "1d"), ("cart", "2d"), ("cart", "3d")]


def _mc_particles(vdf, n_mc, weight):
    r"""Number of Monte-Carlo particles in each bin of vdf: n_mc in every bin
    with data (None), or n_mc on average in proportion to vdf ("lin") or to
    1 + ln(vdf / min(vdf)) ("log")."""
    n_sum = n_mc * np.sum(vdf != 0)  # total number of Monte Carlo particles

    if weight == "lin":
        n_mc_mat = np.ceil(n_sum / np.sum(vdf) * vdf)
    elif weight == "log":
        # 1 + ln(vdf / vdf_min) over the bins with data: logarithmic whatever
        # the units of vdf (irfu-matlab's log10(vdf + 1) equals "lin" for SI
        # phase-space densities, f << 1), and at least 1 in each bin
        positive = vdf > 0
        log_w = np.zeros_like(vdf, dtype=np.float64)

        if np.any(positive):
            log_w[positive] = 1 + np.log(vdf[positive] / np.min(vdf[positive]))
            n_mc_mat = np.ceil(n_sum / np.sum(log_w) * log_w)
        else:
            n_mc_mat = log_w
    else:
        n_mc_mat = np.zeros_like(vdf)
        n_mc_mat[vdf != 0] = n_mc

    return n_mc_mat.astype(int)


def _speed_bin_edges(velocity):
    r"""Default edges of the instrument speed bins.

    Instrument channels are log-spaced in energy (hence in speed), so the edges
    are placed at the geometric mean of neighbouring speeds, and the outer edges
    are extrapolated symmetrically in log space. If any speed is not strictly
    positive (e.g., channels below the spacecraft potential set to 0), the
    arithmetic midpoints are used instead, with the lower edge clipped at 0.

    Parameters
    ----------
    velocity : numpy.ndarray
        Speeds of the instrument bins, in increasing order.

    Returns
    -------
    numpy.ndarray
        Edges of the speed bins, shape (len(velocity) + 1,).

    """
    velocity = np.asarray(velocity, dtype=np.float64)

    if len(velocity) < 2:
        raise ValueError("At least two speed bins are needed to infer the edges")

    if np.all(velocity > 0.0):
        mid = np.sqrt(velocity[:-1] * velocity[1:])
        low = velocity[0] ** 2 / mid[0]
        upp = velocity[-1] ** 2 / mid[-1]
    else:
        mid = (velocity[:-1] + velocity[1:]) / 2.0
        low = max(2.0 * velocity[0] - mid[0], 0.0)
        upp = 2.0 * velocity[-1] - mid[-1]

    return np.hstack([low, mid, upp])


def _check_speed_widths(velocity, d_v_m, d_v_p):
    r"""Check that the speed bins [velocity - d_v_m, velocity + d_v_p] are valid.

    Raises
    ------
    ValueError
        If the widths don't match the speeds, are negative, or extend below
        zero speed (the Monte-Carlo particles would then have negative speeds,
        i.e., the opposite direction).

    """
    n_v = len(velocity)

    if d_v_m.shape != (n_v,) or d_v_p.shape != (n_v,):
        raise ValueError("d_v_m and d_v_p must have the same length as velocity")

    if np.any(d_v_m < 0.0) or np.any(d_v_p < 0.0):
        raise ValueError("Speed bin widths must be non-negative")

    if np.any(velocity - d_v_m < 0.0):
        raise ValueError("Speed bins must not extend below zero speed")


def _uniform_step(edges):
    r"""Return the bin width if ``edges`` is uniformly spaced to a tight
    (near machine-precision) tolerance, otherwise 0.0: the 1 % uniformity
    required of a cartesian grid is too loose for index arithmetic (the drift
    over many bins could shift the index by more than one bin), and the
    kernel searches the edges when it is 0.0."""

    diffs = np.diff(edges)
    if diffs.size == 0:
        return 0.0

    ref = diffs[0]
    if ref == 0.0:
        return 0.0

    if np.allclose(diffs, ref, rtol=1e-9, atol=1e-12):
        return float(ref)

    return 0.0


@numba.njit(cache=True, inline="always")
def _grid_index(value, edges, v_step):
    r"""Index of the bin of the grid `edges` holding `value` (within the
    grid), MATLAB's discretize convention (see the implementation notes):
    computed and checked against the edges for v_step > 0, searched
    otherwise."""
    n_g = len(edges) - 1

    if v_step > 0.0:
        # Non-negative: value >= edges[0] here
        idx = min(int((value - edges[0]) / v_step), n_g - 1)

        while idx > 0 and value < edges[idx]:
            idx -= 1

        while idx < n_g - 1 and value >= edges[idx + 1]:
            idx += 1
    else:
        idx = np.searchsorted(edges, value, side="right") - 1

    # The last bin is closed on both ends
    return min(idx, n_g - 1)


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_project(
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
    v_step,
    n_dim,
    n_threads,
):
    r"""Monte-Carlo projection of a skymap onto a grid, flattened: 1d polar
    (n_dim = 1: the speed along the first axis, the transverse speed limited
    by v_lim and a_lim), 2d or 3d cartesian (the out-of-plane speed limited
    by v_lim and a_lim).

    Parameters
    ----------
    vdf : numpy.ndarray
        Skymap (speed, azimuth, elevation), without NaN.
    v, phi, theta : numpy.ndarray
        Speeds, azimuths and elevations (rad) of the instrument bins.
    d_v, d_v_m : numpy.ndarray
        Speed widths of the bins, and their parts below the speeds.
    d_phi, d_theta : numpy.ndarray
        Azimuth and elevation widths of the bins (rad).
    vg_edges : numpy.ndarray
        Edges of the projection grid (the same on each axis).
    d_a_grid : numpy.ndarray
        Speed widths of the 1d polar grid, or the volume of a bin of the
        cartesian grids (one value).
    v_lim, a_lim : numpy.ndarray
        Limits of the transverse or out-of-plane speed, and of the angle from
        the projection axis or plane (rad).
    n_mc : numpy.ndarray
        Number of Monte-Carlo particles of each instrument bin.
    r_mat : numpy.ndarray
        Frame transformation matrix.
    v_step : float
        Width of the bins of a uniform grid (index arithmetic), or 0.0.
    n_dim : int
        Dimension of the projection.
    n_threads : int
        Number of worker threads.

    Returns
    -------
    numpy.ndarray
        Projected distribution, (n_g ** n_dim,) with n_g bins on each axis.

    """
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1

    # Invariants of the speeds and elevations
    v2 = v**2
    cos_theta = np.cos(theta)
    sin_theta_1_arr = np.sin(theta - 0.5 * d_theta)
    d_sin_theta_arr = np.sin(theta + 0.5 * d_theta) - sin_theta_1_arr

    f_g_threads = np.zeros((n_threads, n_vg**n_dim))

    for i in numba.prange(n_v):
        tid = numba.get_thread_id()

        for j in range(n_ph):
            for k in range(n_th):
                n_mc_ijk = n_mc[i, j, k]

                if vdf[i][j][k] == 0.0:
                    continue

                dtau_ijk = v2[i] * cos_theta[k] * d_v[i] * d_phi[j] * d_theta[k]
                c_ijk = dtau_ijk / n_mc_ijk
                f_ijk = vdf[i, j, k]

                for l_mc in range(n_mc_ijk):
                    # Speed, azimuth and elevation of the particle (the first
                    # at the centre of the bin)
                    if l_mc == 0:
                        d_v_mc = 0.0
                        d_phi_mc = 0.0
                    else:
                        d_v_mc = random.random() * d_v[i] - d_v_m[i]
                        d_phi_mc = (random.random() - 0.5) * d_phi[j]

                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc

                    # Uniform in sin(elevation): uniform in solid angle
                    if l_mc == 0:
                        theta_mc = theta[k]
                    else:
                        sin_theta_mc = (
                            sin_theta_1_arr[k] + random.random() * d_sin_theta_arr[k]
                        )
                        theta_mc = asin(sin_theta_mc)

                    v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                    v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                    v_z = v_mc * sin(theta_mc)

                    # Velocity in the primed coordinate system
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    # Transverse speed (1d) or out-of-plane speed, and the
                    # angle from the projection axis or plane, within limits
                    if n_dim == 1:
                        v_z_p = sqrt(v_y_p**2 + v_z_p**2)

                    alpha = asin(v_z_p / v_mc)

                    use_point = v_lim[0] <= v_z_p < v_lim[1]
                    use_point = use_point and (a_lim[0] <= alpha < a_lim[1])

                    # Particles out of the grid
                    if v_x_p > vg_edges[-1] or v_x_p < vg_edges[0]:
                        continue

                    if n_dim > 1 and (v_y_p > vg_edges[-1] or v_y_p < vg_edges[0]):
                        continue

                    if n_dim > 2 and (v_z_p > vg_edges[-1] or v_z_p < vg_edges[0]):
                        continue

                    # Flat index of the bin (x, y, z), C order
                    i_vxg = _grid_index(v_x_p, vg_edges, v_step)
                    idx = i_vxg

                    if n_dim > 1:
                        idx = idx * n_vg + _grid_index(v_y_p, vg_edges, v_step)

                    if n_dim > 2:
                        idx = idx * n_vg + _grid_index(v_z_p, vg_edges, v_step)

                    d_a = d_a_grid[i_vxg] if n_dim == 1 else d_a_grid[0]

                    if use_point:
                        f_g_threads[tid, idx] += f_ijk * c_ijk / d_a

    return f_g_threads.sum(axis=0)
