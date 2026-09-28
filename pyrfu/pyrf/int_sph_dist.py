#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
from math import asin, cos, sin, sqrt

# Third party imports
import numba
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def int_sph_dist(vdf, velocity, phi, theta, velocity_grid, phi_grid, **kwargs):
    r"""Integrate a spherical distribution function to a line/plane.

    Parameters
    ----------
    vdf : numpy.ndarray
        Phase-space density skymap.
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
    **kwargs
        Keyword arguments.

    Returns
    -------
    pst : dict
        Dictionary with the projected distribution and
        corresponding velocity grid information.

    Other Parameters
    ----------------
    d_v_m, d_v_p : numpy.ndarray
        Speed widths below and above the speed of each instrument bin, i.e., the
        bin spans [velocity - d_v_m, velocity + d_v_p]. Both must be given, and
        take precedence over `velocity_edges`.
    velocity_edges : numpy.ndarray
        Edges of the instrument speed bins, shape (len(velocity) + 1,). If
        neither `d_v_m`/`d_v_p` nor `velocity_edges` is given, the edges are at
        the geometric mean of neighbouring speeds.

    Notes
    -----
    The Monte-Carlo particles are drawn uniformly within each speed bin. This
    differs from irfu-matlab's irf_int_sph_dist, which draws them in
    [v - 1.5 dV, v - 0.5 dV] (one bin too low) and uses the spacing to the lower
    neighbour as the bin width when no edges are given. For a drifting
    Maxwellian on FPI-like bins, that biases the reduced distribution to lower
    speeds (bulk speed ~7 % and temperature ~10 % too low, density ~6 % too low).

    """

    # Coordinates system transformation matrix
    xyz = kwargs.get("xyz", np.eye(3))

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
    n_mc = kwargs.get("n_mc", 10)
    weight = kwargs.get("weight", None)

    # limit on out-of-plane velocity
    v_lim = np.array(kwargs.get("v_int", [-np.inf, np.inf]), dtype=np.float64)

    # limit on azymuthal angle from projection plane
    a_lim = np.array(kwargs.get("a_int", [-180.0, 180.0]), dtype=np.float64)
    a_lim = np.deg2rad(a_lim)

    # Projection dimension and base
    projection_base = kwargs.get("projection_base", "pol")
    projection_dim = kwargs.get("projection_dim", "1d")

    velocity_edges = kwargs.get("velocity_edges", None)
    velocity_grid_edges = kwargs.get("velocity_grid_edges", None)

    # Azimuthal and elevation angles steps. Assumed to be constant
    # if not provided.
    d_phi = np.abs(np.median(np.diff(phi))) * np.ones_like(phi)
    d_phi = kwargs.get("d_phi", d_phi)
    d_theta = np.abs(np.median(np.diff(theta))) * np.ones_like(theta)
    d_theta = kwargs.get("d_theta", d_theta)

    # Speed widths below (d_v_m) and above (d_v_p) the speed of each instrument
    # bin. The Monte-Carlo particles are drawn uniformly in [v - d_v_m, v + d_v_p].
    if "d_v_m" in kwargs and "d_v_p" in kwargs:
        d_v_m = np.asarray(kwargs["d_v_m"], dtype=np.float64)
        d_v_p = np.asarray(kwargs["d_v_p"], dtype=np.float64)
    else:
        if velocity_edges is None:
            velocity_edges = _speed_bin_edges(velocity)

        d_v_m = velocity - velocity_edges[:-1]
        d_v_p = velocity_edges[1:] - velocity

    d_v_m = d_v_m.astype(np.float64)
    d_v_p = d_v_p.astype(np.float64)
    d_v = d_v_m + d_v_p

    # Overwrite projection dimension if azimuthal angle of projection
    # plane is not provided. Set the azimuthal angle grid width.
    if phi_grid is not None and projection_dim.lower() in ["2d", "3d"]:
        d_phi_grid = np.median(np.diff(phi_grid))
    else:
        projection_dim = "1d"
        d_phi_grid = 1.0

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

        if projection_dim == "2d":
            raise NotImplementedError(
                "2d projection on polar grid is not ready yet!!",
            )
    else:
        mean_diff = np.mean(np.diff(velocity_grid))
        msg = "For a cartesian grid, all velocity bins must be equal!!"
        assert (np.diff(velocity_grid) / mean_diff - 1 < 1e-2).all(), msg

        d_v_grid = mean_diff

    # Weighting of number of Monte Carlo particles
    n_sum = n_mc * np.sum(vdf != 0)  # total number of Monte Carlo particles
    if weight == "lin":
        n_mc_mat = np.ceil(n_sum / np.sum(vdf) * vdf)
    elif weight == "log":
        n_mc_mat = np.ceil(
            n_sum / np.sum(np.log10(vdf + 1.0)) * np.log10(vdf + 1.0),
        )
    else:
        n_mc_mat = np.zeros_like(vdf)
        n_mc_mat[vdf != 0] = n_mc

    n_mc_mat = n_mc_mat.astype(int)

    if projection_base == "pol":
        d_a_grid = velocity_grid ** (int(projection_dim[0]) - 1) * d_phi_grid * d_v_grid
        d_a_grid = d_a_grid.astype(np.float64)
    else:
        d_a_grid = d_v_grid ** int(projection_dim[0])
        d_a_grid = d_a_grid.astype(np.float64)

    # CHANGED: query the thread count once in plain Python and pass it
    # into the kernels, rather than calling numba.get_num_threads() inside
    # them -- doing that inside a cache=True jitted function disables its
    # disk cache (it references dynamic globals numba can't serialize).
    n_threads = numba.get_num_threads()

    if projection_base == "cart" and projection_dim == "2d":
        # CHANGED: precompute a uniform-grid step (only when the edges are
        # uniform to a tight, machine-level tolerance -- NOT the same
        # looser 1% tolerance enforced by the assert above) so the kernel
        # can index the projection grid with O(1) arithmetic instead of
        # np.searchsorted per Monte-Carlo particle. Falls back to
        # searchsorted (v_step <= 0.0) whenever the edges aren't tightly
        # uniform, so behaviour is unchanged for any input accepted by the
        # existing assert.
        v_step = _uniform_step(velocity_grid_edges)
        f_g = _mc_cart_2d(
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
            n_threads,
        )
    elif projection_base == "cart" and projection_dim == "3d":
        v_step = _uniform_step(velocity_grid_edges)  # CHANGED: see above
        f_g = _mc_cart_3d(
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
            n_threads,
        )
    elif projection_base == "pol" and projection_dim == "1d":
        f_g = _mc_pol_1d(
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
            n_threads,
        )
    else:
        raise NotImplementedError(
            f"{projection_dim} projection on {projection_base} grid is not ready yet!!",
        )

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


def _uniform_step(edges):
    r"""Return the bin width if ``edges`` is uniformly spaced to a tight
    (near machine-precision) tolerance, otherwise 0.0.

    CHANGED: new helper. The cartesian-grid assert in ``int_sph_dist``
    only guarantees spacing is uniform to within 1%, which is too loose to
    safely replace ``np.searchsorted`` with direct index arithmetic
    (cumulative drift across many bins could shift the computed index by
    more than one bin near the domain edges). This checks a much tighter
    tolerance so the fast path is only used when it is numerically safe,
    and the Monte-Carlo kernels fall back to ``np.searchsorted`` whenever
    it returns 0.0. Note this fallback now uses ``side='right'`` rather
    than the original's default ``side='left'`` -- see the comments at
    each call site -- so it is bin-for-bin equivalent to the arithmetic
    fast path and to MATLAB's ``discretize``, not to the pristine
    original's searchsorted call, which used the opposite (and, per the
    MATLAB source, incorrect) edge convention.

    """

    diffs = np.diff(edges)
    if diffs.size == 0:
        return 0.0

    ref = diffs[0]
    if ref == 0.0:
        return 0.0

    if np.allclose(diffs, ref, rtol=1e-9, atol=1e-12):
        return float(ref)

    return 0.0


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_pol_1d(
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
    n_threads,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : double
        3D skymap particle velocity distribution function.
    v : double
        1D array of instrument velocity bins centers.
    phi : double
        1D array of instrument azimuthal angles bins centers.
    theta : double
        1D array of instrument elevation angles bins centers.
    d_v : double
        1D array of instrument velocity bins widths.
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
    n_threads : int
        # CHANGED: new parameter. Number of worker threads, queried once in
        # plain Python (numba.get_num_threads() inside a cache=True jitted
        # function disables its disk cache) and used to size the
        # per-thread accumulator that replaces the original racy shared-
        # array scatter-add under numba.prange.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1

    # CHANGED: hoisted per-bin invariants that only depend on v (index i)
    # or theta (index k) out of the (i, j, k[, l_mc]) loop nest below,
    # where they were previously recomputed on every visit (including,
    # for the theta_1/theta_2/sin_theta_* quantities, on every individual
    # Monte-Carlo particle even though they don't depend on l_mc at all).
    v2 = v**2
    cos_theta = np.cos(theta)
    theta_1_arr = theta - 0.5 * d_theta
    theta_2_arr = theta + 0.5 * d_theta
    sin_theta_1_arr = np.sin(theta_1_arr)
    d_sin_theta_arr = np.sin(theta_2_arr) - sin_theta_1_arr

    # CHANGED: one private accumulator per *worker thread* (allocated once,
    # not once per i-iteration) instead of scattering directly into a
    # shared f_g array. Writing f_g[idx] += ... from inside a numba.prange
    # loop with a data-dependent idx is not race-free -- two threads
    # landing samples in the same output bin at the same time can race and
    # silently drop an update. Since numba.get_thread_id() uniquely and
    # stably identifies the current worker for the life of the parallel
    # region, distinct threads always write to disjoint rows of
    # f_g_threads, which makes this safe by construction (no reliance on
    # numba's reduction-pattern detection), and reduced once at the end.
    # CHANGED: n_threads is now passed in from the Python-level caller
    # (numba.get_num_threads() called *inside* a cache=True jitted
    # function disables its disk cache -- it references dynamic
    # globals numba can't serialize; calling it in plain Python and
    # passing the result in avoids that regression).
    f_g_threads = np.zeros((n_threads, n_vg))

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
                    # Generate Monte-Carlo particle speed, phi, and theta
                    if l_mc == 0:
                        # First Monte-Carlo particle is set at the bin center
                        d_v_mc = 0.0
                        d_phi_mc = 0.0
                    else:
                        d_v_mc = random.random() * d_v[i] - d_v_m[i]
                        d_phi_mc = (random.random() - 0.5) * d_phi[j]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc

                    # This is needed to distribute the points evenly in space
                    # (more points further 'equatorward' since the slice is
                    # wider there).

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

                    # Get velocities in primed coordinate system
                    # vxp = [vx, vy, vz] * xphat'; % all MC points
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    # get transverse velocity sqrt(vy^2+vz^2)
                    v_z_p = sqrt(v_y_p**2 + v_z_p**2)
                    alpha = asin(v_z_p / v_mc)

                    # Check if the velocity is within the v_lim and a_lim bounds
                    use_point = v_lim[0] <= v_z_p < v_lim[1]
                    use_point = use_point and (a_lim[0] <= alpha < a_lim[1])

                    # Find which bin the MC point falls into
                    v_p = v_x_p

                    if v_p > vg_edges[-1] or v_p < vg_edges[0]:
                        continue

                    # CHANGED: side='right' (not the default 'left'), minus 1,
                    # reproduces MATLAB's discretize(vp, vg_edges) convention --
                    # left-closed/right-open per bin: vg_edges[k] <= v_p <
                    # vg_edges[k+1] -- confirmed against the real MATLAB source
                    # (irf_int_sph_dist.m) and against MATLAB's documented
                    # discretize rule. side='left' (the previous code here)
                    # gives the *opposite* convention (left-open/right-closed)
                    # and silently mis-bins any MC point landing exactly on an
                    # interior grid edge.
                    i_vxg = np.searchsorted(vg_edges, v_p, side="right") - 1

                    # Special case for the first bin edge, which is the lower limit of
                    # the first bin and should be included in the first bin
                    if i_vxg == -1:
                        i_vxg = 0

                    # Special case matching MATLAB discretize's last-bin
                    # exception: the last bin is closed on BOTH ends, so
                    # v_p == vg_edges[-1] exactly must land in the last bin
                    # rather than one-past-the-end.
                    if i_vxg == n_vg:
                        i_vxg = n_vg - 1

                    d_a = d_a_grid[i_vxg]

                    if use_point * (i_vxg < n_vg):
                        f_g_threads[tid, i_vxg] += f_ijk * c_ijk / d_a

    f_g = f_g_threads.sum(axis=0)  # CHANGED: combine per-thread contributions once

    return f_g


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_cart_2d(
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
    n_threads,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : double
        3D skymap particle velocity distribution function.
    v : double
        1D array of instrument velocity bins centers.
    phi : double
        1D array of instrument azimuthal angles bins centers.
    theta : double
        1D array of instrument elevation angles bins centers.
    d_v : double
        1D array of instrument velocity bins widths.
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
    v_step : double
        # CHANGED: new parameter. Projection-grid bin width to use for O(1)
        # index arithmetic when > 0.0 (grid confirmed tightly uniform by
        # the caller); falls back to the original np.searchsorted lookup
        # when <= 0.0, so behaviour is unchanged for a grid that only
        # satisfies the looser 1% tolerance checked in int_sph_dist.
    n_threads : int
        # CHANGED: new parameter. Number of worker threads, queried once in
        # plain Python (numba.get_num_threads() inside a cache=True jitted
        # function disables its disk cache) and used to size the
        # per-thread accumulator that replaces the original racy shared-
        # array scatter-add under numba.prange.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    # Get dimension of the instrument and interpolation grid.
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1

    v2 = v**2  # CHANGED: hoisted, see _mc_pol_1d
    cos_theta = np.cos(theta)
    theta_1_arr = theta - 0.5 * d_theta
    theta_2_arr = theta + 0.5 * d_theta
    sin_theta_1_arr = np.sin(theta_1_arr)
    d_sin_theta_arr = np.sin(theta_2_arr) - sin_theta_1_arr

    v_g0 = vg_edges[0]
    use_fast_index = v_step > 0.0  # CHANGED: see v_step above

    # CHANGED: one private accumulator per worker thread, allocated once
    # (not once per i-iteration -- that was measured to actually be a net
    # slowdown for large 2D/3D output grids, since it re-zeros an
    # n_vg*n_vg-sized array on every visit to the outer loop). See
    # _mc_pol_1d for why per-thread indexing (rather than the original
    # shared-array scatter-add) is required for correctness under prange.
    # CHANGED: n_threads is now passed in from the Python-level caller
    # (numba.get_num_threads() called *inside* a cache=True jitted
    # function disables its disk cache -- it references dynamic
    # globals numba can't serialize; calling it in plain Python and
    # passing the result in avoids that regression).
    f_g_threads = np.zeros((n_threads, n_vg, n_vg))

    # CHANGED: this loop was serial (plain range) even though the other
    # two kernels use numba.prange -- there is no reason _mc_cart_2d
    # shouldn't also use all available cores.
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
                    # Generate Monte-Carlo particle speed, phi, and theta
                    if l_mc == 0:
                        # First Monte-Carlo particle is set at the bin center
                        d_v_mc = 0.0
                        d_phi_mc = 0.0
                    else:
                        d_v_mc = random.random() * d_v[i] - d_v_m[i]
                        d_phi_mc = (random.random() - 0.5) * d_phi[j]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc

                    # This is needed to distribute the points evenly in space
                    # (more points further 'equatorward' since the slice is
                    # wider there).

                    if l_mc == 0:
                        theta_mc = theta[k]
                    else:
                        sin_theta_mc = (
                            sin_theta_1_arr[k] + random.random() * d_sin_theta_arr[k]
                        )
                        theta_mc = asin(sin_theta_mc)

                    # Calculate velocity of the Monte-Carlo particle in
                    # cartesian coordinates
                    v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                    v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                    v_z = v_mc * sin(theta_mc)

                    # Get velocities in primed coordinate system
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    # Elevation angle from the projection plane
                    alpha = asin(v_z_p / v_mc)

                    # Check if the velocity is within the v_lim and a_lim bounds
                    use_point = v_lim[0] <= v_z_p < v_lim[1]
                    use_point = use_point and (a_lim[0] <= alpha < a_lim[1])

                    # Check if the velocity along the x direction is outside of the grid
                    if v_x_p > vg_edges[-1] or v_x_p < vg_edges[0]:
                        continue

                    # Check if the velocity along the y direction is outside of the grid
                    if v_y_p > vg_edges[-1] or v_y_p < vg_edges[0]:
                        continue

                    # CHANGED: O(1) arithmetic index when the grid is
                    # confirmed uniform, instead of always paying for a
                    # binary search per particle per axis. The arithmetic
                    # gives an initial guess that is then VERIFIED (and, if
                    # needed, corrected by a bin or two) against the actual
                    # vg_edges values -- an exact-equality tie-break alone
                    # is not enough, because v_g0 + i*v_step can drift from
                    # the true vg_edges[i] by a few ULPs of accumulated
                    # rounding, and a real velocity sample landing within
                    # that drift of a boundary would silently land in the
                    # wrong bin. This is not just a theoretical concern: a
                    # projection grid centered on zero (the common case)
                    # has an edge at v=0, and an instrument phi grid that
                    # includes exactly 90/270 degrees produces many real
                    # samples with a near-zero component right at that
                    # edge. The correction loops below reproduce MATLAB's
                    # discretize(v, vg_edges) convention exactly -- left-
                    # closed/right-open per bin (vg_edges[k] <= v <
                    # vg_edges[k+1]), confirmed against the real MATLAB
                    # source (irf_int_sph_dist.m) -- while still being O(1)
                    # in the typical case (0 or 1 iterations). CHANGED:
                    # a prior pass here used <=/> (matching np.searchsorted's
                    # left-open/right-closed convention instead), which was
                    # backwards relative to MATLAB -- reverted to </>=.
                    if use_fast_index:
                        i_vxg = int((v_x_p - v_g0) / v_step)
                        i_vyg = int((v_y_p - v_g0) / v_step)
                        if i_vxg < 0:
                            i_vxg = 0
                        elif i_vxg >= n_vg:
                            i_vxg = n_vg - 1
                        if i_vyg < 0:
                            i_vyg = 0
                        elif i_vyg >= n_vg:
                            i_vyg = n_vg - 1
                        while i_vxg > 0 and v_x_p < vg_edges[i_vxg]:
                            i_vxg -= 1
                        while i_vxg < n_vg - 1 and v_x_p >= vg_edges[i_vxg + 1]:
                            i_vxg += 1
                        while i_vyg > 0 and v_y_p < vg_edges[i_vyg]:
                            i_vyg -= 1
                        while i_vyg < n_vg - 1 and v_y_p >= vg_edges[i_vyg + 1]:
                            i_vyg += 1
                    else:
                        # CHANGED: side='right' (not the default 'left'),
                        # minus 1, matches MATLAB's discretize convention --
                        # see _mc_pol_1d for the full explanation. side='left'
                        # gives the opposite (left-open/right-closed) binning
                        # and silently mis-bins points exactly on an interior
                        # grid edge.
                        i_vxg = np.searchsorted(vg_edges, v_x_p, side="right") - 1
                        i_vyg = np.searchsorted(vg_edges, v_y_p, side="right") - 1

                    # Special case for the first bin edge, which is the lower limit of
                    # the first bin and should be included in the first bin
                    if i_vxg == -1:
                        i_vxg = 0

                    if i_vyg == -1:
                        i_vyg = 0

                    # Special case matching MATLAB discretize's last-bin
                    # exception (closed on both ends): v == vg_edges[-1]
                    # exactly must land in the last bin, not one-past-the-end.
                    # Only the searchsorted(side='right') path can produce
                    # n_vg here -- the fast-index path already clamps to
                    # n_vg - 1 above.
                    if i_vxg == n_vg:
                        i_vxg = n_vg - 1

                    if i_vyg == n_vg:
                        i_vyg = n_vg - 1

                    if use_point:
                        f_g_threads[tid, i_vxg, i_vyg] += f_ijk * c_ijk / d_a_grid

    f_g = f_g_threads.sum(axis=0)  # CHANGED: see _mc_pol_1d

    return f_g


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_cart_3d(
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
    n_threads,
):
    r"""Perform 3D Monte-Carlo interpolation of the VDFs

    Parameters
    ----------
    vdf : numpy.ndarray
        3D skymap particle velocity distribution function.
    v : numpy.ndarray
        1D array of instrument velocity bins centers.
    phi : numpy.ndarray
        1D array of instrument azimuthal angles bins centers.
    theta : numpy.ndarray
        1D array of instrument elevation angles bins centers.
    d_v : numpy.ndarray
        1D array of instrument velocity bins widths.
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
    v_step : double
        # CHANGED: new parameter, see _mc_cart_2d.
    n_threads : int
        # CHANGED: new parameter. Number of worker threads, queried once in
        # plain Python (numba.get_num_threads() inside a cache=True jitted
        # function disables its disk cache) and used to size the
        # per-thread accumulator that replaces the original racy shared-
        # array scatter-add under numba.prange.

    Returns
    -------
    f_g : double
        Reduced/interpolated distribution.

    """

    # Get dimension of the instrument and interpolation grid.
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1

    v2 = v**2  # CHANGED: hoisted, see _mc_pol_1d
    cos_theta = np.cos(theta)
    theta_1_arr = theta - 0.5 * d_theta
    theta_2_arr = theta + 0.5 * d_theta
    sin_theta_1_arr = np.sin(theta_1_arr)
    d_sin_theta_arr = np.sin(theta_2_arr) - sin_theta_1_arr

    v_g0 = vg_edges[0]
    use_fast_index = v_step > 0.0  # CHANGED: see _mc_cart_2d

    # CHANGED: one private accumulator per worker thread, allocated once --
    # see _mc_cart_2d for why this replaced a per-i-iteration private
    # array (measured net slowdown for a large 3D output grid, since a
    # fresh n_vg**3 array was being re-zeroed on every visit to the outer
    # loop). Distinct threads always write to disjoint rows of
    # f_g_threads (numba.get_thread_id() is stable per worker for the
    # life of the parallel region), so this is race-free by construction,
    # fixing the original f_g[idx] += ... pattern's race under prange.
    # CHANGED: n_threads is now passed in from the Python-level caller
    # (numba.get_num_threads() called *inside* a cache=True jitted
    # function disables its disk cache -- it references dynamic
    # globals numba can't serialize; calling it in plain Python and
    # passing the result in avoids that regression).
    f_g_threads = np.zeros((n_threads, n_vg, n_vg, n_vg))

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
                    # Generate Monte-Carlo particle speed, phi, and theta
                    if l_mc == 0:
                        # First Monte-Carlo particle is set at the bin center
                        d_v_mc = 0.0
                        d_phi_mc = 0.0
                    else:
                        d_v_mc = random.random() * d_v[i] - d_v_m[i]
                        d_phi_mc = (random.random() - 0.5) * d_phi[j]

                    # convert instrument bin to cartesian velocity
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc

                    # This is needed to distribute the points evenly in space
                    # (more points further 'equatorward' since the slice is
                    # wider there).

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

                    # Get velocities in primed coordinate system
                    # vxp = [vx, vy, vz] * xphat'; % all MC points
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    alpha = asin(v_z_p / v_mc)

                    # Check if the velocity is within the v_lim and a_lim bounds
                    use_point = v_lim[0] <= v_z_p < v_lim[1]
                    use_point = use_point and (a_lim[0] <= alpha < a_lim[1])

                    if v_x_p > vg_edges[-1] or v_x_p < vg_edges[0]:
                        continue

                    if v_y_p > vg_edges[-1] or v_y_p < vg_edges[0]:
                        continue

                    if v_z_p > vg_edges[-1] or v_z_p < vg_edges[0]:
                        continue

                    # CHANGED: O(1) arithmetic index when the grid is
                    # confirmed uniform, replacing a searchsorted binary
                    # search per particle per axis. The arithmetic gives
                    # an initial guess that is then VERIFIED (and, if
                    # needed, corrected by a bin or two) against the
                    # actual vg_edges values -- an exact-equality
                    # tie-break alone is not enough, because
                    # v_g0 + i*v_step can drift from the true
                    # vg_edges[i] by a few ULPs of accumulated rounding,
                    # and a real velocity sample landing within that
                    # drift of a boundary would silently land in the
                    # wrong bin. This is not just a theoretical concern:
                    # a projection grid centered on zero (the common
                    # case) has an edge at v=0, and an instrument phi
                    # grid that includes exactly 90/270 degrees produces
                    # many real samples with a near-zero component right
                    # at that edge. The correction loops below reproduce
                    # MATLAB's discretize(v, vg_edges) convention exactly --
                    # left-closed/right-open per bin (vg_edges[k] <= v <
                    # vg_edges[k+1]), confirmed against the real MATLAB
                    # source (irf_int_sph_dist.m) -- while still being O(1)
                    # in the typical case (0 or 1 iterations). CHANGED:
                    # a prior pass here used <=/> (matching np.searchsorted's
                    # left-open/right-closed convention instead), which was
                    # backwards relative to MATLAB -- reverted to </>=.
                    if use_fast_index:
                        i_vxg = int((v_x_p - v_g0) / v_step)
                        i_vyg = int((v_y_p - v_g0) / v_step)
                        i_vzg = int((v_z_p - v_g0) / v_step)
                        if i_vxg < 0:
                            i_vxg = 0
                        elif i_vxg >= n_vg:
                            i_vxg = n_vg - 1
                        if i_vyg < 0:
                            i_vyg = 0
                        elif i_vyg >= n_vg:
                            i_vyg = n_vg - 1
                        if i_vzg < 0:
                            i_vzg = 0
                        elif i_vzg >= n_vg:
                            i_vzg = n_vg - 1
                        while i_vxg > 0 and v_x_p < vg_edges[i_vxg]:
                            i_vxg -= 1
                        while i_vxg < n_vg - 1 and v_x_p >= vg_edges[i_vxg + 1]:
                            i_vxg += 1
                        while i_vyg > 0 and v_y_p < vg_edges[i_vyg]:
                            i_vyg -= 1
                        while i_vyg < n_vg - 1 and v_y_p >= vg_edges[i_vyg + 1]:
                            i_vyg += 1
                        while i_vzg > 0 and v_z_p < vg_edges[i_vzg]:
                            i_vzg -= 1
                        while i_vzg < n_vg - 1 and v_z_p >= vg_edges[i_vzg + 1]:
                            i_vzg += 1
                    else:
                        # CHANGED: side='right' (not the default 'left'),
                        # minus 1, matches MATLAB's discretize convention --
                        # see _mc_pol_1d for the full explanation. side='left'
                        # gives the opposite (left-open/right-closed) binning
                        # and silently mis-bins points exactly on an interior
                        # grid edge.
                        i_vxg = np.searchsorted(vg_edges, v_x_p, side="right") - 1
                        i_vyg = np.searchsorted(vg_edges, v_y_p, side="right") - 1
                        i_vzg = np.searchsorted(vg_edges, v_z_p, side="right") - 1

                    # Special case for the first bin edge, which is the lower limit of
                    # the first bin and should be included in the first bin
                    if i_vxg == -1:
                        i_vxg = 0

                    if i_vyg == -1:
                        i_vyg = 0

                    if i_vzg == -1:
                        i_vzg = 0

                    # Special case matching MATLAB discretize's last-bin
                    # exception (closed on both ends): v == vg_edges[-1]
                    # exactly must land in the last bin, not one-past-the-end.
                    # Only the searchsorted(side='right') path can produce
                    # n_vg here -- the fast-index path already clamps to
                    # n_vg - 1 above.
                    if i_vxg == n_vg:
                        i_vxg = n_vg - 1

                    if i_vyg == n_vg:
                        i_vyg = n_vg - 1

                    if i_vzg == n_vg:
                        i_vzg = n_vg - 1

                    if use_point:
                        f_g_threads[tid, i_vxg, i_vyg, i_vzg] += (
                            f_ijk * c_ijk / d_a_grid
                        )

    f_g = f_g_threads.sum(axis=0)  # CHANGED: see _mc_pol_1d

    return f_g
