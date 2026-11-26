#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
from math import asin, atan2, cos, pi, sin, sqrt
from typing import Literal

# 3rd party imports
import numba
import numpy as np

# Implementation notes (numba kernel _mc_project):
# - one accumulator per chunk of speeds (speed i in chunk i % n_chunks, one
#   chunk per worker thread), summed in order at the end: a shared
#   accumulator under numba.prange loses updates when two threads add to the
#   same bin, and accumulators per thread (numba.get_thread_id) sum the speeds
#   in an order that depends on the scheduling; the number of threads is
#   passed in from Python (numba.get_num_threads() inside a cache=True
#   function disables its disk cache);
# - with a seed, numba's generator (random.seed has no effect on it from
#   Python) is seeded at each speed with the seed of that speed, so that the
#   particles of a speed do not depend on the thread that draws them: the
#   same seed gives the same result for the same number of threads;
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
    weight: Literal["lin", "log"] | None = None,
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
    seed: int | None = None,
):
    r"""Integrate a spherical distribution function to a line/plane.

    Parameters
    ----------
    vdf : numpy.ndarray
        Phase-space density skymap. Non-finite values (NaN) have no data and
        add nothing; without any finite value, the projection is NaN (no
        data), not 0.
    velocity : numpy.ndarray
        Velocity of the instrument bins,
    phi : numpy.ndarray
        Azimuthal angle of the instrument bins.
    theta : numpy.ndarray
        Elevation angle of the instrument bins.
    velocity_grid : numpy.ndarray
        Velocity grid for interpolation.
    phi_grid : numpy.ndarray
        Azimuths (rad) of the centres of the bins of the 2d polar grid, from
        x in the projection plane, equally spaced over at most a turn (e.g.
        32 bins centred on (k + 0.5) 2 pi / 32); not used by the other
        projections.

    Returns
    -------
    pst : dict
        Projected distribution ``f``, on the speeds of the grid ``vx`` and
        their edges ``vx_edges`` (1d), and ``vy``, ``vz`` (2d and 3d
        cartesian, the same grid on each axis). For the 2d polar projection,
        ``f`` is (azimuth, speed), on the azimuths ``phi`` (rad) and speeds
        ``v`` of the bins and their edges ``phi_edges`` and ``v_edges``, with
        the velocities ``vx``, ``vy`` of the centres of the bins (2d).

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
        cartesian projections. The 1d projection is the same in both bases, a
        cartesian grid being uniform.
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
        the median spacing of `phi` and `theta` (required with a single bin,
        which has no spacing).
    seed : int
        Seed of the Monte-Carlo particles (a non-negative integer): the same
        seed gives the same result (bit for bit with the same number of numba
        threads). Default is None, a different draw at each call
        (``random.seed`` and ``numpy.random.seed`` do not reach the numba
        generator).

    Raises
    ------
    ValueError
        If the speed widths are negative or extend below zero speed, if only
        one of `d_v_m` and `d_v_p` is given, if `weight` is not one of the
        above, if a cartesian projection grid is not uniform, or if `phi` or
        `theta` has a single bin without `d_phi` or `d_theta`, if `seed` is
        not a non-negative integer, or if the `phi_grid` of a 2d polar
        projection is not equally spaced or spans more than a turn.
    NotImplementedError
        If the projection is 3d polar.

    Notes
    -----
    The projected distribution of a bin is the integral of the distribution
    over the bin divided by its length (1d), its area (2d) or its volume
    (3d): the area of a bin of the 2d polar grid is (v_2^2 - v_1^2) d_phi / 2
    between its edges v_1 and v_2 (the speeds below 0 not counted), so that
    the density is the sum of f times the areas.

    The Monte-Carlo particles are drawn uniformly within each speed bin. This
    differs from irfu-matlab's irf_int_sph_dist, which draws them in
    [v - 1.5 dV, v - 0.5 dV] (one bin too low) and uses the spacing to the lower
    neighbour as the bin width when no edges are given. For a drifting
    Maxwellian on FPI-like bins, that biases the reduced distribution to lower
    speeds (bulk speed ~7 % and temperature ~10 % too low, density ~6 % too low).

    """

    # Non-finite values have no data (NaNs would spread to the projection);
    # without data, the projection is NaN
    no_data = not np.any(np.isfinite(vdf))
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

    # Check the weighting of the number of Monte Carlo particles
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
    # if not provided. A single bin has no spacing to take its width from
    for name, angles, widths in [("phi", phi, d_phi), ("theta", theta, d_theta)]:
        if widths is None and np.size(angles) < 2:
            raise ValueError(f"{name} has a single bin: give its width d_{name}")

    if d_phi is None:
        d_phi = np.abs(np.median(np.diff(phi))) * np.ones_like(phi)

    if d_theta is None:
        d_theta = np.abs(np.median(np.diff(theta))) * np.ones_like(theta)

    # Speed widths below (d_v_m) and above (d_v_p) the speed of each bin: the
    # Monte-Carlo particles are drawn uniformly in [v - d_v_m, v + d_v_p]
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
    # plane; the supported projections are 1d polar, 2d and 3d cartesian
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

    # Azimuths of the 2d polar grid (empty for the other grids)
    phig_edges, phi_step = np.zeros(0), 0.0

    if projection_base == "pol":
        d_v_grid = np.diff(velocity_grid_edges)

        if n_dim == 2:
            phig_edges, phi_step = _azimuth_edges(phi_grid)
    else:
        # All bins must have the same width (to 1 %), which is used for all
        d_v_grid = np.diff(velocity_grid_edges)
        mean_diff = np.mean(d_v_grid)

        if not np.all(np.abs(d_v_grid / mean_diff - 1) < 1e-2):
            raise ValueError("For a cartesian grid, all velocity bins must be equal!!")

        d_v_grid = mean_diff

    # Weighting of number of Monte Carlo particles
    n_mc_mat = _mc_particles(vdf, n_mc, weight)

    # One seed per speed (none: an empty array, numba's generator as it is)
    speed_seeds = _speed_seeds(seed, len(velocity))

    # Element of the projection grid: the speed width of each bin of the 1d
    # grids, the area of the bins of each speed of the 2d polar grid, the
    # volume of the bins of the cartesian grids
    if n_dim == 1:
        d_a_grid = np.broadcast_to(d_v_grid, (len(velocity_grid),))
        d_a_grid = d_a_grid.astype(np.float64)
    elif projection_base == "pol":
        v_edges_2 = np.clip(velocity_grid_edges, 0.0, None) ** 2
        d_a_grid = 0.5 * np.diff(v_edges_2) * phi_step
    else:
        d_a_grid = np.full(1, d_v_grid**n_dim, dtype=np.float64)

    # The grid indices computed on a cartesian grid uniform to near machine
    # precision, searched otherwise (see the implementation notes)
    if n_dim > 1:
        v_step = _uniform_step(velocity_grid_edges)
    else:
        v_step = 0.0

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
        speed_seeds,
        numba.get_num_threads(),
        phig_edges,
        phi_step,
    )

    if len(phig_edges) > 0:
        f_g = f_g.reshape((len(phig_edges) - 1, len(velocity_grid)))
    else:
        f_g = f_g.reshape((len(velocity_grid),) * n_dim)

    if no_data:
        f_g = np.full(f_g.shape, np.nan)

    if projection_dim == "2d" and projection_base == "pol":
        phig_centres = phig_edges[:-1] + phi_step / 2.0
        v_mesh, phi_mesh = np.meshgrid(velocity_grid, phig_centres)
        pst = {
            "f": f_g,
            "phi": phig_centres,
            "v": velocity_grid,
            "phi_edges": phig_edges,
            "v_edges": velocity_grid_edges,
            "vx": v_mesh * np.cos(phi_mesh),
            "vy": v_mesh * np.sin(phi_mesh),
        }
    elif projection_dim == "2d" and projection_base == "cart":
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
_PROJECTIONS = [
    ("pol", "1d"),
    ("pol", "2d"),
    ("cart", "1d"),
    ("cart", "2d"),
    ("cart", "3d"),
]


def _azimuth_edges(phi_grid):
    r"""Edges (rad) and width of the azimuths of a 2d polar grid from the
    centres of its bins (a single bin is a turn)."""

    phi_grid = np.atleast_1d(np.asarray(phi_grid, dtype=np.float64))

    if len(phi_grid) < 2:
        d_phi = 2 * np.pi
    else:
        d_phi = float(np.median(np.diff(phi_grid)))

    if d_phi <= 0 or not np.allclose(np.diff(phi_grid), d_phi, rtol=1e-6, atol=0):
        raise ValueError("phi_grid must be equally spaced and increasing")

    if len(phi_grid) * d_phi > 2 * np.pi * (1 + 1e-9):
        raise ValueError("phi_grid must span at most a turn")

    edges = np.hstack([phi_grid - d_phi / 2, phi_grid[-1] + d_phi / 2])

    return edges, d_phi


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


def _speed_seeds(seed, n_v):
    r"""Seeds of the Monte-Carlo particles of the n_v speeds from `seed`
    (int64, from numpy's SeedSequence), or an empty array without seed."""

    if seed is None:
        return np.zeros(0, dtype=np.int64)

    is_int = isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)

    if not is_int or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, not {seed!r}")

    return np.random.SeedSequence(int(seed)).generate_state(n_v).astype(np.int64)


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
    speed_seeds,
    n_chunks,
    phig_edges,
    phi_step,
):
    r"""Monte-Carlo projection of a skymap onto a grid, flattened: 1d (n_dim
    = 1: the speed along the first axis, the transverse speed limited by
    v_lim and a_lim), 2d polar (n_dim = 2 with `phig_edges`: the azimuth and
    the speed in the plane), 2d or 3d cartesian (the out-of-plane speed
    limited by v_lim and a_lim).

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
        Edges of the projection grid (the same on each axis; the speeds of
        the 2d polar grid).
    d_a_grid : numpy.ndarray
        Speed widths of the 1d grid, areas of the bins of each speed of the
        2d polar grid, or the volume of a bin of the cartesian grids (one
        value).
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
    speed_seeds : numpy.ndarray
        Seed of the particles of each speed, or empty (no seeding).
    n_chunks : int
        Number of chunks of speeds (the number of worker threads).
    phig_edges : numpy.ndarray
        Edges of the azimuths of the 2d polar grid (rad, spanning at most a
        turn), or empty for the other grids.
    phi_step : float
        Width of the azimuths of the 2d polar grid (index arithmetic), or 0.0.

    Returns
    -------
    numpy.ndarray
        Projected distribution, (n_g ** n_dim,) with n_g bins on each axis,
        or (n_phi * n_g,) for the 2d polar grid (azimuth, speed).

    """

    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    polar = len(phig_edges) > 0

    if polar:
        n_out = (len(phig_edges) - 1) * n_vg
    else:
        n_out = n_vg**n_dim

    # Invariants of the speeds and elevations
    v2 = v**2
    cos_theta = np.cos(theta)
    sin_theta_1_arr = np.sin(theta - 0.5 * d_theta)
    d_sin_theta_arr = np.sin(theta + 0.5 * d_theta) - sin_theta_1_arr

    f_g_chunks = np.zeros((n_chunks, n_out))

    for c in numba.prange(n_chunks):
        for i in range(c, n_v, n_chunks):
            if len(speed_seeds) > 0:
                random.seed(int(speed_seeds[i]))

            for j in range(n_ph):
                for k in range(n_th):
                    n_mc_ijk = n_mc[i, j, k]

                    if vdf[i][j][k] == 0.0:
                        continue

                    dtau_ijk = v2[i] * cos_theta[k] * d_v[i] * d_phi[j] * d_theta[k]
                    c_ijk = dtau_ijk / n_mc_ijk
                    f_ijk = vdf[i, j, k]

                    for l_mc in range(n_mc_ijk):
                        # Speed and azimuth, the first particle at the centre
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
                                sin_theta_1_arr[k]
                                + random.random() * d_sin_theta_arr[k]
                            )
                            theta_mc = asin(sin_theta_mc)

                        v_x = v_mc * cos(theta_mc) * cos(phi_mc)
                        v_y = v_mc * cos(theta_mc) * sin(phi_mc)
                        v_z = v_mc * sin(theta_mc)

                        # Velocity in the primed coordinate system
                        v_x_p = (
                            r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                        )
                        v_y_p = (
                            r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                        )
                        v_z_p = (
                            r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z
                        )

                        # Transverse (1d) or out-of-plane speed, and the angle
                        # from the projection axis or plane, within limits
                        if n_dim == 1:
                            v_z_p = sqrt(v_y_p**2 + v_z_p**2)

                        alpha = asin(v_z_p / v_mc)

                        use_point = v_lim[0] <= v_z_p < v_lim[1]
                        use_point = use_point and (a_lim[0] <= alpha < a_lim[1])

                        if polar:
                            # Azimuth in the turn from the first edge of the
                            # grid, and speed in the plane
                            phi_p = atan2(v_y_p, v_x_p) - phig_edges[0]
                            phi_p = phig_edges[0] + phi_p % (2 * pi)
                            v_p = sqrt(v_x_p**2 + v_y_p**2)

                            # Particles out of the grid
                            if v_p > vg_edges[-1] or v_p < vg_edges[0]:
                                continue

                            if phi_p > phig_edges[-1] or phi_p < phig_edges[0]:
                                continue

                            # Flat index of the bin (azimuth, speed), C order
                            i_vg = _grid_index(v_p, vg_edges, v_step)
                            i_phig = _grid_index(phi_p, phig_edges, phi_step)
                            idx = i_phig * n_vg + i_vg
                            d_a = d_a_grid[i_vg]
                        else:
                            # Particles out of the grid
                            if v_x_p > vg_edges[-1] or v_x_p < vg_edges[0]:
                                continue

                            if n_dim > 1 and (
                                v_y_p > vg_edges[-1] or v_y_p < vg_edges[0]
                            ):
                                continue

                            if n_dim > 2 and (
                                v_z_p > vg_edges[-1] or v_z_p < vg_edges[0]
                            ):
                                continue

                            # Flat index of the bin (x, y, z), C order
                            i_vxg = _grid_index(v_x_p, vg_edges, v_step)
                            idx = i_vxg

                            if n_dim > 1:
                                idx = idx * n_vg + _grid_index(v_y_p, vg_edges, v_step)

                            if n_dim > 2:
                                idx = idx * n_vg + _grid_index(v_z_p, vg_edges, v_step)

                            if n_dim == 1:
                                d_a = d_a_grid[i_vxg]
                            else:
                                d_a = d_a_grid[0]

                        if use_point:
                            f_g_chunks[c, idx] += f_ijk * c_ijk / d_a

    return f_g_chunks.sum(axis=0)
