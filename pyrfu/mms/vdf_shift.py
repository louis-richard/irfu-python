#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.resample import resample
from ..pyrf.ts_skymap import ts_skymap
from ._psd_units import _attrs
from ._vdf_bins import _is_skymap, _mass, _relativistic, _speed
from ._vdf_mc import _bins_d3v, _rng, _sample, _source_bins, _speed_to_energy
from .vdf_convert import vdf_convert
from .vdf_velocity import _rotation_matrices

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.1"
__status__ = "Prototype"


def _frame_velocity(vdf, v_frame):
    # Frame velocity (m/s) at the times of the distribution (time, 3)
    if isinstance(v_frame, xr.DataArray):
        return resample(v_frame, vdf.time).data * 1e3

    v_frame = np.asarray(v_frame, dtype=np.float64)

    if v_frame.shape != (3,):
        raise ValueError("v_frame must be a 3-vector or a time series of vectors")

    return np.tile(v_frame * 1e3, (len(vdf.time), 1))


def _grid_edges(grid, name, default):
    # Edges of the new grid: given (grid[name]) or the default
    if grid is None or grid.get(name) is None:
        return default

    return np.asarray(grid[name], dtype=np.float64)


def _default_speed_edges(bins, i_t, v_frame):
    # Speed edges of the old grid, the first and last widened to hold the
    # shifted bin centres, as irfu-matlab PDist.shift
    v_edges = np.hstack([bins["v_lower"][i_t], bins["v_upper"][i_t, -1]])
    centre = (bins["v_lower"][i_t] + bins["v_upper"][i_t]) / 2
    phi = np.deg2rad((bins["phi_lower"][i_t] + bins["phi_upper"][i_t]) / 2)
    cos_t = (bins["cos_lower"] + bins["cos_upper"]) / 2
    sin_t = np.sqrt(1 - cos_t**2)
    look = np.stack(
        np.broadcast_arrays(
            sin_t[None, :] * np.cos(phi)[:, None],
            sin_t[None, :] * np.sin(phi)[:, None],
            cos_t[None, :],
        ),
        axis=-1,
    )
    shifted = -centre[:, None, None, None] * look[None] - v_frame
    speed = np.linalg.norm(shifted, axis=-1)
    d_v = np.diff(v_edges)
    v_edges[0] = max(min(v_edges[0], speed.min() - d_v[0] / 2), 0.0)
    v_edges[-1] = max(v_edges[-1], speed.max() + d_v[-1] / 2)
    return v_edges


def _rebin(points, weights, v_edges, phi_edges, theta_edges):
    # Sum of the weights of the points (look directions, n x 3) in the new
    # bins (energy, phi, theta), and the sum of those outside
    speed = np.linalg.norm(points, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.rad2deg(np.arccos(np.clip(points[:, 2] / speed, -1.0, 1.0)))

    phi = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))
    phi = phi_edges[0] + np.mod(phi - phi_edges[0], 360.0)

    shape = (len(v_edges) - 1, len(phi_edges) - 1, len(theta_edges) - 1)
    index = [
        np.searchsorted(edges, values, side="right") - 1
        for edges, values in zip([v_edges, phi_edges, theta_edges], [speed, phi, theta])
    ]
    inside = np.all([(i >= 0) & (i < n) for i, n in zip(index, shape)], axis=0)
    flat = np.ravel_multi_index([i[inside] for i in index], shape)
    summed = np.bincount(flat, weights[inside], minlength=np.prod(shape))
    return summed.reshape(shape), np.sum(weights[~inside])


def _shift(
    vdf, v_frame, rotation, n_mc, grid, random_state, sc_pot=None, phi_per_time=False
):
    # Monte-Carlo rebinning of the skymap in the frame moving with v_frame and
    # the rotated coordinates (vdf_shift and vdf_rebin)
    relativistic = _relativistic(vdf, None)
    units = vdf.data.attrs["UNITS"]
    f_si = np.nan_to_num(vdf_convert(vdf, "s^3/m^6").data.data)
    v_frame = _frame_velocity(vdf, v_frame)
    rotations = _rotation_matrices(rotation, len(vdf.time))
    mass, rng = _mass(vdf), _rng(random_state)

    bins = _source_bins(vdf, sc_pot)
    d3v = _bins_d3v(bins)
    energy_grid = _grid_edges(grid, "energy", None)

    # Azimuth edges of each time: given, those of the first time, or those of
    # each time (phi_per_time)
    phi_edges = [
        _grid_edges(
            grid,
            "phi",
            np.hstack([bins["phi_lower"][i], bins["phi_upper"][i, -1]]),
        )
        for i in (range(len(vdf.time)) if phi_per_time else [0] * len(vdf.time))
    ]
    theta_edges = _grid_edges(
        grid,
        "theta",
        np.rad2deg(np.arccos(np.hstack([bins["cos_upper"], bins["cos_lower"][-1]]))),
    )

    out_f, energy_edges, lost = [], [], np.zeros(len(vdf.time))

    for i_t in range(len(vdf.time)):
        if energy_grid is not None:
            v_edges = _speed(energy_grid, mass, relativistic)
        else:
            v_edges = _default_speed_edges(bins, i_t, v_frame[i_t])

        f_d3v = f_si[i_t] * d3v[i_t]
        i_e, i_p, i_q = np.nonzero(f_d3v)
        i_e, i_p, i_q = (np.repeat(i, n_mc) for i in [i_e, i_p, i_q])
        velocity = _sample(rng, bins, np.full(len(i_e), i_t), i_e, i_p, i_q)

        # Look directions of the particles in the new frame and coordinates
        points = -((velocity - v_frame[i_t]) @ rotations[i_t].T)
        summed, lost_t = _rebin(
            points, f_d3v[i_e, i_p, i_q] / n_mc, v_edges, phi_edges[i_t], theta_edges
        )

        new_d3v = (
            np.diff(v_edges**3)[:, None, None]
            / 3
            * np.deg2rad(np.diff(phi_edges[i_t]))[None, :, None]
            * -np.diff(np.cos(np.deg2rad(theta_edges)))[None, None, :]
        )
        out_f.append(summed / new_d3v)
        energy_edges.append(_speed_to_energy(v_edges, mass, relativistic))
        total = np.sum(f_d3v)
        lost[i_t] = lost_t / total if total > 0 else 0.0

    energy_edges = np.array(energy_edges)
    # Geometric centres of the energy bins (those of FPI, so that the bins of
    # an unshifted skymap are unchanged), arithmetic from 0 eV
    lower, upper = energy_edges[:, :-1], energy_edges[:, 1:]
    energy = np.where(lower > 0, np.sqrt(lower * upper), (lower + upper) / 2)
    phi_edges = np.array(phi_edges)
    phi = (phi_edges[:, :-1] + phi_edges[:, 1:]) / 2
    theta = (theta_edges[:-1] + theta_edges[1:]) / 2
    n_t = len(vdf.time)

    out = ts_skymap(
        vdf.time.data,
        np.array(out_f),
        energy,
        phi,
        theta,
        energy0=energy[0],
        energy1=energy[0],
        esteptable=np.zeros(n_t, dtype=np.uint8),
        attrs={**vdf.data.attrs, "UNITS": "s^3/m^6"},
        coords_attrs={
            k: dict(vdf[k].attrs) for k in ["time", "energy", "phi", "theta"]
        },
        glob_attrs={
            **{
                k: v
                for k, v in vdf.attrs.items()
                if not k.startswith(("delta_", "energy", "esteptable", "coverage"))
                or k.startswith("delta_time")
            },
            "delta_energy_minus": energy - energy_edges[:, :-1],
            "delta_energy_plus": energy_edges[:, 1:] - energy,
            "delta_phi_minus": phi - phi_edges[:, :-1],
            "delta_phi_plus": phi_edges[:, 1:] - phi,
            "delta_theta_minus": theta - theta_edges[:-1],
            "delta_theta_plus": theta_edges[1:] - theta,
            "frame_velocity": v_frame / 1e3,
            "lost": lost,
        },
    )
    # Back in the units of vdf, with the attributes of its data (LABLAXIS was
    # dropped through s^3/m^6)
    out = vdf_convert(out, units)
    out.data.attrs = _attrs(vdf.data.attrs, out.data.attrs["UNITS"])

    return out


def vdf_shift(
    vdf,
    v_frame,
    rotation=None,
    n_mc: int = 100,
    grid: dict = None,
    random_state=None,
):
    r"""Skymap distribution in a frame moving with velocity `v_frame` and in
    a rotated coordinate system, as irfu-matlab PDist.shift.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta), in any units of
        `vdf_convert`.
    v_frame : array_like or xarray.DataArray
        Velocity of the new frame (km/s) in the coordinate system of `vdf`: a
        3-vector or a time series, resampled to the times of `vdf`.
    rotation : array_like, Optional
        3x3 matrix, or (time, 3, 3) matrices, whose rows are the new unit
        vectors in the coordinate system of `vdf`, e.g. [e_perp1, e_perp2, b]
        so that theta is the pitch angle of the look direction (180 degrees
        minus that of the particles). Default is no rotation.
    n_mc : int, Optional
        Number of Monte-Carlo particles per bin with data. Default is 100.
    grid : dict, Optional
        Edges of the new bins: ``energy`` (eV), ``phi`` and ``theta``
        (degrees, look directions); a missing or None entry keeps the default.
        By default the angular bins are those of `vdf` (of its first time for
        phi) and the energy bins too, with the first and last widened to hold
        the shifted distribution, as irfu-matlab.
    random_state : int or numpy.random.Generator, Optional
        Seed or generator of the Monte-Carlo particles.

    Returns
    -------
    out : xarray.Dataset
        Skymap in the new frame and coordinates, in the units of `vdf`, with
        the bin widths (``delta_energy_*``, ``delta_phi_*``,
        ``delta_theta_*``), ``frame_velocity`` (km/s) and ``lost`` (time,),
        the fraction of the density outside the new grid.

    Raises
    ------
    ValueError
        If `vdf` is not a skymap or `v_frame` or `rotation` have the wrong
        shape.

    See Also
    --------
    pyrfu.mms.vdf_macroparticles, pyrfu.pyrf.convert_fac

    Notes
    -----
    Each bin with data is sampled by `n_mc` particles, uniform in its volume
    (v^3, cos(theta) and phi), carrying f d3v / `n_mc`; their velocities are
    shifted by `v_frame` (Galilean) and rotated, and their sum in each new bin
    divided by its volume is the new phase-space density. The density is
    conserved, apart from the particles outside the new grid (``lost``). NaN
    bins count as empty. The energies of the new bins are the geometric means
    of their edges, so that an unshifted skymap is unchanged. Rebinning
    spreads each bin over its neighbours: a 1 keV Maxwellian shifted by half
    its thermal speed on the FPI grid is 1.5 % hotter. irfu-matlab draws the
    speed and the polar angle uniformly (not the volume), takes the energy
    edges halfway between the channels, the arithmetic mean of the edges as
    energies (0.8 % hotter without a shift) and keeps the energies of the last
    time only.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> v_i = mms.get_data("vi_dbcs_fpi_brst_l2", tint, 1)
    >>> vdf_i_plasma = mms.vdf_shift(vdf_i, v_i, random_state=0)

    """
    if not _is_skymap(vdf):
        raise ValueError("vdf must be a skymap")

    return _shift(vdf, v_frame, rotation, n_mc, grid, random_state)
