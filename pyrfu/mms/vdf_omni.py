#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.ts_spectr import ts_spectr
from ._vdf_bins import (
    _band_solid_angles,
    _mass,
    _phi_edges,
    _relativistic,
    _speed,
    _speed_edges,
    _theta_edges,
)
from ._vdf_mc import _rng
from .vdf_shift import _frame_velocity

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _bin_index(lower, upper, values):
    # Index of the bin [lower, upper) of each value, -1 outside the bins
    order = np.argsort(lower)
    index = np.searchsorted(lower[order], values, side="right") - 1
    index = np.where(index >= 0, order[np.clip(index, 0, None)], -1)
    inside = (index >= 0) & (values < upper[index])
    return np.where(inside, index, -1)


def _omni_frame(vdf, v_frame, energy, n_mc, random_state):
    # Spherical means of the skymap on spheres centred on v_frame, of radii
    # the speeds of the energies, as irfu-matlab get_omni_dist_framedep
    n_t = len(vdf.time)
    mass, relativistic = _mass(vdf), _relativistic(vdf, None)
    v_frame = _frame_velocity(vdf, v_frame)

    if energy is None:
        energy = np.broadcast_to(vdf.energy.data, vdf.data.shape[:2])
    else:
        energy = np.asarray(energy, dtype=np.float64)

    speed = _speed(np.broadcast_to(energy, (n_t, energy.shape[-1])), mass, relativistic)

    # Points uniform on the unit sphere (Archimedes: z and phi uniform)
    rng = _rng(random_state)
    u_z = rng.uniform(-1.0, 1.0, n_mc)
    u_phi = rng.uniform(0.0, 2.0 * np.pi, n_mc)
    u_xyz = np.stack(
        [np.sqrt(1 - u_z**2) * np.cos(u_phi), np.sqrt(1 - u_z**2) * np.sin(u_phi), u_z],
        axis=-1,
    )

    v_lower, v_upper = _speed_edges(vdf, None, relativistic)
    phi_lower, phi_upper = _phi_edges(vdf)
    theta_lower, theta_upper = _theta_edges(vdf)
    data = vdf.data.data

    omni = np.zeros(speed.shape)
    coverage = np.zeros(speed.shape)

    for i_t in range(n_t):
        # Look directions (minus the particle velocities in the spacecraft
        # frame) of the points
        look = speed[i_t, :, np.newaxis, np.newaxis] * u_xyz[np.newaxis]
        look = -(look + v_frame[i_t])
        norm = np.linalg.norm(look, axis=-1)

        with np.errstate(invalid="ignore", divide="ignore"):
            theta = np.rad2deg(np.arccos(np.clip(look[..., 2] / norm, -1, 1)))

        phi = np.rad2deg(np.arctan2(look[..., 1], look[..., 0]))
        phi = phi_lower[i_t].min() + np.mod(phi - phi_lower[i_t].min(), 360.0)

        i_e = _bin_index(v_lower[i_t], v_upper[i_t], norm)
        i_p = _bin_index(phi_lower[i_t], phi_upper[i_t], phi)
        i_q = _bin_index(theta_lower, theta_upper, theta)
        valid = (i_e >= 0) & (i_p >= 0) & (i_q >= 0)
        values = np.where(valid, data[i_t, i_e, i_p, i_q], np.nan)
        valid &= np.isfinite(values)

        # Points outside the bins or in bins without data count as zero
        omni[i_t] = np.sum(np.where(valid, values, 0.0), axis=1) / n_mc
        coverage[i_t] = np.sum(valid, axis=1) / n_mc

    return omni, energy, coverage, v_frame / 1e3


def _omni_spacecraft(vdf, method):
    # Solid-angle weighted mean or sum over the angles (time, energy)
    data = vdf.data.data

    with warnings.catch_warnings():
        # Channels without data at all angles give NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method.lower() == "mean":
            # Solid angles of the (phi, theta) bins (time, phi, theta), from
            # the angular widths of the attributes when present
            solid_angles = _band_solid_angles(vdf)[:, np.newaxis]
            omni = np.nanmean(np.nanmean(data * solid_angles, axis=3), axis=2)
            return omni / np.mean(solid_angles, axis=(2, 3))

        return np.nansum(np.nansum(data, axis=3), axis=2)


def vdf_omni(
    vdf,
    method: str = "mean",
    v_frame=None,
    energy=None,
    n_mc: int = 5000,
    random_state=None,
):
    r"""Computes omni-directional distribution, without changing the units.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the 3D velocity distribution with :
            * time : Time samples.
            * data : 3D velocity distribution.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    method : {"mean", "sum"}, Optional
        Method of computation. Use "sum" for counts and "mean" for
        everything else. Default is "mean".
    v_frame : array_like or xarray.DataArray, Optional
        Velocity of the frame (km/s) in the coordinate system of `vdf`: a
        3-vector or a time series, resampled to the times of `vdf`. The
        omni-directional distribution is then that of this frame, as
        irfu-matlab PDist.omni(V) ("mean" only). Default is the spacecraft
        frame.
    energy : array_like, Optional
        With `v_frame`, the energies (eV, in the new frame) of the
        omni-directional distribution. Default is the energies of `vdf`.
    n_mc : int, Optional
        With `v_frame`, number of Monte-Carlo points per energy. Default is
        5000, as irfu-matlab.
    random_state : int or numpy.random.Generator, Optional
        With `v_frame`, seed or generator of the Monte-Carlo points.

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the omnidirectional velocity distribution function: a
        DataArray (time, energy) with one energy table, or a Dataset with
        (time, energy) energies when the energy tables alternate. Both keep
        the units and the attributes of vdf (UNITS on the data variable of
        the Dataset, as for vdf).

    Raises
    ------
    ValueError
        If method is not "mean" or "sum", or is "sum" with `v_frame`.

    Notes
    -----
    As irfu-matlab PDist.omni, "mean" averages the distribution weighted by
    the solid angles over theta, then phi, and divides by the mean solid
    angle; without NaNs, this is the solid-angle weighted mean. The solid
    angles are those of `vdf_solid_angle`: from the angular widths of the
    attributes when present (e.g. the polar bands of different widths of the
    FEEPS skymaps), else from the spacing of the angles. A channel without
    data at all angles gives NaN ("mean") or 0 ("sum").

    With `v_frame`, the distribution at speed v in the new frame is the
    spherical mean of `vdf` on the sphere of radius v centred on `v_frame`,
    as irfu-matlab get_omni_dist_framedep: `n_mc` points uniform on the
    sphere (the same at all times), each taking the value of the bin it falls
    in. The points outside the energy range of `vdf` or in bins without data
    count as zero, so that the distribution is a lower limit where the
    ``coverage`` attribute (fraction of the points in bins with data, time x
    energy) is below 1. The bin edges are those of `vdf_solid_angle` and
    `vdf_d3v`; the speeds are relativistic if the ``relativistic`` attribute
    of `vdf` is True. The energies of each time are used (irfu-matlab uses
    those of the first time).

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> v_i = mms.get_data("vi_dbcs_fpi_brst_l2", tint, 1)
    >>> omni_i_rest = mms.vdf_omni(vdf_i, v_frame=v_i, random_state=0)

    """

    if method.lower() not in ["mean", "sum"]:
        raise ValueError(f"Invalid method {method!r}, use mean or sum")

    time = vdf.time.data

    frame_attrs = {}

    if v_frame is not None:
        if method.lower() != "mean":
            raise ValueError("v_frame needs method mean")

        omni, energy, coverage, v_frame = _omni_frame(
            vdf, v_frame, energy, n_mc, random_state
        )
        frame_attrs = {"frame_velocity": v_frame, "coverage": coverage, "n_mc": n_mc}
    else:
        omni, energy = _omni_spacecraft(vdf, method), vdf.energy.data

    # Use global and zVariable attributes
    attrs = {**vdf.data.attrs, **vdf.attrs}
    attrs = {k: attrs[k] for k in sorted(attrs)}

    out = ts_spectr(time, energy, omni, attrs=attrs)

    # Time varying energies (alternating tables): units on the data variable
    # and the other attributes global, as for vdf
    if isinstance(out, xr.Dataset):
        out.data.attrs = dict(vdf.data.attrs)
        out.attrs = dict(vdf.attrs)

    out.attrs.update(frame_attrs)
    return out
