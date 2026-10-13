#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from ..pyrf.ts_scalar import ts_scalar
from ._vdf_bins import (
    _is_skymap,
    _mass,
    _phi_edges,
    _relativistic,
    _speed,
    _speed_edges,
)
from .psd_moments import psd_moments
from .vdf_convert import vdf_convert

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def _azimuthal_mean(values, d_phi):
    # Mean over the azimuths (axis 2) weighted by their widths, NaNs left out
    weights = np.where(np.isfinite(values), d_phi, 0.0)
    return np.nansum(values * d_phi, axis=2) / np.sum(weights, axis=2)


def _cut_moments_1d(f_cut, v_cut, d_v, mass):
    # Integral, bulk velocity (m/s) and temperature (eV) of a 1D cut
    norm = np.nansum(f_cut * d_v, axis=1)
    bulk = np.nansum(f_cut * d_v * v_cut, axis=1) / norm
    pressure = mass * np.nansum(f_cut * d_v * (v_cut - bulk[:, None]) ** 2, axis=1)
    return norm, bulk, pressure / norm / constants.elementary_charge


def _cut_moments_2d(f_cut, speed, d_v, phi, d_phi, mass):
    # Integral, bulk velocity (m/s, 2) and temperature (eV, 2x2) of a 2D cut in
    # polar coordinates (speed, look azimuth): velocity = -look
    d2v = speed[:, :, None] * d_v[:, :, None] * d_phi[:, None, :]
    v_xy = (
        -speed[:, :, None, None]
        * np.stack([np.cos(phi), np.sin(phi)], axis=-1)[:, None, :, :]
    )
    weight = f_cut * d2v
    norm = np.nansum(weight, axis=(1, 2))
    bulk = np.nansum(weight[..., None] * v_xy, axis=(1, 2)) / norm[:, None]
    d_vxy = v_xy - bulk[:, None, None, :]
    pressure = mass * np.nansum(
        weight[..., None, None] * d_vxy[..., :, None] * d_vxy[..., None, :],
        axis=(1, 2),
    )
    return norm, bulk, pressure / norm[:, None, None] / constants.elementary_charge


def vdf_cuts(vdf_fac, par_width: float = 10.0, perp_width: float = 10.0):
    r"""Parallel and perpendicular cuts of a skymap in field-aligned
    coordinates and their moments, as irfu-matlab PDist.cuts.

    Parameters
    ----------
    vdf_fac : xarray.Dataset
        Skymap distribution (time, energy, phi, theta) in field-aligned
        coordinates: x and y perpendicular to B, z along B, so that theta is
        the angle of the look direction from B, e.g. from `vdf_shift` with
        ``rotation=[e_perp1, e_perp2, b]``. Any units of `vdf_convert`.
    par_width : float, Optional
        Half-width (degrees) of the cones around B and -B of the parallel cut:
        the theta bins whose centre is within `par_width` of 0 or 180
        degrees. Default is 10, as irfu-matlab.
    perp_width : float, Optional
        Half-width (degrees) of the perpendicular cut: the theta bins whose
        centre is within `perp_width` of 90 degrees. Default is 10.

    Returns
    -------
    dists : xarray.Dataset
        Phase-space densities (s^3/m^6) of the cuts:

        * ``f_par`` (time, v_par): along B, at the signed speeds ``v_par``
          (km/s, negative antiparallel), averaged over the azimuths and the
          bins of the cones;
        * ``f_perp_2d`` (time, energy, phi): in the plane perpendicular to B,
          averaged over the bins of the plane, at the look azimuths ``phi``;
        * ``f_perp`` (time, energy): ``f_perp_2d`` averaged over the azimuths,
          at the speeds ``v_perp`` (km/s);
        * ``f_pa`` (time, energy, pitch_angle): averaged over the azimuths, at
          the pitch angles of the particles (180 degrees minus theta).

    moms : xarray.Dataset
        Moments: of the skymap (`psd_moments` with exact angular integrals),
        ``n`` (cm^-3), ``v`` (km/s) and ``t`` (eV), in field-aligned
        coordinates (perp1, perp2, par); of the parallel cut, ``n_par_cut``
        (integral of f dv, s^2/m^5), ``v_par_cut`` (km/s) and ``t_par_cut``
        (eV); of the perpendicular cut, ``n_perp_cut`` (integral of
        f v dv dphi, s/m^4), ``v_perp_cut`` (km/s, perp1 and perp2) and
        ``t_perp_cut`` (eV, 2x2).

    Raises
    ------
    ValueError
        If `vdf_fac` is not a skymap.

    Notes
    -----
    The cuts are slices of the distribution near the field direction or the
    perpendicular plane, not integrals as the reduced distributions of
    `reduce`. For a bi-Maxwellian, the temperature of the parallel cut is
    T_par and that of the perpendicular cut T_perp. The bins are those of
    `vdf_d3v` (from the ``delta_*`` attributes when present); NaN bins are
    left out of the averages. irfu-matlab handles one time, uses the
    electron mass for the speeds (wrong for ions), labels the cut integrals
    as densities in cm^-3 and averages over the azimuths with NaNs as zeros.

    See Also
    --------
    pyrfu.mms.vdf_shift, pyrfu.mms.reduce, pyrfu.pyrf.convert_fac

    Examples
    --------
    >>> import numpy as np
    >>> from pyrfu import mms, pyrf
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> b_hat = pyrf.normalize(pyrf.resample(b_dmpa, vdf_e.time)).data
    >>> e_perp2 = pyrf.normalize(np.cross(b_hat, [1.0, 0.0, 0.0]))
    >>> e_perp1 = np.cross(e_perp2, b_hat)
    >>> rotation = np.stack([e_perp1, e_perp2, b_hat], axis=1)
    >>> vdf_fac = mms.vdf_shift(vdf_e, [0.0, 0.0, 0.0], rotation=rotation)
    >>> dists, moms = mms.vdf_cuts(vdf_fac)

    """
    if not _is_skymap(vdf_fac):
        raise ValueError("vdf_fac must be a skymap")

    relativistic = _relativistic(vdf_fac, None)
    mass = _mass(vdf_fac)
    time = vdf_fac.time.data
    n_t = len(time)

    f_si = vdf_convert(vdf_fac, "s^3/m^6", relativistic).data.data
    energy = np.broadcast_to(vdf_fac.energy.data, f_si.shape[:2])
    speed = _speed(energy, mass, relativistic)
    v_lower, v_upper = _speed_edges(vdf_fac, None, relativistic)
    d_v = v_upper - v_lower
    phi_lower, phi_upper = _phi_edges(vdf_fac)
    d_phi = np.deg2rad(phi_upper - phi_lower)
    phi = np.deg2rad((phi_lower + phi_upper) / 2)
    theta = np.asarray(vdf_fac.theta.data, dtype=np.float64)

    in_par = theta > 180.0 - par_width
    in_apar = theta < par_width
    in_perp = np.abs(theta - 90.0) < perp_width

    with warnings.catch_warnings():
        # Empty windows and bins without data give NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Look directions near -B are particles along B
        f_along = _azimuthal_mean(np.nanmean(f_si[..., in_par], axis=3), d_phi[:, None])
        f_anti = _azimuthal_mean(np.nanmean(f_si[..., in_apar], axis=3), d_phi[:, None])
        f_par = np.hstack([f_anti[:, ::-1], f_along])
        v_par = np.hstack([-speed[:, ::-1], speed])
        d_v_par = np.hstack([d_v[:, ::-1], d_v])

        f_perp_2d = np.nanmean(f_si[..., in_perp], axis=3)
        f_perp = _azimuthal_mean(f_perp_2d, d_phi[:, None])
        f_pa = np.stack(
            [_azimuthal_mean(f_si[..., i], d_phi[:, None]) for i in range(len(theta))],
            axis=-1,
        )[..., ::-1]

        n_par, v_par_cut, t_par = _cut_moments_1d(f_par, v_par, d_v_par, mass)
        n_perp, v_perp_cut, t_perp = _cut_moments_2d(
            f_perp_2d, speed, d_v, phi, d_phi, mass
        )

    dists = xr.Dataset(
        {
            "f_par": (["time", "idx_par"], f_par, {"UNITS": "s^3/m^6"}),
            "v_par": (["time", "idx_par"], v_par / 1e3, {"UNITS": "km/s"}),
            "f_perp_2d": (["time", "idx0", "idx1"], f_perp_2d, {"UNITS": "s^3/m^6"}),
            "f_perp": (["time", "idx0"], f_perp, {"UNITS": "s^3/m^6"}),
            "v_perp": (["time", "idx0"], speed / 1e3, {"UNITS": "km/s"}),
            "energy": (["time", "idx0"], energy, {"UNITS": "eV"}),
            "phi": (["time", "idx1"], np.rad2deg(phi), {"UNITS": "deg"}),
            "f_pa": (["time", "idx0", "idx2"], f_pa, {"UNITS": "s^3/m^6"}),
            "pitch_angle": (["idx2"], 180.0 - theta[::-1], {"UNITS": "deg"}),
        },
        coords={"time": time},
    )
    dists.attrs = {
        "species": vdf_fac.attrs.get("species"),
        "par_width": par_width,
        "perp_width": perp_width,
    }

    vdf_cm = vdf_convert(vdf_fac, "s^3/cm^6", relativistic)
    n_i, v_i, _, _, t_i, _ = psd_moments(
        vdf_cm,
        ts_scalar(time, np.zeros(n_t)),
        angle_integration="exact",
        relativistic=relativistic,
    )

    fac = ["perp1", "perp2", "par"]
    moms = xr.Dataset(
        {
            "n": ("time", n_i.data, {"UNITS": "cm^-3"}),
            "v": (["time", "comp"], v_i.data, {"UNITS": "km/s"}),
            "t": (["time", "comp", "comp_2"], t_i.data, {"UNITS": "eV"}),
            "n_par_cut": ("time", n_par, {"UNITS": "s^2/m^5"}),
            "v_par_cut": ("time", v_par_cut / 1e3, {"UNITS": "km/s"}),
            "t_par_cut": ("time", t_par, {"UNITS": "eV"}),
            "n_perp_cut": ("time", n_perp, {"UNITS": "s/m^4"}),
            "v_perp_cut": (["time", "comp_perp"], v_perp_cut / 1e3, {"UNITS": "km/s"}),
            "t_perp_cut": (
                ["time", "comp_perp", "comp_perp_2"],
                t_perp,
                {"UNITS": "eV"},
            ),
        },
        coords={
            "time": time,
            "comp": fac,
            "comp_2": fac,
            "comp_perp": fac[:2],
            "comp_perp_2": fac[:2],
        },
    )
    moms.attrs = dict(dists.attrs)
    return dists, moms
