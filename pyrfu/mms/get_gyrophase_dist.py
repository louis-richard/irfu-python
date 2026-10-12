#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.normalize import normalize
from ..pyrf.resample import resample
from ..pyrf.time_clip import time_clip

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def _gyrophase_edges(angles):
    r"""Edges of the gyrophase bins (degrees) from a number of bins or the
    edges."""
    if isinstance(angles, (int, np.integer)):
        return np.linspace(0.0, 360.0, int(angles) + 1)

    if isinstance(angles, (list, tuple, np.ndarray)) and len(angles) > 1:
        return np.asarray(angles, dtype=np.float64)

    raise ValueError("angles must be a number of bins or the bin edges")


def _solid_angles(vdf, n_t):
    r"""Solid angle (sr) of each (phi, theta) bin, (time, phi, theta): from the
    angular widths in the attributes (degrees), or else from the spacing of
    the grids (sin(theta) d_theta d_phi)."""
    phi, theta = vdf.phi.data, vdf.theta.data
    attrs = vdf.attrs
    n_phi = phi.shape[-1]

    if "delta_phi_minus" in attrs and "delta_phi_plus" in attrs:
        d_phi = np.deg2rad(attrs["delta_phi_minus"] + attrs["delta_phi_plus"])
        d_phi = np.broadcast_to(d_phi, (n_t, n_phi))
    else:
        d_phi = np.deg2rad(np.median(np.diff(np.atleast_2d(phi)[0])))
        d_phi = np.full((n_t, n_phi), d_phi)

    if "delta_theta_minus" in attrs and "delta_theta_plus" in attrs:
        theta_a = np.deg2rad(theta - attrs["delta_theta_minus"])
        theta_b = np.deg2rad(theta + attrs["delta_theta_plus"])
        band = np.cos(theta_a) - np.cos(theta_b)
    else:
        d_theta = np.deg2rad(np.median(np.diff(theta)))
        band = np.sin(np.deg2rad(theta)) * d_theta

    return d_phi[:, :, np.newaxis] * band[np.newaxis, np.newaxis, :]


def _reference(r_xyz, vdf_data):
    r"""Reference direction at the times of the distribution (time, 3)."""
    if r_xyz is None:
        r_xyz = [1.0, 0.0, 0.0]

    if isinstance(r_xyz, xr.DataArray):
        return resample(r_xyz, vdf_data).data

    r_xyz = np.asarray(r_xyz, dtype=np.float64)

    if r_xyz.shape != (3,):
        raise ValueError("r_xyz must be a 3-vector or a time series of vectors")

    return np.tile(r_xyz, (len(vdf_data.time), 1))


def get_gyrophase_dist(
    vdf,
    b_xyz,
    tint: list = None,
    r_xyz=None,
    angles=24,
    pitch_angle_range: list = None,
):
    r"""Computes the gyrophase distributions from particle data.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution, with (time, energy) energies, phi as (time, phi)
        or (phi,) and theta as (theta,), the look directions in degrees (as
        FPI, EIS and FEEPS skymaps).
    b_xyz : xarray.DataArray
        Time series of the magnetic field in the same coordinate system as
        vdf (e.g., DMPA for FPI and FEEPS), resampled to the times of vdf.
    tint : list of str, Optional
        Time interval for closeup.
    r_xyz : array_like or xarray.DataArray, Optional
        Reference direction of the gyrophase, in the coordinate system of vdf:
        a 3-vector or a time series. The gyrophase is 0 along the component of
        `r_xyz` perpendicular to B. Default is [1, 0, 0] (the sunward
        direction for DMPA and GSE), as `pyrfu.pyrf.convert_fac`.
    angles : int or array_like, Optional
        Number of gyrophase bins of equal width, or the bin edges in degrees.
        Default is 24 bins of 15 degrees.
    pitch_angle_range : array_like, Optional
        Range of pitch angles [min, max] (degrees, inclusive) of the samples
        used. Default uses all the pitch angles.

    Returns
    -------
    gpd : xarray.Dataset
        Gyrophase distribution, with data (time, energy, gyrophase), energy
        (time, energy) and gyrophase (time, gyrophase) the centres of the
        gyrophase bins.

    Raises
    ------
    ValueError
        If angles, r_xyz or pitch_angle_range is not understood.

    Notes
    -----
    The field-aligned coordinates are those of `pyrfu.pyrf.convert_fac`:
    e_par along B, e_perp2 = e_par x r_xyz / |e_par x r_xyz| and e_perp1 =
    e_perp2 x e_par, so that e_perp1 is along the component of `r_xyz`
    perpendicular to B. The gyrophase of a sample is the angle of the particle
    velocity (minus the look direction) from e_perp1 towards e_perp2, i.e.
    right-handed about B, in [0, 360) degrees, as Li et al. (2022, 2024). It
    is undefined where `r_xyz` is along B. IDL SPEDAS and pyspedas
    (``mms_feeps_getgyrophase``) measure the gyrophase from the sunward
    direction towards the dawnward direction (perpendicular to B), whatever the
    direction of B: the same as here with the default `r_xyz` when e_perp2
    points dawnward (B with a southward component), the opposite (360 degrees
    minus the gyrophase) otherwise.

    A sample is in a bin if its gyrophase is in [lower edge, upper edge). Each
    bin is the mean of its samples weighted by their solid angle (exact for
    the angular widths of the attributes, e.g. the FEEPS skymaps, sin(theta)
    for uniform grids); a bin without data is NaN. The gyrophase resolution
    is limited by the angular grid of the skymap (45 degree azimuth bins for
    FEEPS survey data).

    References
    ----------
    Li, J.-H. et al. (2022). Anomalous resonance between low-energy particles
    and electromagnetic plasma waves. Commun. Phys., 5, 300.

    Li, J.-H. et al. (2024). Identification of coupled Landau and anomalous
    resonances in space plasmas. Phys. Rev. Lett., 133, 035201.

    Examples
    --------
    >>> from pyrfu import mms

    >>> tint = ["2020-01-14T19:23:30.000", "2020-01-14T19:23:50.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)

    Gyrophase distribution of the ions moving perpendicular to B

    >>> options = dict(angles=12, pitch_angle_range=[78.75, 101.25])
    >>> gpd_i = mms.get_gyrophase_dist(vdf_i, b_dmpa, **options)

    """
    edges = _gyrophase_edges(angles)

    if pitch_angle_range is not None and len(pitch_angle_range) != 2:
        raise ValueError("pitch_angle_range must be [min, max] in degrees")

    vdf_data = vdf.data
    energy = vdf.energy.data

    # Azimuths as (time, phi)
    phi = vdf.phi.data
    if phi.ndim == 1:
        phi = np.broadcast_to(phi, (len(vdf.time), len(phi)))

    solid_angles = _solid_angles(vdf, len(vdf.time))

    if tint is not None:
        b_xyz = time_clip(b_xyz, tint)
        vdf_data = time_clip(vdf_data, tint)
        in_tint = np.isin(vdf.time.data, vdf_data.time.data)
        energy, phi = energy[in_tint], phi[in_tint]
        solid_angles = solid_angles[in_tint]

    time = vdf_data.time.data
    data = vdf_data.data

    # Field-aligned unit vectors (time, 3), as convert_fac
    b_hat = normalize(resample(b_xyz, vdf_data)).data
    e_perp2 = np.cross(b_hat, _reference(r_xyz, vdf_data))

    with np.errstate(invalid="ignore", divide="ignore"):
        e_perp2 /= np.linalg.norm(e_perp2, axis=1, keepdims=True)

    e_perp1 = np.cross(e_perp2, b_hat)

    # Particle velocity directions (minus the look directions), (time, phi,
    # theta, 3)
    phi_r = np.deg2rad(phi)[:, :, np.newaxis]
    theta_r = np.deg2rad(vdf.theta.data)[np.newaxis, np.newaxis, :]
    u_xyz = -np.stack(
        np.broadcast_arrays(
            np.cos(phi_r) * np.sin(theta_r),
            np.sin(phi_r) * np.sin(theta_r),
            np.cos(theta_r),
        ),
        axis=-1,
    )

    def _dot(e_xyz):
        return np.einsum("tpqi,ti->tpq", u_xyz, e_xyz)

    gyrophase = np.rad2deg(np.arctan2(_dot(e_perp2), _dot(e_perp1))) % 360.0
    pitch_angle = np.rad2deg(np.arccos(np.clip(_dot(b_hat), -1.0, 1.0)))

    # Samples used (time, phi, theta)
    used = np.isfinite(gyrophase)

    if pitch_angle_range is not None:
        used &= pitch_angle >= pitch_angle_range[0]
        used &= pitch_angle <= pitch_angle_range[1]

    finite = np.isfinite(data)
    data0 = np.where(finite, data, 0.0)
    weights = finite * solid_angles[:, np.newaxis]

    gpd_arr = np.full((len(time), energy.shape[1], len(edges) - 1), np.nan)

    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = used & (gyrophase >= lower) & (gyrophase < upper)
        w_bin = weights * in_bin[:, np.newaxis]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gpd_arr[..., i] = np.sum(w_bin * data0, axis=(2, 3)) / np.sum(
                w_bin, axis=(2, 3)
            )

    centres = (edges[:-1] + edges[1:]) / 2

    gpd = xr.Dataset(
        {
            "data": (["time", "idx0", "idx1"], gpd_arr),
            "energy": (["time", "idx0"], energy),
            "gyrophase": (["time", "idx1"], np.tile(centres, (len(time), 1))),
            "time": time,
            "idx0": np.arange(energy.shape[1]),
            "idx1": np.arange(len(centres)),
        },
    )

    gpd.attrs = {
        **vdf.attrs,
        "delta_gyrophase_minus": centres - edges[:-1],
        "delta_gyrophase_plus": edges[1:] - centres,
        "pitch_angle_range": (
            "all" if pitch_angle_range is None else np.asarray(pitch_angle_range)
        ),
        "reference": (
            "time series"
            if isinstance(r_xyz, xr.DataArray)
            else str([1.0, 0.0, 0.0] if r_xyz is None else list(r_xyz))
        ),
    }

    gpd.time.attrs = dict(vdf.time.attrs)
    gpd.energy.attrs = dict(vdf.energy.attrs)
    gpd.data.attrs["UNITS"] = vdf.data.attrs.get("UNITS", "")

    return gpd
