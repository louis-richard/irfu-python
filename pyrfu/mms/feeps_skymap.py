#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.ts_skymap import ts_skymap
from .dpf2psd import dpf2psd
from .feeps_omni import _eyes_on_omni_energies
from .feeps_pitch_angles import t_bot, t_top, v_fcs
from .feeps_spin_avg import _spin_bounds

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.1"
__status__ = "Prototype"

# Spin phase (degrees) at the centre of spin sector 0, from the rotation of B
# between BCS and DMPA (FGM) on 2015-2024 survey and burst data
_SECTOR_PHASE_OFFSET = {1: -75.94, 2: -75.93, 3: -75.96, 4: -75.81}

# Polar angles (degrees) of the look directions of the eyes, in BCS
_RINGS = {"electron": [5.0, 65.0, 115.0, 175.0], "ion": [12.0, 49.0, 131.0, 168.0]}

_SECTOR_WIDTH = 360.0 / 64.0

# A survey sample is labelled with the last of the 8 sectors it accumulates:
# the centre of the accumulation is 3.5 sectors before the label
_SRVY_CENTRE_SHIFT = 3.5

# B closer to the spin axis than about 6 degrees (|B_perp| / |B| < 0.1) gives
# an undefined spin phase from FGM
_MIN_B_PERP = 0.1


def _look_direction(eye):
    r"""Polar angle and azimuth (degrees) of the look direction of an eye
    ("top-1", ...) in BCS."""
    s_type, sensor_id = eye.split("-")
    rotation = t_top if s_type == "top" else t_bot
    look = rotation @ np.array(v_fcs[int(sensor_id)])
    look /= np.linalg.norm(look)
    theta = np.rad2deg(np.arccos(look[2]))
    phi = np.rad2deg(np.arctan2(look[1], look[0])) % 360.0
    return theta, phi


def _sector_phase(spin_sectors, tmmode, mms_id):
    r"""Spin phase (degrees) at the centre of the accumulation of each sample.
    Survey samples accumulate the 8 sectors up to the labelled one."""
    sectors = np.asarray(spin_sectors, dtype=float)

    if tmmode == "srvy":
        sectors = sectors - _SRVY_CENTRE_SHIFT

    return sectors * _SECTOR_WIDTH + _SECTOR_PHASE_OFFSET[mms_id]


def _spin_phase(inp_dataset, b_bcs, b_dmpa):
    r"""Spin phase (degrees) of each sample: from the spin sectors, or from FGM
    where it is defined (checked against the sectors). Returns the phases and
    the name of their source."""
    attrs = inp_dataset.attrs
    tmmode, mms_id = attrs["tmmode"], attrs["mmsId"]
    psi = _sector_phase(inp_dataset.spinsectnum.data, tmmode, mms_id)

    if b_bcs is None or b_dmpa is None:
        return psi, "spin sectors"

    t_fgm = b_bcs.time.data.astype("datetime64[ns]").astype(np.int64)
    b_dmpa = b_dmpa.interp(time=b_bcs.time.data)
    b_xyz, d_xyz = b_bcs.data[:, :3], b_dmpa.data[:, :3]

    # B close to the spin axis gives an undefined rotation
    b_perp = np.linalg.norm(b_xyz[:, :2], axis=1) / np.linalg.norm(b_xyz, axis=1)
    good = (
        np.isfinite(b_perp) & np.isfinite(d_xyz).all(axis=1) & (b_perp >= _MIN_B_PERP)
    )

    if not good.any():
        return psi, "spin sectors (B along the spin axis)"

    psi_fgm = np.arctan2(d_xyz[:, 1], d_xyz[:, 0]) - np.arctan2(
        b_xyz[:, 1], b_xyz[:, 0]
    )
    psi_fgm = np.rad2deg(np.unwrap(psi_fgm[good]))

    t_feeps = inp_dataset.time.data.astype("datetime64[ns]").astype(np.int64)
    psi_fgm = np.interp(t_feeps, t_fgm[good], psi_fgm, left=np.nan, right=np.nan)

    if tmmode == "srvy":
        # time tag at the centre of the labelled sector, 3.5 sectors after the
        # centre of the accumulation
        psi_fgm -= _SRVY_CENTRE_SHIFT * _SECTOR_WIDTH

    delta = (psi_fgm - psi + 180.0) % 360.0 - 180.0

    if np.isfinite(delta).any() and np.nanmedian(np.abs(delta)) > 1.0:
        warnings.warn(
            "The FGM spin phase differs from the spin sectors by "
            f"{np.nanmedian(delta):.1f} degrees (median)",
            UserWarning,
        )

    return np.where(np.isfinite(delta), psi + delta, psi), "FGM (b_bcs, b_dmpa)"


def _bin_spins(dalleyes, eyes, psi, bounds, rings, phi_edges):
    r"""Sums and numbers of the finite samples of the eyes in each (spin,
    energy, phi, theta) bin."""
    n_t, n_e, n_eyes = dalleyes.shape
    n_spins, n_phi, n_theta = len(bounds) - 1, len(phi_edges) - 1, len(rings)

    # spin of each sample
    i_spin = np.repeat(np.arange(n_spins), np.diff(bounds))

    # despun azimuth and ring of each sample and eye
    look = np.array([_look_direction(eye) for eye in eyes])
    i_theta = np.array([np.argmin(np.abs(np.array(rings) - t)) for t in look[:, 0]])
    phi = (look[np.newaxis, :, 1] + psi[:, np.newaxis]) % 360.0
    i_phi = np.clip(np.searchsorted(phi_edges, phi, side="right") - 1, 0, n_phi - 1)

    values = np.moveaxis(dalleyes, 2, 1).reshape(n_t * n_eyes, n_e)
    finite = np.isfinite(values)
    index = (
        np.repeat(i_spin, n_eyes),
        i_phi.ravel(),
        np.tile(i_theta, n_t),
    )

    sums = np.zeros((n_spins, n_phi, n_theta, n_e))
    counts = np.zeros((n_spins, n_phi, n_theta, n_e))
    np.add.at(sums, index, np.where(finite, values, 0.0))
    np.add.at(counts, index, finite)

    # (spin, energy, phi, theta)
    return np.moveaxis(sums, 3, 1), np.moveaxis(counts, 3, 1)


def _energy_widths(inp_dataset, eyes, energies):
    r"""Lower and upper widths (keV) of the channels around the omni-directional
    energies, from the channel bounds of the eyes (mean over the eyes), or None
    if the eyes have no bounds."""
    lower, upper = [], []

    for eye in eyes:
        names = [f"energy_{b}_bound_{eye}" for b in ["lower", "upper"]]

        if all(name in inp_dataset[eye].coords for name in names):
            lower.append(inp_dataset[eye][names[0]].data)
            upper.append(inp_dataset[eye][names[1]].data)

    if not lower:
        return None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lower, upper = np.nanmean(lower, axis=0), np.nanmean(upper, axis=0)

    return energies - lower, upper - energies


def _coverage(data, phi_edges, theta_edges):
    r"""Fraction of the sphere covered by the finite bins (spin, energy)."""
    d_phi = np.deg2rad(np.diff(phi_edges))[:, np.newaxis]
    d_cos = -np.diff(np.cos(np.deg2rad(theta_edges)))[np.newaxis, :]
    solid_angle = d_phi * d_cos  # (phi, theta)
    covered = np.isfinite(data) * solid_angle[np.newaxis, np.newaxis]
    return covered.sum(axis=(2, 3)) / (4.0 * np.pi)


def _fill_empty(data, sums, counts, how):
    r"""Fills the empty bins: with the mean of their ring ("ring"), then the
    empty rings with the mean of all the samples of the spin ("omni")."""
    with np.errstate(invalid="ignore"):
        ring = sums.sum(axis=2, keepdims=True) / counts.sum(axis=2, keepdims=True)
        spin = sums.sum(axis=(2, 3), keepdims=True) / counts.sum(
            axis=(2, 3), keepdims=True
        )

    data = np.where(np.isnan(data), ring, data)

    if how == "omni":
        data = np.where(np.isnan(data), spin, data)

    return data


def feeps_skymap(
    inp_dataset,
    b_bcs: xr.DataArray = None,
    b_dmpa: xr.DataArray = None,
    n_phi: int = None,
    to_psd: bool = True,
    fill_empty: str = None,
):
    r"""Skymaps of the FEEPS data, one per spin.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Energy spectra of all the eyes, corrected (`feeps_corrections`), with
        the spin sectors (``spinsectnum``).
    b_bcs : xarray.DataArray, Optional
        Magnetic field in BCS, e.g. ``mms.get_data("b_bcs_fgm_srvy_l2", ...)``.
        With `b_dmpa`, the spin phase is taken from the rotation of B between
        the two frames instead of the spin sectors.
    b_dmpa : xarray.DataArray, Optional
        Magnetic field in DMPA, from the same FGM data as `b_bcs`.
    n_phi : int, Optional
        Number of azimuthal bins. Default is 32 for burst data and 8 for
        survey data (a survey sample spans 45 degrees of the spin).
    to_psd : bool, Optional
        Convert the intensity to the relativistic phase-space density
        (`dpf2psd` with ``relativistic=True``) in s^3/cm^6, as the FPI
        distributions. Default is True.
    fill_empty : {None, "ring", "omni"}, Optional
        Bins without data: left NaN (None, default), filled with the mean of
        their ring ("ring"), or also, for rings without data, with the mean of
        all the samples of the spin, i.e. the spin-averaged omni-directional
        flux ("omni"). Filling assumes gyrotropy around the spin axis ("ring")
        or isotropy ("omni"); the ``coverage`` attribute is that of the data.

    Returns
    -------
    out : xarray.Dataset
        Skymaps (time, energy, phi, theta), as `pyrfu.pyrf.ts_skymap`, in DMPA.
        phi and theta are the look directions of the bins, as for FPI.

    Raises
    ------
    ValueError
        If the integral channel of the eyes is not split, if `to_psd` is True
        and the data are not intensities, or if `fill_empty` is invalid.

    See Also
    --------
    pyrfu.mms.feeps_corrections, pyrfu.mms.feeps_omni, pyrfu.mms.psd_moments

    Notes
    -----
    Each spin, from its first sample up to the first sample of the next one
    (as `feeps_spin_avg`), gives one skymap time stamped at its first sample.
    The eyes are put on the omni-directional energies as in `feeps_omni` (10 %
    check and correction factor); the energy widths are the mean of the
    channel bounds of the eyes (`get_feeps_alleyes`), if available.

    The look direction of an eye is fixed in BCS (`feeps_pitch_angles`); its
    azimuth in DMPA is the BCS azimuth plus the spin phase at the centre of the
    accumulation. The spin phase is 5.625 degrees per spin sector plus a
    constant per spacecraft (-75.8 to -76.0 degrees), measured from the
    rotation of B between BCS and DMPA on 2015 to 2024 data. A survey sample
    accumulates the 8 sectors up to the labelled one, so its centre is 3.5
    sectors before the label. With `b_bcs` and `b_dmpa`, the spin phase is
    taken from FGM where B is at least about 6 degrees from the spin axis
    (|B_perp| / |B| >= 0.1), and a
    warning is given if it differs from the spin sectors by more than 1 degree.

    The eyes lie on 4 cones around the spin axis (electrons 5, 65, 115 and 175
    degrees, ions 12, 49, 131 and 168 degrees), the theta bins. Their widths
    (``delta_theta_minus/plus``) reach halfway to the next cone, so that the
    bins cover the sphere: moments are coarse partial moments above about 30
    keV. The eyes measure the total ion flux, which is converted to phase-space
    density as protons. Zero intensities are kept, as in `feeps_omni`.

    Bad eyes, the energy check and the sun masks leave bins, sometimes whole
    rings, without data. They count as zero in `psd_moments`, so the moments
    are lower limits where the ``coverage`` attribute (fraction of the sphere
    with data, per time and energy) is below 1; see `fill_empty`. The skymaps
    have ``angle_integration = "exact"``, so that `psd_moments` integrates the
    wide bins exactly, and ``relativistic = True``, so that `psd_moments` uses
    the relativistic speeds (the non-relativistic moments underestimate the
    electron density by about 10 % at the FEEPS energies). The moments and the
    reduced distributions (`reduce`) of the phase-space density are then the
    integrals of the intensity over the energy and the solid angle.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-24T12:48:30.000", "2017-07-24T12:50:23.000"]
    >>> feeps_e = mms.get_feeps_alleyes("fluxe_brst_l2", tint, 1)
    >>> feeps_e = mms.feeps_corrections(feeps_e)
    >>> vdf_e = mms.feeps_skymap(feeps_e)

    With the spin phase from FGM

    >>> b_bcs = mms.get_data("b_bcs_fgm_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> vdf_e = mms.feeps_skymap(feeps_e, b_bcs, b_dmpa)

    """
    eyes, dalleyes, energies, g_fact = _eyes_on_omni_energies(inp_dataset)
    attrs = inp_dataset.attrs
    tmmode, d_type, mms_id = attrs["tmmode"], attrs["dtype"], attrs["mmsId"]
    units = inp_dataset[eyes[0]].attrs.get("UNITS", "1/(cm^2 s sr keV)")

    if to_psd and units.lower() != "1/(cm^2 s sr kev)":
        raise ValueError(f"Phase-space density needs intensities, not {units}")

    if fill_empty not in [None, "ring", "omni"]:
        raise ValueError(f"Invalid fill_empty {fill_empty}: must be None, ring or omni")

    if n_phi is None:
        n_phi = 32 if tmmode == "brst" else 8

    psi, phase_source = _spin_phase(inp_dataset, b_bcs, b_dmpa)
    bounds = _spin_bounds(np.asarray(inp_dataset.spinsectnum.data))
    rings = _RINGS[d_type]
    phi_edges = np.linspace(0.0, 360.0, n_phi + 1)

    sums, counts = _bin_spins(dalleyes, eyes, psi, bounds, rings, phi_edges)

    with np.errstate(invalid="ignore"):
        data = g_fact * sums / counts

    theta_edges = np.hstack([0.0, (np.array(rings[1:]) + rings[:-1]) / 2.0, 180.0])
    coverage = _coverage(data, phi_edges, theta_edges)

    if fill_empty is not None:
        data = _fill_empty(data, g_fact * sums, counts, fill_empty)

    time = inp_dataset.time.data[bounds[:-1]]
    n_spins = len(time)
    phi = np.tile(phi_edges[:-1] + np.diff(phi_edges) / 2.0, (n_spins, 1))

    glob_attrs = {
        "species": f"{d_type}s",
        "mmsId": mms_id,
        "tmmode": tmmode,
        "spin_phase": phase_source,
        "angle_integration": "exact",
        "relativistic": bool(to_psd),
        "coverage": coverage,
        "fill_empty": str(fill_empty),
        "delta_phi_minus": np.full(phi.shape, 180.0 / n_phi),
        "delta_phi_plus": np.full(phi.shape, 180.0 / n_phi),
        "delta_theta_minus": np.array(rings) - theta_edges[:-1],
        "delta_theta_plus": theta_edges[1:] - np.array(rings),
    }

    delta_minus, delta_plus = _energy_widths(inp_dataset, eyes, energies)

    if delta_minus is not None:
        # (time, energy) in eV, as the FPI distributions
        glob_attrs["delta_energy_minus"] = np.tile(1e3 * delta_minus, (n_spins, 1))
        glob_attrs["delta_energy_plus"] = np.tile(1e3 * delta_plus, (n_spins, 1))

    coords_attrs = {
        "phi": {"UNITS": "deg", "COORDINATE_SYSTEM": "DMPA"},
        "theta": {"UNITS": "deg", "COORDINATE_SYSTEM": "DMPA"},
        "energy": {"UNITS": "eV"},
    }

    energy = np.tile(1e3 * energies, (n_spins, 1))

    out = ts_skymap(
        time,
        data,
        energy,
        phi,
        np.array(rings),
        energy0=energy[0],
        energy1=energy[0],
        esteptable=np.zeros(n_spins, dtype=np.uint8),
        attrs={
            "UNITS": units,
            "FIELDNAM": f"mms{mms_id:d}_feeps_{tmmode}_{d_type}_skymap",
        },
        glob_attrs=glob_attrs,
        coords_attrs=coords_attrs,
    )

    if to_psd:
        # s^3/cm^6, as the FPI distributions
        out = dpf2psd(out, relativistic=True)
        out.data.data *= 1e-12
        out.data.attrs["UNITS"] = "s^3/cm^6"

    return out
