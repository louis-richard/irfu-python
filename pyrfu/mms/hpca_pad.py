#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf._time import _ns
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.nanavg_4sc import _without_gaps
from ..pyrf.normalize import normalize
from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar
from ._hpca import _half_spins
from .db_get_variable import db_get_variable

logger = logging.getLogger(__name__)


def _hpca_elevations(vdf, source: str = "default", data_path: str = ""):
    r"""Polar angles (colatitudes) of the 16 HPCA anodes, read from
    mms?_hpca_centroid_elevation_angle in the file of `vdf`."""
    try:
        dataset = vdf.attrs["GLOBAL"]["Logical_source"]  # e.g., mms1_hpca_brst_l2_ion
    except (KeyError, TypeError) as err:
        raise ValueError(
            "Can't find the HPCA dataset of vdf (GLOBAL Logical_source attribute): "
            "give the anode elevations with elevation"
        ) from err

    cdf_name = f"{dataset.split('_')[0]}_hpca_centroid_elevation_angle"
    tint = list(datetime642iso8601(vdf.time.data[[0, -1]]))
    elevation = db_get_variable(
        dataset, cdf_name, tint, verbose=False, data_path=data_path, source=source
    )

    return np.asarray(elevation.data, dtype=np.float64).ravel()


def _match_half_spins(vdf, saz, aze, n_az: int = 16):
    r"""First sample of vdf of each half-spin of aze, matched by time as in
    :func:`pyrfu.mms.get_hpca_dist`, and whether it is complete: its n_az
    samples in vdf with start azimuths 0, 1, ..., n_az - 1 and non-zero
    azimuths. The aze time is either the start of the half-spin (as in the
    files), or its centre (as :func:`pyrfu.mms.get_data` gives it). The
    half-spins were paired by count after the first one: a missing aze record
    shifted all the next half-spins by 180 deg, and a gap in vdf raised."""
    if len(saz) != len(vdf) or not np.array_equal(saz.time.data, vdf.time.data):
        raise ValueError("saz and vdf must have the same times")

    i_start, complete = _half_spins(vdf, aze)
    i_samples = np.clip(i_start[:, None] + np.arange(n_az), 0, len(vdf) - 1)
    complete &= np.all(saz.data[i_samples] == np.arange(n_az), axis=1)

    if not np.any(complete):
        raise ValueError("no half-spin of aze complete in vdf")

    return i_start, complete


def hpca_pad(
    vdf,
    saz,
    aze,
    b_xyz,
    elim=None,
    elevation=None,
    source: str = "default",
    data_path: str = "",
):
    r"""Computes HPCA pitch angle distribution.

    Parameters
    ----------
    vdf : xarray.DataArray
        Ion PSD or flux; [nt, npo16, ner63].
    saz : xarray.DataArray
        Start index of azimuthal angle; [nt], (0 - 15), at the times of `vdf`.
    aze : xarray.DataArray
        Azimuthal angle per energy; [nT, naz16, npo16, ner63], one record per
        half-spin.
    b_xyz : xarray.DataArray
        B in dmpa coordinate. No pitch angle (NaN) outside its time range and
        in its data gaps, as :func:`pyrfu.mms.feeps_pitch_angles` (it was held
        at its last value and interpolated across).
    elim : list, Optional
        [emin, emax], energy range for PAD. Default is the whole energy range.
    elevation : array_like, Optional
        Polar angles (colatitudes, deg) of the 16 anodes. Default reads
        mms?_hpca_centroid_elevation_angle in the file of `vdf`.
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to read the anode elevations from (if `elevation` is None).
        Default uses the default resource of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).
    data_path : str, Optional
        Path of MMS data (for `source` "local").

    Returns
    -------
    pad_spec : xarray.DataArray
        PAD spectrum, averaged over the energy range, for 12 pitch angle bins of
        15 deg.

    Raises
    ------
    ValueError
        If `saz` is not at the times of `vdf`, or if no half-spin of `aze` is
        complete in `vdf`.

    Notes
    -----
    The azimuths of `aze` are those of the look directions of the anodes, and the
    elevations the polar angles of the particle velocities: the particles move at
    azimuth + 180 deg and polar angle elevation. With this convention, the
    flux-weighted direction of the H+ distributions matches the direction of the
    official HPCA bulk velocity (to 3 deg in the spin plane on 2017-07-06, MMS1,
    22:36-22:47 UT), and the pitch angle asymmetry that of FPI. B is resampled to
    the time of each energy step.

    Each half-spin of `aze` is matched with its samples of `vdf` by time, as in
    :func:`pyrfu.mms.get_hpca_dist`. The PAD covers the samples from the first
    to the last complete half-spin, and is NaN for the samples of the
    half-spins that are not complete in `vdf` (data gaps, start azimuths not
    0 to 15) or without azimuths (zeros, dropped as in IDL SPEDAS), and for
    the samples without half-spin. The half-spins were paired by count after
    the first one, and the zero azimuths were used as azimuths.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
    >>> vdf = mms.get_data("dpfhplus_hpca_brst_l2", tint, 1)
    >>> saz = mms.get_data("saz_hpca_brst_l2", tint, 1)
    >>> aze = mms.get_data("azimuth_hpca_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> pad_ = mms.hpca_pad(vdf, saz, aze, b_dmpa, elim=[500, 3000])

    """

    # 1. get data
    ien = vdf.ccomp
    n_az = 16  # azimuthal angles per half-spin

    if elevation is None:
        elevation = _hpca_elevations(vdf, source, data_path)

    elevation = np.asarray(elevation, dtype=np.float64).ravel()

    if elim is None:
        elim = [ien.data[0], ien.data[-1]]

    # 2. complete half-spins of aze, matched with vdf by time, from the first to
    # the last one; the samples of the others are NaN
    i_start, complete = _match_half_spins(vdf, saz, aze, n_az)
    i_first = i_start[complete][0]
    vdf = vdf.isel(time=slice(i_first, i_start[complete][-1] + n_az))
    rows = (i_start[complete] - i_first)[:, None] + np.arange(n_az)
    aze_data = np.full((len(vdf),) + aze.shape[2:], np.nan)
    aze_data[rows.ravel()] = np.asarray(aze.data[complete], dtype=np.float64).reshape(
        (-1,) + aze.shape[2:]
    )

    # 3. compute PAD
    # 3.1. pitchangle
    # Default pitch angles. 15 degree angle widths
    angle_vec = np.linspace(15, 180, 12)

    d_angle = np.median(np.diff(angle_vec)) * np.ones(len(angle_vec))
    pitch_a = angle_vec - d_angle / 2

    # 3.2. data dimension
    n_en, n_ti = len(ien), len(vdf)
    tt_ = vdf.time.data

    # 3.3. particle velocity directions [nt, npo, ner]: each sample has the
    # azimuths of its step in its half-spin (NaN without). The azimuths are
    # those of the look directions (velocity at azimuth + 180 deg), the
    # elevations the polar angles of the velocities.
    phi = np.deg2rad(aze_data)
    theta = np.deg2rad(elevation)[None, :, None]
    v_dir = np.stack(
        [
            -np.sin(theta) * np.cos(phi),
            -np.sin(theta) * np.sin(phi),
            np.cos(theta) * np.ones_like(phi),
        ],
        axis=-1,
    )

    # 3.4. B at the time of each energy step (samples k * ner + e) [nt, 1, ner]
    t0_ = _ns(tt_)
    dt0 = np.diff(t0_, append=2 * t0_[-1] - t0_[-2])
    t1_tt = t0_[:, None] + np.arange(n_en)[None, :] * dt0[:, None] / n_en
    t1_tt = t1_tt.ravel().astype(np.int64).astype("datetime64[ns]")
    b_xyz_r = resample(b_xyz, ts_scalar(t1_tt, np.zeros(len(t1_tt))))
    b_xyz_r = _without_gaps(b_xyz, b_xyz_r)
    b_hat = normalize(b_xyz_r).data.reshape(n_ti, n_en, 3)[:, None, :, :]

    # Pitch angle of the particles
    cos_pa = np.sum(v_dir * b_hat, axis=-1)
    theta_b = np.rad2deg(np.arccos(np.clip(cos_pa, -1.0, 1.0)))

    # 3.5. select dist for PAD (NaN for the samples without azimuths)
    vdf_data = np.where(np.isnan(phi) | np.isnan(theta_b), np.nan, vdf.data)

    # energies averaged (both ends of elim included)
    i_elim = np.argmin(abs(ien.data - elim[0]))
    e_min = ien.data[i_elim]
    j_elim = np.argmin(abs(ien.data - elim[1]))
    e_max = ien.data[j_elim]
    logger.info("PSD/pflux pitch angle dist. from %s [eV] to %s [eV]", e_min, e_max)

    # Each pitch-angle bin [angle - d_angle, angle], one at a time (the 12
    # masked copies of the data were kept): mean over the anodes [n_ti, n_en],
    # then over the energies of elim [n_ti]
    pads = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for lower, upper in zip(angle_vec - d_angle, angle_vec):
            in_bin = (theta_b >= lower) & (theta_b <= upper)
            pad = np.nanmean(np.where(in_bin, vdf_data, np.nan), axis=1)
            pads.append(np.nanmean(pad[:, i_elim : j_elim + 1], axis=1))

    padd_ = np.transpose(np.stack(pads))  # [nt, npitcha12]

    # 3.6. make spectrum
    coords = [tt_, pitch_a]
    dims = ["time", "theta"]
    pad_spec = xr.DataArray(padd_, coords=coords, dims=dims)

    # Units of vdf (none), pitch angles in deg
    if "UNITS" in vdf.attrs:
        pad_spec.attrs["UNITS"] = vdf.attrs["UNITS"]

    pad_spec.theta.attrs["UNITS"] = "deg"

    return pad_spec
