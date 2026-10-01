#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.normalize import normalize
from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar
from .db_get_variable import db_get_variable

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


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
    r"""First sample of vdf and first record of aze of the matching half-spins,
    and the number of complete half-spins.

    Each aze record gives the azimuths of one half-spin, i.e., of n_az consecutive
    samples of vdf with start azimuths 0, 1, ..., n_az - 1. The aze time is either
    the start of the half-spin (time of the start azimuth 0 sample, as in the
    files), or its centre (times shifted to the middle of the accumulations, as by
    :func:`pyrfu.mms.get_data`: 7.5 samples later).

    """
    if len(saz) != len(vdf) or not np.array_equal(saz.time.data, vdf.time.data):
        raise ValueError("saz and vdf must have the same times")

    t_vdf = vdf.time.data.astype("datetime64[ns]").astype(np.int64)
    t_aze = aze.time.data.astype("datetime64[ns]").astype(np.int64)
    d_t = np.median(np.diff(t_vdf))
    starts = np.flatnonzero(saz.data == 0)

    for offset in [0.0, (n_az - 1) / 2 * d_t]:
        for i_aze, t_rec in enumerate(t_aze):
            if starts.size == 0:
                break

            mismatch = np.abs(t_vdf[starts] + offset - t_rec)

            if np.min(mismatch) < d_t / 2:
                i_vdf = int(starts[np.argmin(mismatch)])
                n_spins = min(len(aze) - i_aze, (len(vdf) - i_vdf) // n_az)
                expected = np.tile(np.arange(n_az), n_spins)
                start_az = saz.data[i_vdf : i_vdf + n_spins * n_az]

                if n_spins < 1 or not np.array_equal(start_az, expected):
                    raise ValueError("start azimuths of vdf don't match aze half-spins")

                return i_vdf, i_aze, n_spins

    raise ValueError("start times of aze and vdf don't match!")


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
        B in dmpa coordinate
    elim : list, Optional
        [emin, emax], energy range for PAD. Default is the whole energy range.
    elevation : array_like, Optional
        Polar angles (colatitudes, deg) of the 16 anodes. Default reads
        mms?_hpca_centroid_elevation_angle in the file of `vdf`.
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to read the anode elevations from (if `elevation` is None).
        Default uses default in `pyrfu/mms/config.json`.
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
        If the half-spins of `aze` and the samples of `vdf` don't match.

    Notes
    -----
    The azimuths of `aze` are those of the look directions of the anodes, and the
    elevations the polar angles of the particle velocities: the particles move at
    azimuth + 180 deg and polar angle elevation. With this convention, the
    flux-weighted direction of the H+ distributions matches the direction of the
    official HPCA bulk velocity (to 3 deg in the spin plane on 2017-07-06, MMS1,
    22:36-22:47 UT), and the pitch angle asymmetry that of FPI. B is resampled to
    the time of each energy step.

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

    # 2. complete half-spins of aze and vdf
    i_vdf, i_aze, n_spins = _match_half_spins(vdf, saz, aze, n_az)
    vdf = vdf.isel(time=slice(i_vdf, i_vdf + n_spins * n_az))
    aze_data = np.asarray(aze.data[i_aze : i_aze + n_spins], dtype=np.float64)

    # 3. compute PAD
    # 3.1. pitchangle
    # Default pitch angles. 15 degree angle widths
    angle_vec = np.linspace(15, 180, 12)

    d_angle = np.median(np.diff(angle_vec)) * np.ones(len(angle_vec))
    pitch_a = angle_vec - d_angle / 2

    # 3.2. data dimension
    n_po, n_en, n_ti = len(elevation), len(ien), len(vdf)
    tt_ = vdf.time.data

    # 3.3. particle velocity directions [nt, npo, ner]: sample k is the azimuth
    # step k % 16 of the half-spin k // 16, so aze [nT, naz, npo, ner] reshapes in
    # this order. The azimuths are those of the look directions (velocity at
    # azimuth + 180 deg), the elevations the polar angles of the velocities.
    phi = np.deg2rad(aze_data.reshape(n_ti, n_po, n_en))
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
    t0_ = tt_.astype("datetime64[ns]").astype(np.int64)
    dt0 = np.diff(t0_, append=2 * t0_[-1] - t0_[-2])
    t1_tt = t0_[:, None] + np.arange(n_en)[None, :] * dt0[:, None] / n_en
    t1_tt = t1_tt.ravel().astype(np.int64).astype("datetime64[ns]")
    b_xyz_r = resample(b_xyz, ts_scalar(t1_tt, np.zeros(len(t1_tt))))
    b_hat = normalize(b_xyz_r).data.reshape(n_ti, n_en, 3)[:, None, :, :]

    # Pitch angle of the particles
    cos_pa = np.sum(v_dir * b_hat, axis=-1)
    theta_b = np.rad2deg(np.arccos(np.clip(cos_pa, -1.0, 1.0)))

    # 3.5. select dist for PAD
    vdfs_ = [vdf.data.copy() for _ in range(len(angle_vec))]

    vdfs_[0][theta_b > angle_vec[0]] = np.nan

    for i, vdf_ in enumerate(vdfs_):
        vdf_[theta_b < (angle_vec[i] - d_angle[i])] = np.nan
        vdf_[theta_b > angle_vec[i]] = np.nan

    vdfs_[-1][theta_b < (angle_vec[-1] - d_angle[len(angle_vec) - 1])] = np.nan

    # [n_ti, n_po, n_en] --> [n_ti, n_en]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vdfs_ = [np.nanmean(vdf_, axis=1) for vdf_ in vdfs_]

    # average among energy dimension (both ends of elim included)
    i_elim = np.argmin(abs(ien.data - elim[0]))
    e_min = ien.data[i_elim]
    j_elim = np.argmin(abs(ien.data - elim[1]))
    e_max = ien.data[j_elim]
    logging.info("PSD/pflux pitch angle dist. from %s [eV] to %s [eV]", e_min, e_max)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vdfs_ = [np.nanmean(vdf_[:, i_elim : j_elim + 1], axis=1) for vdf_ in vdfs_]

    padd_ = np.transpose(np.stack(vdfs_))  # [nt, npitcha12]

    # 3.6. make spectrum
    coords = [tt_, pitch_a]
    dims = ["time", "theta"]
    pad_spec = xr.DataArray(padd_, coords=coords, dims=dims)

    return pad_spec
