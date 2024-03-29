#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from ..pyrf.calc_dt import calc_dt
from ..pyrf.convert_fac import convert_fac
from ..pyrf.extend_tint import extend_tint
from ..pyrf.filt import filt
from ..pyrf.resample import resample
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def lh_wave_analysis(
    tints,
    e_xyz,
    b_scm,
    b_xyz,
    n_e,
    min_freq: float = 10.0,
    max_freq: float = 0.0,
    lowpass_b_xyz: float = 2.0,
):
    r"""Calculates lower-hybrid wave properties from MMS data

    Parameters
    ----------
    tints : list of str
        Time interval
    e_xyz : xarray.DataArray
        Time series pf the electric field
    b_scm : xarray.DataArray
        Time series of the fluctuations of the magnetic field
    b_xyz : xarray.DataArray
        Time series of the background magnetic field
    n_e : xarray.DataArray
        Time series of the number density
    min_freq : float, Optional
        Minimum frequency in the highpass filter for LH fluctuations.
        Default is 10.
    max_freq : float, Optional
        Maximum frequency in the bandpass filter for LH fluctuations.
        Default is 0 (highpass filter).
    lowpass_b_xyz : float, Optional
         Maximum frequency for low-pass filter of background magnetic
         field (FGM)

    Returns
    -------
    phi_eb : xarray.DataArray
        to fill
    v_best : ndarray
        to fill
    dir_best : ndarray
        to fill
    thetas : ndarray
        to fill
    corrs : ndarray
        to fill

    Examples
    --------
    >>> from pyrfu.mms import get_data, lh_wave_analysis

    Define time intervals

    >>> tint_long = ["2015-12-14T01:17:39.000", "2015-12-14T01:17:43.000"]
    >>> tint_zoom = ["2015-12-14T01:17:40.200","2015-12-14T01:17:41.500"]

    Load fields and density

    >>> b_gse = get_data("b_gse_fgm_brst_l2", tint_long, 2)
    >>> e_gse = get_data("e_gse_edp_brst_l2", tint_long, 2)
    >>> b_scm = get_data("b_gse_scm_brst_l2", tint_long, 2)
    >>> n_e = get_data("ne_fpi_brst_l2", tint_long, 2)

    Lower Hybrid Waves Analysis

    >>> opt = dict(lhfilt=[5, 100])
    >>> res = lh_wave_analysis(tint, e_xyz, b_scm, b_xyz, n_e, **opt)

    """

    # Bandpass filter data
    b_xyz = filt(b_xyz, 0, lowpass_b_xyz, 5)
    e_xyz = resample(e_xyz, b_scm)
    n_e = resample(n_e, b_scm)
    b_xyz = resample(b_xyz, b_scm)
    b_scm_fac = convert_fac(b_scm, b_xyz, [1, 0, 0])
    b_scm_fac = filt(b_scm_fac, min_freq, max_freq, 5)
    e_xyz = filt(e_xyz, min_freq, max_freq, 5)

    q_e = constants.elementary_charge
    mu0 = constants.mu_0

    b_mag = np.linalg.norm(b_xyz, axis=1)
    phi_b = b_scm_fac.data[:, 2] * b_mag * 1e-18 / (n_e.data * q_e * mu0 * 1e6)
    phi_b = ts_scalar(b_scm_fac.time.data, phi_b)

    # short buffer so phi_E does not begin at zero.
    tint = extend_tint(tints, [-0.2, 0.2])

    e_xyz = time_clip(e_xyz, tint)
    phi_bs = time_clip(phi_b, tints)

    # Rotate e_xyz into field-aligned coordinates
    b_xyz_tc = time_clip(b_xyz, tints)
    b_mean = np.mean(b_xyz_tc.data, axis=0)
    b_vec = b_mean / np.linalg.norm(b_mean)
    r_temp = [1, 0, 0]
    bxr = np.cross(b_vec, r_temp)
    bxr /= np.linalg.norm(bxr)
    bxrxb = np.cross(bxr, b_vec)

    er1 = e_xyz.data[:, 0] * bxrxb[0]
    er1 += e_xyz.data[:, 1] * bxrxb[1]
    er1 += e_xyz.data[:, 2] * bxrxb[2]

    er2 = e_xyz.data[:, 0] * bxr[0]
    er2 += e_xyz.data[:, 1] * bxr[1]
    er2 += e_xyz.data[:, 2] * bxr[2]

    er3 = e_xyz.data[:, 0] * b_vec[0]
    er3 += e_xyz.data[:, 1] * b_vec[1]
    er3 += e_xyz.data[:, 2] * b_vec[2]

    e_fac = ts_vec_xyz(e_xyz.time.data, np.vstack([er1, er2, er3]).T)

    # Find best direction
    dt_e_fac = calc_dt(e_fac)
    thetas = np.linspace(0, 2 * np.pi, 361)
    corrs = np.zeros(len(thetas))

    for i, theta in enumerate(thetas):
        e_temp = np.cos(theta) * e_fac.data[:, 0]
        e_temp += np.sin(theta) * e_fac.data[:, 1]

        phi_temp = ts_scalar(e_xyz.time.data, np.cumsum(e_temp) * dt_e_fac)
        phi_temp = time_clip(phi_temp, tints)
        phi_temp -= np.mean(phi_temp)

        corrs[i] = np.corrcoef(phi_bs.data, phi_temp.data)

    corrpos = np.argmax(corrs)
    e_best = np.cos(thetas[corrpos]) * e_fac.data[:, 0]
    e_best += np.sin(thetas[corrpos]) * e_fac.data[:, 1]
    e_best = ts_scalar(e_xyz.time.data, e_best)
    phi_best = ts_scalar(e_xyz.time.data, np.cumsum(e_best) * dt_e_fac)
    phi_best = time_clip(phi_best, tints)
    phi_best -= np.mean(phi_best)
    theta_best = thetas[corrpos]
    dir_best = bxrxb * np.cos(theta_best) + bxr * np.sin(theta_best)

    # Find best speed
    # Maximum velocity may need to be increased in rare cases
    vph_vec = np.linspace(1e1, 5e2, 491)
    corr_v = np.zeros(len(vph_vec))

    for i, vph in enumerate(vph_vec):
        phi_e_temp = phi_best.data * vph
        corr_v[i] = np.sum(np.abs(phi_e_temp - phi_bs.data) ** 2)

    corr_vpos = np.argmin(corr_v)
    phi_e_best = phi_best.data * vph_vec[corr_vpos]
    phi_e_best = ts_scalar(phi_bs.time.data, phi_e_best)
    v_best = vph_vec[corr_vpos]

    options = {"coords": [phi_bs.time, ["Ebest", "Bs"]], "dims": ["time", "comp"]}
    phi_eb = xr.DataArray(
        np.vstack([phi_e_best.data, phi_bs.data]).T,
        **options,
    )

    return phi_eb, v_best, dir_best, thetas, corrs
