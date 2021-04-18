#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import bisect
import numpy as np
import xarray as xr

from ..pyrf import resample, avg_4sc, time_clip, wavelet


def fk_power_spectrum_4sc(e, r, b, tints, cav=8, num_k=500, num_f=200, df=None,  w_width=1,
                          f_range=None):
    """Calculates the frequency-wave number power spectrum using the four MMS spacecraft. Uses a
    generalization of mms.fk_powerspectrum. Wavelet based cross-spectral analysis is used to
    calculate the phase difference each spacecraft pair and determine 3D wave vector. A
    generalization of the method used in mms.fk_powerspectrum to four point measurements.

    Parameters
    ----------
    e : list of xarray.DataArray
        Fields to apply 4SC cross-spectral analysis to. e.g., e or b fields (if multiple
        components only the first is used).

    r : list of xarray.DataArray
        Positions of the four spacecraft.

    b : list of xarray.DataArray
        background magnetic field in the same coordinates as r. Used to determine the parallel
        and perpendicular wave numbers using 4SC average.

    tints : list of str
        Time interval over which the power spectrum is calculated. To avoid boundary effects use
        a longer time interval for e and b.

    cav : int, optional
        Number of points in time series used to estimate phase. Default ``cav`` = 8.

    num_k : int, optional
        Number of wave numbers used in spectrogram. Default ``num_k`` = 500.

    df : float, optional
        Linear spacing of frequencies. Default ``df`` = None (log spacing).

    num_f : int, optional
        Number of frequencies used in spectrogram. Default ``num_f`` = 200.

    w_width : float, optional
        Multiplier for Morlet wavelet width. Default ``w_width`` = 1.

    f_range : list of float, optional
        Frequency range for k-k plots. [minf maxf].

    Returns
    -------
    out : xarray.Dataset
        Dataset of array of powers as a function of frequency and wavenumber. Power is normalized
        to the maximum value.

    Notes
    -----
    Wavelength must be larger than twice the spacecraft separations, otherwise spatial aliasing
    will occur.

    Examples
    --------
    >>> from pyrfu.mms import get_data, fk_power_spectrum_4sc
    >>> from pyrfu.pyrf import extend_tint, convert_fac

    Load data

    >>> tint_brst = ["2015-10-16T13:05:24.00", "2015-10-16T13:05:50.000"]
    >>> b_fgm_mms = [get_data("b_gse_fgm_brst_l2", tint_brst, i) for i in range(1, 5)]
    >>> b_scm_mms = [get_data("b_gse_scm_brst_l2", tint_brst, i) for i in range(1, 5)]

    Load spacecraft position

    >>> tint_long = extend_tint(tint, [-60, 60])
    >>> r_gse_mms = [get_data("r_gse", tint_long, i) for i in range(1, 5)]

    Convert magnetic field fluctuations to field aligned coordinates

    >>> b_scm_fac = [convert_fac(b_scm, b_fgm) for b_scm, b_fgm in zip(b_scm_mms, b_fgm_mms)]
    >>> b_scm_par = [b_scm[:, 0] for b_scm in b_scm_fac]

    Compute dispersion relation

    >>> tint = ["2015-10-16T13:05:26.500", "2015-10-16T13:05:27.000"]
    >>> pwer = fk_power_spectrum_4sc(b_scm_par, r_gse_mms, b_fgm_mms, tint, 4, 500, 2, 10, 2)

    """

    ic = np.arange(1, 5)

    e = [resample(e[i - 1], e[0]) for i in ic]
    r = [resample(r[i - 1], e[0]) for i in ic]
    b = [resample(b[i - 1], e[0]) for i in ic]

    b_avg = avg_4sc(b)

    times = e[0].time
    use_linear = df is not None

    idx = time_clip(e[0].time, tints)

    # If odd, remove last data point (as is done in irf_wavelet)
    if len(idx) % 2:
        idx = idx[:-1]

    if use_linear:
        w = []
        for i in range(4):
            cwt_options = dict(linear=df, returnpower=False, wavelet_width=5.36 * w_width)
            w.append(wavelet(e[i], **cwt_options))
    else:
        w = []
        for i in range(4):
            cwt_options = dict(nf=num_f, returnpower=False, wavelet_width=5.36 * w_width)
            w.append(wavelet(e[i], **cwt_options))

    num_f = len(w[0].frequency)

    times = time_clip(times, tints)
    nt = len(times)

    w = [time_clip(w[i], tints) for i in range(4)]

    fk_power = 0
    for i in range(4):
        fk_power += w[i].data * np.conj(w[i].data) / 4

    n = int(np.floor(nt/cav)-1)
    pos_av = cav / 2 + np.arange(n) * cav
    av_times = times[pos_av.astype(int)]

    b_avg = resample(b_avg, av_times)

    r = [resample(r[i], av_times) for i in range(4)]

    cx12, cx13, cx14 = [np.zeros((n, num_f), dtype="complex128") for _ in range(3)]
    cx23, cx24, cx34 = [np.zeros((n, num_f), dtype="complex128") for _ in range(3)]

    power_avg = np.zeros((n, num_f), dtype="complex128")

    for m, pos_avm in enumerate(pos_av):
        lb, ub = [int(pos_avm - cav / 2 + 1), int(pos_avm + cav / 2)]

        cx12[m, :] = np.nanmean(w[0].data[lb:ub, :] * np.conj(w[1].data[lb:ub, :]), axis=0)
        cx13[m, :] = np.nanmean(w[0].data[lb:ub, :] * np.conj(w[2].data[lb:ub, :]), axis=0)
        cx14[m, :] = np.nanmean(w[0].data[lb:ub, :] * np.conj(w[3].data[lb:ub, :]), axis=0)
        cx23[m, :] = np.nanmean(w[1].data[lb:ub, :] * np.conj(w[2].data[lb:ub, :]), axis=0)
        cx24[m, :] = np.nanmean(w[1].data[lb:ub, :] * np.conj(w[3].data[lb:ub, :]), axis=0)
        cx34[m, :] = np.nanmean(w[2].data[lb:ub, :] * np.conj(w[3].data[lb:ub, :]), axis=0)

        power_avg[m, :] = np.nanmean(fk_power[lb:ub, :], axis=0)

    # Compute phase differences between each spacecraft pair
    th12 = np.arctan2(np.imag(cx12), np.real(cx12))
    th13 = np.arctan2(np.imag(cx13), np.real(cx13))
    th14 = np.arctan2(np.imag(cx14), np.real(cx14))
    th23 = np.arctan2(np.imag(cx23), np.real(cx23))
    th24 = np.arctan2(np.imag(cx24), np.real(cx24))
    th34 = np.arctan2(np.imag(cx34), np.real(cx34))

    w_mat = 2 * np.pi * np.tile(w[0].frequency.data, (n, 1))

    # Convert phase difference to time delay
    dt12, dt13, dt14, dt23, dt24, dt34 = [th / w_mat for th in [th12, th13, th14, th23, th24, th34]]

    # Weighted averaged time delay using all spacecraft pairs
    dt2 = 0.5 * dt12 + 0.2 * (dt13 - dt23) + 0.2 * (dt14 - dt24) + 0.1 * (dt14 - dt34 - dt23)
    dt3 = 0.5 * dt13 + 0.2 * (dt12 + dt23) + 0.2 * (dt14 - dt34) + 0.1 * (dt12 + dt24 - dt34)
    dt4 = 0.5 * dt14 + 0.2 * (dt12 + dt24) + 0.2 * (dt13 + dt34) + 0.1 * (dt12 + dt23 + dt34)

    # Compute phase speeds
    r = [r[i].data for i in range(4)]

    k_x, k_y, k_z = [np.zeros((n, num_f)) for _ in range(3)]

    # Volumetric tensor with SC1 as center.
    dr = np.reshape(np.hstack(r[1:]), (n, 3, 3)) - np.reshape(np.tile(r[0], (1, 3)), (n, 3, 3))
    dr = np.transpose(dr, [0, 2, 1])

    # Delay tensor with SC1 as center.
    # dT = np.reshape(np.hstack([dt2,dt3,dt4]),(N,num_f,3))
    tau = np.dstack([dt2, dt3, dt4])

    for ii in range(num_f):
        m = np.linalg.solve(dr, np.squeeze(tau[:, ii, :]))

        k_x[:, ii] = 2 * np.pi * w[0].frequency[ii].data * m[:, 0]
        k_y[:, ii] = 2 * np.pi * w[0].frequency[ii].data * m[:, 1]
        k_z[:, ii] = 2 * np.pi * w[0].frequency[ii].data * m[:, 2]

    k_x, k_y, k_z = [k / 1e3 for k in [k_x, k_y, k_z]]

    k_mag = np.linalg.norm(np.array([k_x, k_y, k_z]), axis=0)

    b_avg_x_mat = np.tile(b_avg.data[:, 0], (num_f, 1)).T
    b_avg_y_mat = np.tile(b_avg.data[:, 1], (num_f, 1)).T
    b_avg_z_mat = np.tile(b_avg.data[:, 2], (num_f, 1)).T

    b_avg_abs = np.linalg.norm(b_avg, axis=1)
    b_avg_abs_mat = np.tile(b_avg_abs, (num_f, 1)).T

    k_par = (k_x * b_avg_x_mat + k_y * b_avg_y_mat + k_z * b_avg_z_mat) / b_avg_abs_mat
    k_perp = np.sqrt(k_mag ** 2 - k_par ** 2)

    k_max = np.max(k_mag) * 1.1
    k_min = -k_max
    k_vec = np.linspace(-k_max, k_max, num_k)
    k_mag_vec = np.linspace(0, k_max, num_k)

    dk_mag = k_max / num_k
    dk = 2 * k_max / num_k

    # Sort power into frequency and wave vector
    print("notice : Computing power versus kx,f; ky,f, kz,f")
    power_k_x_f, power_k_y_f, power_k_z_f = [np.zeros((num_f, num_k)) for _ in range(3)]
    power_k_mag_f = np.zeros((num_f, num_k))

    for nn in range(num_f):
        k_x_number = np.floor((k_x[:, nn] - k_min) / dk).astype(int)
        k_y_number = np.floor((k_y[:, nn] - k_min) / dk).astype(int)
        k_z_number = np.floor((k_z[:, nn] - k_min) / dk).astype(int)
        k_number = np.floor((k_mag[:, nn]) / dk_mag).astype(int)

        power_k_x_f[nn, k_x_number] += np.real(power_avg[:, nn])
        power_k_y_f[nn, k_y_number] += np.real(power_avg[:, nn])
        power_k_z_f[nn, k_z_number] += np.real(power_avg[:, nn])

        power_k_mag_f[nn, k_number] += np.real(power_avg[:, nn])

    # power_k_x_f[power_k_x_f == 0] 	= np.nan
    # power_k_y_f[power_k_y_f == 0] 	= np.nan
    # power_k_z_f[power_k_z_f == 0] 	= np.nan
    # power_k_mag_f[power_k_mag_f == 0] = np.nan

    power_k_x_f /= np.max(power_k_x_f)
    power_k_y_f /= np.max(power_k_y_f)
    power_k_z_f /= np.max(power_k_z_f)
    power_k_mag_f /= np.max(power_k_mag_f)

    # power_k_x_f[power_k_x_f < 1.0e-6] 		= 1e-6
    # power_k_y_f[power_k_y_f < 1.0e-6] 		= 1e-6
    # power_k_z_f[power_k_z_f < 1.0e-6] 		= 1e-6
    # power_k_mag_f[power_k_mag_f < 1.0e-6] 	= 1e-6

    frequencies = w[0].frequency.data
    idx_f = np.arange(num_f)

    if f_range is not None:
        idx_min_freq = bisect.bisect_left(frequencies, np.min(f_range))
        idx_max_freq = bisect.bisect_left(frequencies, np.max(f_range))
        idx_f = idx_f[idx_min_freq:idx_max_freq]

    print("notice : Computing power versus kx,ky; kx,kz; ky,kz\n")
    power_k_x_k_y = np.zeros((num_k, num_k))
    power_k_x_k_z = np.zeros((num_k, num_k))
    power_k_y_k_z = np.zeros((num_k, num_k))
    power_k_perp_k_par = np.zeros((num_k, num_k))

    for nn in idx_f:
        k_x_number = np.floor((k_x[:, nn] - k_min) / dk).astype(int)
        k_y_number = np.floor((k_y[:, nn] - k_min) / dk).astype(int)
        k_z_number = np.floor((k_z[:, nn] - k_min) / dk).astype(int)

        k_par_number = np.floor((k_par[:, nn] - k_min) / dk).astype(int)
        k_perp_number = np.floor((k_perp[:, nn]) / dk_mag).astype(int)

        power_k_x_k_y[k_y_number, k_x_number] += np.real(power_avg[:, nn])
        power_k_x_k_z[k_z_number, k_x_number] += np.real(power_avg[:, nn])
        power_k_y_k_z[k_z_number, k_y_number] += np.real(power_avg[:, nn])

        power_k_perp_k_par[k_par_number, k_perp_number] += np.real(power_avg[:, nn])

    # power_k_x_k_y[power_k_x_k_y == 0] = np.nan
    # power_k_x_k_z[power_k_x_k_z == 0] = np.nan
    # power_k_y_k_z[power_k_y_k_z == 0] = np.nan
    # power_k_perp_k_par[power_k_perp_k_par == 0] = np.nan

    power_k_x_k_y /= np.max(power_k_x_k_y)
    power_k_x_k_z /= np.max(power_k_x_k_z)
    power_k_y_k_z /= np.max(power_k_y_k_z)
    power_k_perp_k_par /= np.max(power_k_perp_k_par)

    # power_k_x_k_y(power_k_x_k_y < 1.0e-6) 				= 1e-6
    # power_k_x_k_z(power_k_x_k_z < 1.0e-6) 				= 1e-6
    # power_k_y_k_z(power_k_y_k_z < 1.0e-6) 				= 1e-6
    # power_k_perp_k_par[power_k_perp_k_par < 1.0e-6] 	= 1e-6

    out_dict = {"k_x_f": (["k_x", "f"], power_k_x_f.T), "k_y_f": (["k_x", "f"], power_k_y_f.T),
                "k_z_f": (["k_x", "f"], power_k_z_f.T),
                "k_mag_f": (["k_mag", "f"], power_k_mag_f.T),
                "k_x_k_y": (["k_x", "k_y"], power_k_x_k_y.T),
                "k_x_k_z": (["kx", "kz"], power_k_x_k_z.T),
                "k_y_k_z": (["k_y", "k_z"], power_k_y_k_z.T),
                "k_perp_k_par": (["k_perp", "k_par"], power_k_perp_k_par.T),
                "k_x": k_vec, "k_y": k_vec, "k_z": k_vec, "k_mag": k_mag_vec, "k_perp": k_mag_vec,
                "k_par": k_vec,
                "f": frequencies}

    out = xr.Dataset(out_dict)

    return out
