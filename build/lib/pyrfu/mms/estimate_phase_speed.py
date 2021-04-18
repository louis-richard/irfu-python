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

import numpy as np

from scipy.optimize import curve_fit


def estimate_phase_speed(f_k_power, freq, k, f_min=100.):
    """Simple function to estimate the phase speed from the frequency
    wave number power spectrum. Fits :math:`f = v k/ 2 \\pi` to the
    power spectrum.

    Parameters
    ----------
    f_k_power : ndarray
        2D array of powers.

    freq : ndarray
        1D array of frequencies.

    k : ndarray
        1D array of wave numbers.

    f_min : float, optional
        Set low frequency threshold of points used to estimate the
        speed. Default ``f_min`` = 100.

    Returns
    -------
    vph : float
        Estimated phase speed by fitting linear dispersion relation to
        data.

    Notes
    -----
    Draft version but seems to work well. Does not yet handle multiple
    modes in the same power spectrum.

    See also
    --------
    pyrfu.mms.fk_power_spectrum : Calculates the frequency-wave number
                                    power spectrum
    pyrfu.mms.probe_align_times : Returns times when f-a electrostatic
                                    waves can be characterized.

    """

    # Remove spurious points; specifically at large k.
    k_max = 2.0 * np.max(k) / 3.0
    power_temp = f_k_power
    rm_k = np.where(abs(k) > k_max)
    rm_f = np.where(freq < f_min)

    power_temp[:, rm_k] = 0.0
    power_temp[rm_f, :] = 0.0
    power_temp[np.isnan(power_temp)] = 0.0

    # Find position of maximum power to guess vph
    max_pos = np.unravel_index(np.argmax(power_temp), power_temp.shape)

    k_mat, f_mat = np.meshgrid(k, freq)
    max_f, max_k = [f_mat[max_pos], k_mat[max_pos]]

    # Initial guess
    vph_guess = max_f / max_k

    if vph_guess > 0.0:
        power_temp[:, k < 0.0] = 0
    else:
        power_temp[:, k > 0.0] = 0

    vph_range = [vph_guess / 3, vph_guess * 3]

    # Find all points around this guess vph
    highppos = np.where(power_temp > 0.3 * np.max(power_temp))

    p_power = power_temp[highppos]
    f_power = f_mat[highppos]
    k_power = k_mat[highppos]

    p_power2, f_power2, k_power2 = [[], [], []]

    for i, p_pow in enumerate(p_power):
        if np.abs(vph_range[0]) < np.abs(f_power[i] / k_power[i]) \
                < np.abs(vph_range[1]):
            p_power2.append(p_pow)
            f_power2.append(f_power[i])
            k_power2.append(k_power[i])

    p_power2 = np.array(p_power2)
    f_power2 = np.array(f_power2)
    k_power2 = np.array(k_power2)

    weights = 1 + np.log10(p_power2 / np.max(p_power2))

    def fun(k_pow, f_pow):
        return f_pow * k_pow

    res = curve_fit(fun, k_power2, f_power2, p0=vph_guess, sigma=weights)
    vph = res[0][0] * 2 * np.pi

    return vph
