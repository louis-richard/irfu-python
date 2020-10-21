#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
estimate_phase_speed.py

@author : Louis RICHARD
"""

import numpy as np

from scipy.optimize import curve_fit


def estimate_phase_speed(f_k_power=None, f=None, k=None, f_min=100.):
    """Simple function to estimate the phase speed from the frequency wave number power spectrum.
    Fits :math:`f = v k/ 2 \\pi` to the power spectrum.

    Parameters
    ----------
    f_k_power : numpy.ndarray
        2D array of powers.

    f : numpy.ndarray
        1D array of frequencies.

    k : numpy.ndarray
        1D array of wave numbers.

    f_min : float or int, optional
        Set low frequency threshold of points used to estimate the speed. Default ``f_min`` = 100

    Returns
    -------
    vph : float
        Estimated phase speed by fitting linear dispersion relation to data.

    Notes
    -----
    Draft version but seems to work well. Does not yet handle multiple modes in the same power
    spectrum.

    See also
    --------
    pyrfu.mms.fk_power_spectrum : Calculates the frequency-wave number power spectrum
    pyrfu.mms.probe_align_times : Returns times when f-a electrostatic waves can be characterized.

    """

    assert f_k_power is not None and isinstance(f_k_power, np.ndarray)
    assert f is not None and isinstance(f, np.ndarray)
    assert k is not None and isinstance(k, np.ndarray)
    assert isinstance(f_min, float)

    # Remove spurious points; specifically at large k.
    k_max = 2.0 * np.max(k) / 3.0
    power_temp = f_k_power
    rm_k = np.where(abs(k) > k_max)
    rm_f = np.where(f < f_min)

    power_temp[:, rm_k] = 0.0
    power_temp[rm_f, :] = 0.0
    power_temp[np.isnan(power_temp)] = 0.0

    # Find position of maximum power to guess vph
    [k_mat, f_mat] = np.meshgrid(k, f)
    max_pos = np.unravel_index(np.argmax(power_temp), power_temp.shape)

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

    p_power, f_power, k_power = [power_temp[highppos], f_mat[highppos], k_mat[highppos]]

    p_power2 = []
    f_power2 = []
    k_power2 = []

    for ii, pp in enumerate(p_power):
        if np.abs(vph_range[0]) < np.abs(f_power[ii] / k_power[ii]) < np.abs(vph_range[1]):
            p_power2.append(pp)
            f_power2.append(f_power[ii])
            k_power2.append(k_power[ii])

    p_power2, f_power2, k_power2 = [np.array(p_power2), np.array(f_power2), np.array(k_power2)]

    weights = 1 + np.log10(p_power2 / np.max(p_power2))

    def fun(x, a):
        return a * x

    popt, _ = curve_fit(fun, k_power2, f_power2, p0=vph_guess, sigma=weights)
    vph = popt[0] * 2 * np.pi

    return vph
