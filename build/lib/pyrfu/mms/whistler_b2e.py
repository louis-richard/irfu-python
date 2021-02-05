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

from scipy import constants

from ..pyrf import plasma_calc


def whistler_b2e(b2, freq, theta_k, b_mag, n_e):
    """Computes electric field power as a function of frequency for whistler waves using magnetic
    field power and cold plasma theory.

    Parameters
    ----------
    b2 : xarray.DataArray
        Time series of the power of whistler magnetic field in nT^2 Hz^{-1}.

    freq : ndarray
        frequencies in Hz corresponding B2.

    theta_k : float
        wave-normal angle of whistler waves in radians.

    b_mag : xarray.DataArray
        Time series of the magnitude of the magnetic field in nT.

    n_e : xarray.DataArray
        Time series of the electron number density in cm^{-3}.

    Returns
    -------
    e2 : xarray.DataArray
        Time series of the electric field power.

    Examples
    --------
    >>> from pyrfu import mms
    >>> e_power = mms.whistler_b2e(b_power, freq, theta_k, b_mag, n_e)

    """

    # Calculate plasma parameters
    pparam = plasma_calc(b_mag, n_e, n_e, n_e, n_e)
    fpe, fce = [pparam.Fpe, pparam.Fce]

    speed_of_light = constants.speed_of_light

    # Check input
    if len(b2) != len(freq):
        raise IndexError("B2 and freq lengths do not agree!")

    # Calculate cold plasma parameters
    r = 1 - fpe ** 2 / (freq * (freq - fce))
    l = 1 - fpe ** 2 / (freq * (freq + fce))
    p = 1 - fpe ** 2 / freq ** 2
    d = 0.5 * (r - l)
    s = 0.5 * (r + l)

    n2 = r * l * np.sin(theta_k) ** 2
    n2 += p * s * (1 + np.cos(theta_k) ** 2)
    n2 -= np.sqrt((r * l - p * s) ** 2 * np.sin(theta_k) ** 4 + 4 * (p ** 2) * (d ** 2) * np.cos(
        theta_k) ** 2)
    n2 /= (2 * (s * np.sin(theta_k) ** 2 + p * np.cos(theta_k) ** 2))

    e_temp1 = (p - n2 * np.sin(theta_k) ** 2) ** 2. * ((d / (s - n2)) ** 2 + 1) + (
            n2 * np.cos(theta_k) * np.sin(theta_k)) ** 2

    e_temp2 = (d / (s - n2)) ** 2 * (p - n2 * np.sin(theta_k) ** 2) ** 2 + p ** 2 * np.cos(
        theta_k) ** 2

    e2 = (speed_of_light ** 2 / n2) * e_temp1 / e_temp2 * b2.data * 1e-12

    return e2
