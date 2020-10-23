#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
whistler_b2e.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from astropy import constants

from ..pyrf.plasma_calc import plasma_calc


def whistler_b2e(b2=None, freq=None, theta_k=None, b_mag=None, n_e=None):
    """Computes electric field power as a function of frequency for whistler waves using magnetic field power and cold
    plasma theory.

    Parameters
    ----------
    b2 : xarray.DataArray
        Time series of the power of whistler magnetic field in nT^2 Hz^{-1}.

    freq : numpy.ndarray
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

    assert b2 is not None and isinstance(b2, xr.DataArray)
    assert freq is not None and isinstance(freq, np.ndarray)
    assert theta_k is not None and isinstance(theta_k, float)
    assert b_mag is not None and isinstance(b_mag, xr.DataArray)
    assert n_e is not None and isinstance(n_e, xr.DataArray)

    # Calculate plasma parameters
    pparam = plasma_calc(b_mag, n_e, n_e, n_e, n_e)
    fpe, fce = [pparam.Fpe, pparam.Fce]

    c = constants.c.value

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

    e2 = (c ** 2 / n2) * e_temp1 / e_temp2 * b2 * 1e-12

    return e2
