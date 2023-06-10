#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import constants

# Local imports
from ..pyrf.plasma_calc import plasma_calc

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def whistler_b2e(b2, freq, theta_k, b_mag, n_e):
    r"""Computes electric field power as a function of frequency for whistler
    waves using magnetic field power and cold plasma theory.

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

    # Check input
    if len(b2) != len(freq):
        raise IndexError("B2 and freq lengths do not agree!")

    # Calculate cold plasma parameters
    rr = 1 - fpe**2 / (freq * (freq - fce))
    ll = 1 - fpe**2 / (freq * (freq + fce))
    pp = 1 - fpe**2 / freq**2
    dd = 0.5 * (rr - ll)
    ss = 0.5 * (rr + ll)

    n2 = rr * ll * np.sin(theta_k) ** 2
    n2 += pp * ss * (1 + np.cos(theta_k) ** 2)
    n2 -= np.sqrt(
        (rr * ll - pp * ss) ** 2 * np.sin(theta_k) ** 4
        + 4 * (pp**2) * (dd**2) * np.cos(theta_k) ** 2,
    )
    n2 /= 2 * (ss * np.sin(theta_k) ** 2 + pp * np.cos(theta_k) ** 2)

    e_temp1 = (pp - n2 * np.sin(theta_k) ** 2) ** 2.0 * ((dd / (ss - n2)) ** 2 + 1) + (
        n2 * np.cos(theta_k) * np.sin(theta_k)
    ) ** 2
    e_temp2 = (dd / (ss - n2)) ** 2 * (
        pp - n2 * np.sin(theta_k) ** 2
    ) ** 2 + pp**2 * np.cos(theta_k) ** 2

    e2 = (constants.speed_of_light**2 / n2) * e_temp1 / e_temp2 * b2.data
    e2 *= 1e-12

    return e2
