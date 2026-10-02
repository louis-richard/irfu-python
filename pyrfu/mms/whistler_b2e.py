#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from ..pyrf.resample import resample

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _plasma_frequencies(b_mag, n_e, ref):
    # Electron plasma and cyclotron frequencies [Hz], as columns for a
    # spectrogram reference
    if isinstance(ref, xr.DataArray) and ref.ndim == 2:
        b_mag, n_e = [
            (
                resample(inp, ref).data[:, None]
                if isinstance(inp, xr.DataArray)
                else np.asarray(inp, dtype=float)
            )
            for inp in [b_mag, n_e]
        ]
    else:
        b_mag, n_e = [np.asarray(inp, dtype=float) for inp in [b_mag, n_e]]

    q_e, m_e, ep0 = constants.e, constants.m_e, constants.epsilon_0
    f_pe = np.sqrt(1e6 * n_e * q_e**2 / (m_e * ep0)) / (2 * np.pi)
    f_ce = q_e * 1e-9 * b_mag / m_e / (2 * np.pi)

    return f_pe, f_ce


def whistler_b2e(b2, freq, theta_k, b_mag, n_e):
    r"""Computes electric field power as a function of frequency for whistler
    waves using magnetic field power and cold plasma theory.

    As mms.whistlerBtoE in irfu-matlab, with either one spectrum (b2 and freq
    1-D arrays, b_mag and n_e scalars) or a spectrogram (b2 a time series of
    spectra, b_mag and n_e scalars or time series resampled to b2).

    Parameters
    ----------
    b2 : array_like or xarray.DataArray
        Power of whistler magnetic field in nT^2 Hz^{-1}, as a function of
        frequency, or a time series of it (time, frequency).
    freq : array_like
        frequencies in Hz corresponding B2.
    theta_k : float
        wave-normal angle of whistler waves in radians.
    b_mag : float or xarray.DataArray
        Magnitude of the magnetic field in nT, or time series of it.
    n_e : float or xarray.DataArray
        Electron number density in cm^{-3}, or time series of it.

    Returns
    -------
    e2 : ndarray or xarray.DataArray
        Electric field power in (mV/m)^2 Hz^{-1}, with the shape (and for a
        spectrogram the coordinates) of b2.

    Raises
    ------
    IndexError
        If the lengths of b2 and freq do not agree.

    Examples
    --------
    >>> from pyrfu import mms
    >>> e_power = mms.whistler_b2e(b_power, freq, theta_k, b_mag, n_e)

    """

    freq = np.asarray(freq, dtype=float)
    is_spectrogram = isinstance(b2, xr.DataArray) and b2.ndim == 2
    b2_data = b2.data if isinstance(b2, xr.DataArray) else np.asarray(b2)

    # Check input
    if b2_data.shape[-1] != len(freq):
        raise IndexError("B2 and freq lengths do not agree!")

    # Calculate plasma parameters
    fpe, fce = _plasma_frequencies(b_mag, n_e, b2)

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

    e2 = (constants.speed_of_light**2 / n2) * e_temp1 / e_temp2 * b2_data
    e2 *= 1e-12

    if is_spectrogram:
        e2 = xr.DataArray(
            e2, coords=b2.coords, dims=b2.dims, attrs={"UNITS": "(mV/m)^2 Hz^-1"}
        )

    return e2
