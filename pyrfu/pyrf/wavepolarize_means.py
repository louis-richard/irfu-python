#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np
from scipy import ndimage

# Local imports
from .resample import resample
from .ts_spectr import ts_spectr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)

# Smoothing profile of the spectral matrix in frequency: normalised 7 points
# Hamming window, of which irfu-matlab uses the values rounded to 3 decimals
# (0.024, 0.093, 0.232, 0.301, ...), as pyspedas wavpol
SMOOTH_FREQ = np.hamming(7) / np.sum(np.hamming(7))


def _mfa(b_wave, b_bgd):
    # Wave magnetic field in mean field aligned coordinates (perp1, perp2, par)
    n_b = b_bgd / np.linalg.norm(b_bgd, axis=1, keepdims=True)
    n_perp1 = np.cross(n_b, [0.0, 1.0, 0.0])
    n_perp1 /= np.linalg.norm(n_perp1, axis=1, keepdims=True)
    n_perp2 = np.cross(n_b, n_perp1)
    return np.stack(
        [np.sum(b_wave * vec, axis=1) for vec in [n_perp1, n_perp2, n_b]], axis=1
    )


def _rotation_angle(upper, lower):
    # Angle of the rotation that makes the state vector real. ATAN(upper,
    # lower) in the original IDL wavpol is a two-argument arctangent; irfu-matlab
    # uses atan(upper / lower), which is wrong when lower < 0.
    angle = np.arctan2(upper, lower)
    return np.where(upper > 0, angle, 2 * np.pi + angle)


def _helicity_ellipticity(e_spec, wave_angle):
    # Helicity and ellipticity from the wave state vectors built from each
    # row of the smoothed spectral matrix, averaged over the three rows
    sqrt_diag = [np.sqrt(np.real(e_spec[..., i, i])) for i in range(3)]
    rows = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
    helicity, ellipticity = [np.zeros(e_spec.shape[:2]) for _ in range(2)]
    sign = -np.sign(np.imag(e_spec[..., 0, 1]) * np.sin(wave_angle))

    for i, (i_d, i_1, i_2) in enumerate(rows):
        lambda_u = np.stack(
            [
                sqrt_diag[i_d] + 0j,
                np.conj(e_spec[..., i_d, i_1]) / sqrt_diag[i_d],
                np.conj(e_spec[..., i_d, i_2]) / sqrt_diag[i_d],
            ],
            axis=-1,
        )

        # Helicity
        upper = np.sum(2 * np.real(lambda_u) * np.imag(lambda_u), axis=-1)
        lower = np.sum(np.real(lambda_u) ** 2 - np.imag(lambda_u) ** 2, axis=-1)
        gamma = _rotation_angle(upper, lower)
        lambda_u = np.exp(-0.5j * gamma)[..., None] * lambda_u
        helicity += np.linalg.norm(np.imag(lambda_u), axis=-1) / np.linalg.norm(
            np.real(lambda_u), axis=-1
        )

        # Ellipticity
        lam = lambda_u[..., :2]
        upper = np.sum(np.imag(lam) * np.real(lam), axis=-1)
        lower = np.sum(np.real(lam) ** 2 - np.imag(lam) ** 2, axis=-1)
        gamma_rot = _rotation_angle(upper, lower)
        lam = np.exp(-0.5j * gamma_rot)[..., None] * lam
        ellip = np.linalg.norm(np.imag(lam), axis=-1)
        ellipticity += sign * ellip / np.linalg.norm(np.real(lam), axis=-1)

    return helicity / 3, ellipticity / 3


def wavepolarize_means(
    b_wave,
    b_bgd,
    min_psd: float = 1e-25,
    nop_fft: int = 256,
):
    r"""Analysis the polarization of magnetic wave using "means" method

    Parameters
    ----------
    b_wave : xarray.DataArray
        Time series of the magnetic field from Search Coil Magnetometer (SCM).
    b_bgd : xarray.DataArray
        Time series of the magnetic field from Flux Gate Magnetometer (FGM).
    min_psd : float, Optional
        Threshold for the analysis (e.g 1.0e-7). Below this value, the SVD
        analysis is meaningless if min_psd is not given, SVD analysis will
        be done for all waves. Default ``min_psd`` = 1e-25.
    nop_fft : int, Optional
        Number of points in FFT. Default is 256.

    Returns
    -------
    b_psd : xarray.DataArray
        Power spectrum density of magnetic filed wave [nT^2 Hz^-1 for B in
        nT].
    wave_angle : xarray.DataArray
        Spectrogram of the wave normal angle in degrees (form 0 to 90)
    deg_pol : xarray.DataArray
        Spectrogram of the degree of polarization (form 0 to 1).
    ellipticity : xarray.DataArray
        Spectrogram of the ellipticity (form -1 to 1)
    helicity : xarray.DataArray
        Spectrogram of the helicity (form -1 to 1)

    Notes
    -----
    Port of irf_wavepolarize_means.m (H. Fu), with the following differences:
    the FFT windows advance by half a window (irfu-matlab shifts the data and
    the window start, so that the windows advance by a full window and wrap
    around the end of the data), all the full windows are used, the time
    tags are at the window centres, and the frequencies are those of the FFT
    bins (irfu-matlab labels bin k with frequency k instead of k - 1 times
    the bin width). The rotation angles of the state vectors use a two-argument
    arctangent, as in the original IDL wavpol (irfu-matlab uses a one-argument
    arctangent, which is wrong when the denominator is negative), and the
    window and frequency smoothing are those of IDL and pyspedas wavpol, with
    which the results agree to rounding.

    ``b_wave`` and ``b_bgd`` should be from the same satellite and in the same
    coordinates

    .. warning::
        If one component is an order of magnitude or more  greater than the
        other two then the polarization results saturate and erroneously
        indicate high degrees of polarization at all times and frequencies.
        Time series should be eyeballed before running the program. For time
        series containing very rapid changes or spikes the usual problems
        with Fourier analysis arise. Care should be taken in evaluating
        degree of polarization results. For meaningful results there should
        be significant wave power at the frequency where the polarization
        approaches 100%. Remember comparing two straight lines yields 100%
        polarization.

    Examples
    --------
    >>> from pyrfu import pyrf
    >>> polarization = pyrf.wavepolarize_means(b_wave, b_bgd)
    >>> polarization = pyrf.wavepolarize_means(b_wave, b_bgd, 1.0e-7)
    >>> polarization = pyrf.wavepolarize_means(b_wave, b_bgd, 1.0e-7, 256)

    """

    step_length = nop_fft // 2
    n_half = nop_fft // 2
    n_pts = len(b_wave)
    # total number of FFTs
    n_stp = (n_pts - nop_fft) // step_length + 1

    if n_stp < 1:
        raise ValueError("b_wave must be longer than nop_fft")

    # change wave to MFA coordinates
    b_bgd = resample(b_bgd, b_wave)
    b_mfa = _mfa(b_wave.data, b_bgd.data)

    time = b_wave.time.data.astype("datetime64[ns]")
    d_t = np.diff(time) / np.timedelta64(1, "s")
    samp_freq = 1 / d_t[0]

    if not np.isclose(d_t[0], d_t[-1]):
        logger.warning(
            "file sampling frequency changes %g Hz to %g Hz", samp_freq, 1 / d_t[-1]
        )

    # FFT CALCULATION
    # Hamming window over samples 0 to nop_fft - 1, as in IDL/pyspedas wavpol
    # (irfu-matlab uses samples 1 to nop_fft)
    smooth = 0.08 + 0.46 * (1 - np.cos(2 * np.pi * np.arange(nop_fft) / nop_fft))
    idx = step_length * np.arange(n_stp)[:, None] + np.arange(nop_fft)[None, :]
    half_spec = np.fft.fft(smooth[None, :, None] * b_mfa[idx], axis=1)[:, :n_half]

    # CALCULATION OF THE SPECTRAL MATRIX, m[r, c] = conj(spec_r) * spec_c
    spec_mat = np.conj(half_spec)[..., :, None] * half_spec[..., None, :]

    # CALCULATION OF SMOOTHED SPECTRAL MATRIX (NaN within 3 bins of the edges)
    n_side = len(SMOOTH_FREQ) // 2
    inner = slice(n_side, n_half - n_side)
    e_spec = np.full(spec_mat.shape, np.nan + 0j)
    e_spec[:, inner] = ndimage.correlate1d(np.real(spec_mat), SMOOTH_FREQ, axis=1)[
        :, inner
    ]
    e_spec[:, inner] += (
        1j * ndimage.correlate1d(np.imag(spec_mat), SMOOTH_FREQ, axis=1)[:, inner]
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        # CALCULATION OF THE MINIMUM VARIANCE DIRECTION AND WAVENORMAL ANGLE
        im_12, im_13, im_23 = [
            np.imag(e_spec[..., i, j]) for i, j in [(0, 1), (0, 2), (1, 2)]
        ]
        wave_angle = np.arctan(np.hypot(im_23, im_13) / np.abs(im_12))

        # CALCULATION OF THE DEGREE OF POLARISATION
        trace_sqrd = np.real(np.einsum("...ij,...ji->...", e_spec, e_spec))
        trace_spec = np.real(np.einsum("...ii->...", e_spec))
        deg_pol = (3 * trace_sqrd - trace_spec**2) / (2 * trace_spec**2)

        # CALCULATION OF HELICITY, ELLIPTICITY AND THE WAVE STATE VECTOR
        helicity, ellipticity = _helicity_ellipticity(e_spec, wave_angle)

    # CREATING OUTPUT PARAMETER
    centres = step_length * np.arange(n_stp) + n_half
    time_line = time[0] + np.round(centres / samp_freq * 1e9).astype("timedelta64[ns]")
    bin_width = samp_freq / nop_fft
    freq_line = bin_width * np.arange(n_half)

    # scaling power results to units with meaning
    power_spec = 2 * trace_spec / (nop_fft * np.sum(smooth**2) * bin_width)
    power_spec[:, [0, -1]] /= 2

    # KICK OUT THE ANALYSIS OF THE WEAK SIGNALS
    weak = power_spec < min_psd
    for out in [wave_angle, deg_pol, ellipticity, helicity]:
        out[weak] = np.nan

    # Save as DataArrays
    b_psd, wave_angle, deg_pol, ellipticity, helicity = [
        ts_spectr(time_line, freq_line, out, "frequency")
        for out in [
            power_spec,
            np.rad2deg(wave_angle),
            deg_pol,
            ellipticity,
            helicity,
        ]
    ]

    return b_psd, wave_angle, deg_pol, ellipticity, helicity
