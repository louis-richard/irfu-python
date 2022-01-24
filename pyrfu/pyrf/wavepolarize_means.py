#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .resample import resample
from .datetime642unix import datetime642unix

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _spec_mat(half_spec_x, half_spec_y, half_spec_z):
    r"""CALCULATION OF THE SPECTRAL MATRIX"""

    spec_mat = np.zeros([len(half_spec_x), 3, 3])
    spec_mat[:, 0, 0] = half_spec_x * np.conj(half_spec_x)
    spec_mat[:, 1, 0] = half_spec_x * np.conj(half_spec_y)
    spec_mat[:, 2, 0] = half_spec_x * np.conj(half_spec_z)
    spec_mat[:, 0, 1] = half_spec_y * np.conj(half_spec_x)
    spec_mat[:, 1, 1] = half_spec_y * np.conj(half_spec_y)
    spec_mat[:, 2, 1] = half_spec_y * np.conj(half_spec_z)
    spec_mat[:, 0, 2] = half_spec_z * np.conj(half_spec_x)
    spec_mat[:, 1, 2] = half_spec_z * np.conj(half_spec_y)
    spec_mat[:, 2, 2] = half_spec_z * np.conj(half_spec_z)

    return spec_mat


def wavepolarize_means(b_wave, b_bgd, min_psd: float = 1e-25,
                       nop_fft: int = 256):
    r"""Analysis the polarization of magnetic wave using "means" method

    Parameters
    -----------
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
        Power spectrum density of magnetic filed wave.
    wave_angle : xarray.DataArray
        (form 0 to 90)
    deg_pol : xarray.DataArray
        Spectrogram of the degree of polarization (form 0 to 1).
    elliptict : xarray.DataArray
        Spectrogram of the ellipticity (form -1 to 1)
    helict : xarray.DataArray
        Spectrogram of the helicity (form -1 to 1)

    Notes
    ------
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

    step_length = int(nop_fft / 2)
    n_pts = len(b_wave)
    # total number of FFTs
    n_stp = int((n_pts - nop_fft) / step_length)
    # No. of bins in frequency domain
    n_bin = 7
    # Smoothing profile based on Hanning
    aa = np.array([.024, .093, .232, .301, .232, .093, .024])

    # change wave to MFA coordinates
    b_bgd = resample(b_bgd, b_wave)

    b_x, b_y, b_z = [np.zeros(len(b_wave)) for _ in range(3)]

    for ii in range(len(b_wave)):
        nb = b_bgd[ii, :] / np.linalg.norm(b_bgd[ii, :])
        n_perp1 = np.cross(nb, [0, 1, 0])
        n_perp1 = n_perp1 / np.linalg.norm(n_perp1)
        n_perp2 = np.cross(nb, n_perp1)

        b_z[ii] = np.sum(b_wave[ii, :] * nb)
        b_x[ii] = np.sum(b_wave[ii, :] * n_perp1)
        b_y[ii] = np.sum(b_wave[ii, :] * n_perp2)

    ct = datetime642unix(b_wave.time.data)

    # DEFINE ARRAYS
    xs, ys, zs = [b_x, b_y, b_z]

    sample_freq = 1/(ct[1]-ct[0])
    end_sample_freq = 1/(ct[-1]-ct[-2])

    if sample_freq != end_sample_freq:
        warnings.warn(f"file sampling frequency changes {sample_freq} Hz to "
                      f"{end_sample_freq} Hz", UserWarning)
    else:
        print("ac file sampling frequency {} Hz".format(sample_freq))

    # FFT calculation
    # Minimum variance direction and wave normal angle
    wave_angle = np.zeros([n_stp, int(nop_fft / 2)])

    # Degree of Polarization
    sqrd_mat = np.zeros([n_stp, int(nop_fft / 2), 3, 3])
    deg_pol = np.zeros([n_stp, int(nop_fft / 2)])
    trace_spec_mat = np.zeros([n_stp, int(nop_fft / 2)])

    # HELICITY, ELLIPTICITY AND THE WAVE STATE VECTOR
    lambda_u = np.zeros([n_stp, int(nop_fft / 2), 3, 3])

    helic, ellip = [np.zeros([n_stp, int(nop_fft / 2), 3]) for _ in range(2)]

    smooth = np.zeros(nop_fft)

    for j in range(n_stp):
        # FFT CALCULATION
        smooth = 0.08 + 0.46 * (1 - np.cos(
            2 * np.pi * np.arange(1, nop_fft + 1) / nop_fft))
        temp_x = smooth * xs[((j - 1) * step_length + 1):(
                    (j - 1) * step_length + nop_fft)]
        temp_y = smooth * ys[((j - 1) * step_length + 1):(
                    (j - 1) * step_length + nop_fft)]
        temp_z = smooth * zs[((j - 1) * step_length + 1):(
                    (j - 1) * step_length + nop_fft)]

        spec_x = np.fft.fft(temp_x)
        spec_y = np.fft.fft(temp_y)
        spec_z = np.fft.fft(temp_z)

        half_spec_x = spec_x[:(nop_fft / 2)]
        half_spec_y = spec_y[:(nop_fft / 2)]
        half_spec_z = spec_z[:(nop_fft / 2)]

        xs = np.roll(xs, -step_length)
        ys = np.roll(ys, -step_length)
        zs = np.roll(zs, -step_length)

        # CALCULATION OF THE SPECTRAL MATRIX
        spec_mat = _spec_mat(half_spec_x, half_spec_y, half_spec_z)

        # Calculation of smoothed spectral matrix
        e_spec_mat = np.nan * np.ones(spec_mat.shape)

        off_idx = int((n_bin - 1) / 2)

        for k in range(off_idx, int(nop_fft / 2) - off_idx):
            for ir in range(3):
                for ic in range(3):
                    e_spec_mat[k, ir, ic] = np.sum(
                        aa[:n_bin] * spec_mat[(k - off_idx):(k + off_idx), ir, ic])

        # Calculation of the minimum variance direction and wave normal angle
        aaa2 = np.imag(e_spec_mat[:, 0, 1]) ** 2
        aaa2 += np.imag(e_spec_mat[:, 0, 2]) ** 2
        aaa2 += np.imag(e_spec_mat[:, 1, 2]) ** 2
        aaa2 = np.sqrt(aaa2[j, :])

        wn_x = -np.abs(np.imag(e_spec_mat[:, 1, 2]) / aaa2)
        wn_y = -np.abs(np.imag(e_spec_mat[:, 0, 2]) / aaa2)
        wn_z = np.imag(e_spec_mat[:, 0, 1]) / aaa2

        wave_angle[j, :] = np.arctan(np.sqrt(wn_x ** 2 + wn_y ** 2) / np.abs(wn_z))

        # CALCULATION OF THE DEGREE OF POLARISATION
        # calc of square of smoothed spec matrix
        for ir in range(3):
            for ic in range(3):
                sqrd_mat[:, ir, ic] = e_spec_mat[:, ir, 0] * e_spec_mat[:, 0,
                                                             ic]
                sqrd_mat[:, ir, ic] += e_spec_mat[:, ir, 1] * e_spec_mat[:, 1,
                                                              ic]
                sqrd_mat[:, ir, ic] += e_spec_mat[:, ir, 2] * e_spec_mat[:, 2,
                                                              ic]

        trace_sqrd_mat = sqrd_mat[:, 0, 0] + sqrd_mat[:, 1, 1] + sqrd_mat[:, 2, 2]
        trace_spec_mat[j, :] = e_spec_mat[:, 0, 0] + e_spec_mat[:, 1, 1] + e_spec_mat[:, 2, 2]

        deg_pol[j, :] = trace_spec_mat[j, :] * np.nan
        deg_pol[j, off_idx:int(nop_fft / 2) - off_idx] = 3 * trace_sqrd_mat[
                                                             off_idx:int(nop_fft / 2) - off_idx]
        deg_pol[j, off_idx:int(nop_fft / 2) - off_idx] -= trace_spec_mat[j,
                                                          off_idx:int(nop_fft / 2) - off_idx] ** 2
        deg_pol[j, off_idx:int(nop_fft / 2) - off_idx] /= 2 * trace_spec_mat[j, off_idx:int(
            nop_fft / 2) - off_idx] ** 2

        # Calculation of helicity, ellipticity and the wave state vector
        alpha_x = np.sqrt(e_spec_mat[:, 0, 0])
        alpha_y = np.sqrt(e_spec_mat[:, 1, 1])
        alpha_z = np.sqrt(e_spec_mat[:, 2, 2])

        alpha_cos1_x = np.real(e_spec_mat[:, 0, 1]) / np.sqrt(e_spec_mat[:, 0, 0])
        alpha_sin1_x = -np.imag(e_spec_mat[j, :, 0, 1]) / np.sqrt(e_spec_mat[:, 0, 0])
        alpha_cos2_x = np.real(e_spec_mat[:, 0, 2]) / np.sqrt(e_spec_mat[:, 0, 0])
        alpha_sin2_x = -np.imag(e_spec_mat[j, :, 0, 2]) / np.sqrt(e_spec_mat[:, 0, 0])

        alpha_cos1_y = np.real(e_spec_mat[:, 1, 0]) / np.sqrt(e_spec_mat[:, 1, 1])
        alpha_sin1_y = -np.imag(e_spec_mat[:, 1, 0]) / np.sqrt(e_spec_mat[:, 1, 1])
        alpha_cos2_y = np.real(e_spec_mat[:, 1, 2]) / np.sqrt(e_spec_mat[:, 1, 1])
        alpha_sin2_y = -np.imag(e_spec_mat[:, 1, 2]) / np.sqrt(e_spec_mat[:, 1, 1])

        alpha_cos1_z = np.real(e_spec_mat[:, 2, 0]) / np.sqrt(e_spec_mat[:, 2, 2])
        alpha_sin1_z = -np.imag(e_spec_mat[:, 2, 0]) / np.sqrt(e_spec_mat[:, 2, 2])
        alpha_cos2_z = np.real(e_spec_mat[:, 2, 1]) / np.sqrt(e_spec_mat[:, 2, 2])
        alpha_sin2_z = -np.imag(e_spec_mat[:, 2, 1]) / np.sqrt(e_spec_mat[:, 2, 2])

        lambda_u[:, 0, 0] = alpha_x
        lambda_u[:, 1, 0] = alpha_y
        lambda_u[:, 2, 0] = alpha_z

        lambda_u[:, 0, 1] = alpha_cos1_x + alpha_sin1_x * 1j
        lambda_u[:, 0, 2] = alpha_cos2_x + alpha_sin2_x * 1j

        lambda_u[:, 1, 1] = alpha_cos1_y + alpha_sin1_y * 1j
        lambda_u[:, 1, 2] = alpha_cos2_y + alpha_sin2_y * 1j

        lambda_u[:, 2, 1] = alpha_cos1_z + alpha_sin1_z * 1j
        lambda_u[:, 2, 2] = alpha_cos2_z + alpha_sin2_z * 1j

        for k in range(int(nop_fft / 2)):
            for xyz in range(3):
                # HELICITY CALCULATION
                upper = np.sum(2 * np.real(lambda_u[k, xyz, :3]) * (np.imag(lambda_u[k, xyz, :3])))
                lower = np.sum(
                    (np.real(lambda_u[k, xyz, :3])) ** 2 - (np.imag(lambda_u[k, xyz, :3])) ** 2)

                if upper[j, k] > 0:
                    gamma = np.arctan(upper[j, k] / lower[j, k])
                else:
                    gamma = np.pi + (np.pi + np.arctan(upper[j, k] / lower[j,
                                                                          k]))

                lambda_u[k, xyz, :] = np.exp(-0.5 * gamma * 1j) * lambda_u[k, xyz, :]
                helic[j, k, xyz] = np.sqrt(np.sum(np.real(lambda_u[k, xyz, :3]) ** 2))
                helic[j, k, xyz] /= np.sqrt(np.sum(np.imag(lambda_u[k, xyz, :3]) ** 2))
                helic[j, k, xyz] = np.divide(1, helic[j, k, xyz])

                # ELLIPTICITY CALCULATION
                upper_e = np.sum(np.imag(lambda_u[k, xyz, :3]) * np.real(lambda_u[k, xyz, :3]))
                lower_e = np.sum(np.real(lambda_u[k, xyz, :2]) ** 2) - np.sum(
                    np.imag(lambda_u[k, xyz, :2]) ** 2)

                if upper_e > 0:
                    gamma_rot = np.arctan(upper_e / lower_e)
                else:
                    gamma_rot = np.pi + np.pi + np.arctan(upper_e / lower_e)

                lam = lambda_u[k, xyz, :2]
                lambda_u_rot = np.exp(-0.5 * gamma_rot * 1j) * lam

                ellip[j, k, xyz] = np.sqrt(np.sum(np.imag(lambda_u_rot) ** 2))
                ellip[j, k, xyz] /= np.sqrt(np.sum(np.real(lambda_u_rot) ** 2))
                ellip[j, k, xyz] *= -(np.imag(e_spec_mat[k, 0, 1]) * np.sin(wave_angle[j, k]))
                ellip[j, k, xyz] /= np.abs(np.imag(e_spec_mat[k, 0, 1]) * np.sin(wave_angle[j, k]))

    # AVERAGING HELICITY AND ELLIPTICITY RESULTS
    ellipticity = np.mean(ellip, axis=-1)
    helicity = np.mean(helic, axis=-1)

    # CREATING OUTPUT PARAMETER
    time_line = ct[0] + np.abs(nop_fft / 2) / sample_freq \
                + np.arange(1, n_stp + 1) * step_length / sample_freq
    bin_width = sample_freq / nop_fft
    freq_line = bin_width * np.arange(1, nop_fft / 2 + 1)

    # scaling power results to units with meaning
    W = nop_fft * np.sum(smooth ** 2)

    power_spec = np.zeros([n_stp, int(nop_fft / 2)])
    power_spec[:, 1:nop_fft / 2 - 1] = 1 / W * 2 * trace_spec_mat[:, 1:nop_fft / 2 - 1] / bin_width
    power_spec[:, 1] = 1 / W * trace_spec_mat[:, 0] / bin_width
    power_spec[:, nop_fft / 2] = 1 / W * trace_spec_mat[:, nop_fft / 2] / bin_width

    # KICK OUT THE ANALYSIS OF THE WEAK SIGNALS
    wave_angle[power_spec < min_psd] = np.nan
    deg_pol[power_spec < min_psd] = np.nan
    ellipticity[power_spec < min_psd] = np.nan
    helicity[power_spec < min_psd] = np.nan

    # Save as DataArrays
    b_psd = xr.DataArray(power_spec, coords=[time_line, freq_line], dims=["t", "f"])
    wave_angle = xr.DataArray(wave_angle * 180 / np.pi, coords=[time_line, freq_line],
                              dims=["t", "f"])
    ellipticity = xr.DataArray(ellipticity, coords=[time_line, freq_line], dims=["t", "f"])
    deg_pol = xr.DataArray(deg_pol, coords=[time_line, freq_line], dims=["t", "f"])
    helicity = xr.DataArray(helicity, coords=[time_line, freq_line], dims=["t", "f"])

    b_psd.f.attrs["units"] = "Hz"
    wave_angle.f.attrs["units"] = "Hz"
    deg_pol.f.attrs["units"] = "Hz"
    ellipticity.f.attrs["units"] = "Hz"
    helicity.f.attrs["units"] = "Hz"

    return b_psd, wave_angle, deg_pol, ellipticity, helicity
