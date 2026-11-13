#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import logging

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import linalg

# Local imports
from pyrfu.pyrf._units import _check_units
from pyrfu.pyrf.avg_4sc import avg_4sc
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.time_clip import time_clip
from pyrfu.pyrf.wavelet import wavelet

logger = logging.getLogger(__name__)


def _power_map(power, rows, cols, n_rows, num_k):
    r"""Power summed in the bins (rows, cols) of an (n_rows, num_k) map, the
    samples added in their order (the three numba kernels, a clipped
    histogram)."""
    flat = rows * num_k + cols
    out = np.bincount(flat.ravel(), weights=power.ravel(), minlength=n_rows * num_k)
    return out.reshape(n_rows, num_k)


def _weighted_mean_std(x, weights):
    r"""Mean and standard deviation of `x` weighted by `weights`, NaN weights
    left out (written for each of the seven averages)."""
    avg = np.nansum(x * weights) / np.nansum(weights)
    std = np.sqrt(np.nansum((x - avg) ** 2 * weights) / np.nansum(weights))
    return avg, std


def _average_windows(n_t: int, cav: int):
    r"""Start indices and time positions of the contiguous windows of `cav`
    samples over which the cross spectra are averaged: the sample before the
    centre of the window for an even `cav` (as irfu-matlab), its centre for an
    odd one (the windows were then one sample short, and one sample in each
    window was not used)."""
    starts = np.arange(n_t // cav) * cav
    return starts, starts + (cav - 1) // 2


def fk_power_spectrum_4sc(
    e,
    r,
    b,
    tints,
    cav: int = 8,
    num_k: int = 500,
    num_f: int = 200,
    df: float = None,
    w_width: int = 1,
    f_range: list = None,
):
    r"""Calculates the frequency-wave number power spectrum using the four
    MMS spacecraft. Uses a generalization of mms.fk_powerspectrum. Wavelet
    based cross-spectral analysis is used to calculate the phase difference
    each spacecraft pair and determine 3D wave vector. A generalization of
    the method used in mms.fk_powerspectrum to four point measurements.

    Parameters
    ----------
    e : list of xarray.DataArray
        Fields to apply 4SC cross-spectral analysis to. e.g., e or b fields
        (if multiple components only the first is used).
    r : list of xarray.DataArray
        Positions of the four spacecraft.
    b : list of xarray.DataArray
        background magnetic field in the same coordinates as r. Used to
        determine the parallel and perpendicular wave numbers using 4SC
        average.
    tints : list of str
        Time interval over which the power spectrum is calculated. To avoid
        boundary effects use a longer time interval for e and b.
    cav : int, Optional
        Number of points in time series used to estimate phase (contiguous
        windows of `cav` samples, odd or even). Default ``cav`` = 8.
    num_k : int, Optional
        Number of wave numbers used in spectrogram. Default ``num_k`` = 500.
    df : float, Optional
        Linear spacing of frequencies. Default ``df`` = None (log spacing).
    num_f : int, Optional
        Number of frequencies used in spectrogram. Default ``num_f`` = 200.
    w_width : float, Optional
        Multiplier for Morlet wavelet width. Default ``w_width`` = 1.
    f_range : list of float, Optional
        Frequency range for k-k plots. [minf maxf].

    Returns
    -------
    out : xarray.Dataset
        Dataset of array of powers as a function of frequency and
        wavenumber. Power is normalized to the maximum value. The wave
        numbers, k = 2 pi / lambda, are in 1/m (as irfu-matlab) for positions
        in km.

    Raises
    ------
    ValueError
        If no wave vector in tints is finite.

    Notes
    -----
    Wavelength must be larger than twice the spacecraft separations,
    otherwise spatial aliasing will occur. Averaging windows without a
    finite phase difference (e.g., data gaps) or position are left out of
    the power maps.

    The wave numbers are the centres of ``num_k`` bins between -k_max and
    k_max (0 and k_max for :math:`|k|` and k_perp), with k_max 1.1 times
    the largest :math:`|k|`, and the averages and standard deviations in
    the attributes are weighted by the power of each bin. irfu-matlab
    labels the bins with linspace(-k_max, k_max, num_k) (linspace(0, k_max,
    num_k)), up to half a bin from their centres, which offsets its averages
    by up to k_max / num_k.

    Examples
    --------
    >>> from pyrfu.mms import get_data, fk_power_spectrum_4sc
    >>> from pyrfu.pyrf import extend_tint, convert_fac

    Load data

    >>> tint = ["2015-10-16T13:05:24.00", "2015-10-16T13:05:50.000"]
    >>> ic = range(1, 5)
    >>> b_fgm = [get_data("b_gse_fgm_brst_l2", tint, i) for i in ic]
    >>> b_scm = [get_data("b_gse_scm_brst_l2", tint, i) for i in ic]

    Load spacecraft position

    >>> tint_long = extend_tint(tint, [-60, 60])
    >>> r_gse_mms = [get_data("r_gse", tint_long, i) for i in range(1, 5)]

    Convert magnetic field fluctuations to field aligned coordinates

    >>> b_fac = [convert_fac(b_s, b_f) for b_s, b_f in zip(b_scm, b_fgm)]
    >>> b_para = [b_s[:, 0] for b_s in b_fac]

    Compute dispersion relation

    >>> tint = ["2015-10-16T13:05:26.500", "2015-10-16T13:05:27.000"]
    >>> pwer = fk_power_spectrum_4sc(b_para, r_gse, b_fgm, tint, 4, 500, 2,
    ... 10, 2)

    """

    # The wave numbers are in 1/m for positions in km
    _check_units(r[0], "km", "r")

    ic = np.arange(1, 5)

    # Resample magnetic field to common time base
    # and compute 4SC average
    b = [resample(b[i - 1], b[0]) for i in ic]
    b_avg = avg_4sc(b)

    # Only the first component of vector fields is used
    e = [e_i[:, 0] if e_i.ndim > 1 else e_i for e_i in e]

    # Resample e and r to common time base
    e = [resample(e[i - 1], e[0]) for i in ic]
    r = [resample(r[i - 1], e[0]) for i in ic]

    times = e[0].time
    use_linear = df is not None

    if use_linear:
        cwt_options = {
            "linear": df,
            "return_power": False,
            "wavelet_width": 5.36 * w_width,
            "cut_edge": False,
        }
    else:
        cwt_options = {
            "n_freqs": num_f,
            "return_power": False,
            "wavelet_width": 5.36 * w_width,
            "cut_edge": False,
        }

    w = [wavelet(e[i], **cwt_options) for i in range(4)]

    num_f = len(w[0].frequency)

    times = time_clip(times, tints)
    nt = len(times)

    # Ensure even number of time points
    if nt % 2:
        times = times[:-1]
        nt -= 1

    # Wavelet transforms in the time interval, (spacecraft, time, frequency)
    w_s = np.stack([w_i.sel(time=times).data for w_i in w])
    frequencies = w[0].frequency.data

    # Compute averaged power spectrum from all spacecraft
    fk_power = np.zeros((nt, num_f), dtype=np.float64)
    for w_i in w_s:
        fk_power += np.real(w_i * np.conj(w_i)) / 4

    # Find the time positions for averaging
    cav = int(cav)
    starts, pos_av = _average_windows(nt, cav)
    n_av = len(starts)
    av_times = times[pos_av]

    # Resample background magnetic field and spacecraft positions to
    # averaged time positions
    b_avg = resample(b_avg, av_times)
    r = [resample(r[i], av_times).data for i in range(4)]

    def window_mean(x):
        # Average over each window of cav samples (one loop over the windows
        # before)
        return np.nanmean(x[: n_av * cav].reshape(n_av, cav, num_f), axis=1)

    # Cross spectra of the pairs 12, 13, 14, 23, 24 and 34 (six copies before)
    # and power, averaged over the windows
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    cross = np.stack([window_mean(w_s[i] * np.conj(w_s[j])) for i, j in pairs])
    power_avg = window_mean(fk_power)

    # Phase differences of the pairs converted to time delays
    dt12, dt13, dt14, dt23, dt24, dt34 = np.arctan2(cross.imag, cross.real) / (
        2 * np.pi * frequencies
    )

    # Delays of spacecraft 2, 3 and 4 from spacecraft 1, each the weighted
    # average of four estimates from the measured pairs (e.g., for t2 - t1:
    # dt12, dt13 - dt23, dt14 - dt24 and dt14 - dt34 - dt23), with the weights
    # 0.5, 0.2, 0.2 and 0.1 of irfu-matlab mms.fk_powerspec4SC
    dt2 = (
        0.5 * dt12
        + 0.2 * (dt13 - dt23)
        + 0.2 * (dt14 - dt24)
        + 0.1 * (dt14 - dt34 - dt23)
    )
    dt3 = (
        0.5 * dt13
        + 0.2 * (dt12 + dt23)
        + 0.2 * (dt14 - dt34)
        + 0.1 * (dt12 + dt24 - dt34)
    )
    dt4 = (
        0.5 * dt14
        + 0.2 * (dt12 + dt24)
        + 0.2 * (dt13 + dt34)
        + 0.1 * (dt12 + dt23 + dt34)
    )

    k_x, k_y, k_z = [np.zeros((n_av, num_f)) for _ in range(3)]

    # One linalg.solve call per averaging window (was one per (window,
    # frequency) pair). dr depends only on ii, so scipy factors it once and
    # solves all the finite right-hand sides (stacked as columns) in that
    # single call: the same LAPACK routine and factorization as one solve
    # per column.
    for ii in range(n_av):
        dr = np.array(
            [
                r[1][ii, :] - r[0][ii, :],
                r[2][ii, :] - r[0][ii, :],
                r[3][ii, :] - r[0][ii, :],
            ]
        )
        rhs = np.vstack([dt2[ii, :], dt3[ii, :], dt4[ii, :]])

        # Frequencies without a phase (all-NaN window) or a tetrahedron
        # without position are left NaN
        finite = np.all(np.isfinite(rhs), axis=0)
        m = np.full(rhs.shape, np.nan)

        if np.all(np.isfinite(dr)) and np.any(finite):
            m[:, finite] = linalg.solve(dr, rhs[:, finite])

        k_x[ii, :] = 2 * np.pi * frequencies * m[0, :]
        k_y[ii, :] = 2 * np.pi * frequencies * m[1, :]
        k_z[ii, :] = 2 * np.pi * frequencies * m[2, :]

    k_x, k_y, k_z = [k / 1e3 for k in [k_x, k_y, k_z]]

    k_mag = np.linalg.norm(np.array([k_x, k_y, k_z]), axis=0)

    # Parallel and perpendicular wave numbers, along the background magnetic
    # field averaged over the four spacecraft
    b_avg_xyz = b_avg.data
    k_para = (
        k_x * b_avg_xyz[:, [0]] + k_y * b_avg_xyz[:, [1]] + k_z * b_avg_xyz[:, [2]]
    ) / np.linalg.norm(b_avg, axis=1)[:, np.newaxis]
    k_perp = np.sqrt(np.maximum(k_mag**2 - k_para**2, 0.0))

    # Points without a wave vector or a power are left out of the maps (their
    # power set to 0)
    valid = np.isfinite(k_mag) & np.isfinite(k_para) & np.isfinite(power_avg)

    if not np.any(valid):
        raise ValueError("no finite wave vector in tints")

    k_x, k_y, k_z, k_mag, k_para, k_perp = [
        np.where(valid, k_, 0.0) for k_ in [k_x, k_y, k_z, k_mag, k_para, k_perp]
    ]
    power_avg = np.where(valid, power_avg, 0.0)

    # Determine the maximum and minimum wave numbers
    k_max = np.max(k_mag) * 1.1
    k_min = -k_max
    dk_mag = k_max / num_k
    dk = 2 * k_max / num_k

    # Centres of the bins of the power maps, num_k of width dk from k_min (of
    # width dk_mag from 0 for |k| and k_perp); irfu-matlab labels them with
    # linspace(k_min, k_max, num_k), up to half a bin off
    k_vec = k_min + (np.arange(num_k) + 0.5) * dk
    k_mag_vec = (np.arange(num_k) + 0.5) * dk_mag

    # Bins of the wave numbers, those out of range in the edge bins
    i_x, i_y, i_z, i_para = [
        np.clip(np.floor((k_ - k_min) / dk), 0, num_k - 1).astype(np.int64)
        for k_ in [k_x, k_y, k_z, k_para]
    ]
    i_mag, i_perp = [
        np.clip(np.floor(k_ / dk_mag), 0, num_k - 1).astype(np.int64)
        for k_ in [k_mag, k_perp]
    ]

    # Sort power into frequency and wave vector
    i_f = np.broadcast_to(np.arange(num_f), power_avg.shape)
    power_k_x_f, power_k_y_f, power_k_z_f, power_k_mag_f = [
        _power_map(power_avg, i_f, i_k, num_f, num_k) for i_k in [i_x, i_y, i_z, i_mag]
    ]

    idx_f = np.arange(num_f)

    if f_range is not None:
        idx_min_freq = bisect.bisect_left(frequencies, np.min(f_range))
        idx_max_freq = bisect.bisect_left(frequencies, np.max(f_range))
        idx_f = idx_f[idx_min_freq:idx_max_freq]

    # Sort power into wave vector space for (k_x, k_y), (k_x, k_z), (k_y,
    # k_z) and (k_perp, k_para), summed over the frequencies of f_range
    power_f = power_avg[:, idx_f]
    power_k_x_k_y, power_k_x_k_z, power_k_y_k_z, power_k_perp_k_para = [
        _power_map(power_f, rows[:, idx_f], cols[:, idx_f], num_k, num_k)
        for rows, cols in [(i_y, i_x), (i_z, i_x), (i_z, i_y), (i_para, i_perp)]
    ]

    maps = [
        power_k_x_f,
        power_k_y_f,
        power_k_z_f,
        power_k_mag_f,
        power_k_x_k_y,
        power_k_x_k_z,
        power_k_y_k_z,
        power_k_perp_k_para,
    ]

    # Normalize power to maximum value, and zero power to NaN, for plotting
    for map_ in maps:
        map_ /= np.max(map_)
        map_[map_ == 0] = np.nan

    # Find the wave numbers and frequencies weighted by power
    stats = {
        # kmag, f
        "f": (frequencies[idx_f], np.nansum(power_k_mag_f[idx_f, :], axis=1)),
        "k_mag": (k_mag_vec, np.nansum(power_k_mag_f[idx_f, :], axis=0)),
        # kpara, kperp
        "k_para": (k_vec, np.nansum(power_k_perp_k_para, axis=1)),
        "k_perp": (k_mag_vec, np.nansum(power_k_perp_k_para, axis=0)),
        # kx, ky, kz
        "k_x": (k_vec, np.nansum(power_k_x_k_y, axis=0)),
        "k_y": (k_vec, np.nansum(power_k_x_k_y, axis=1)),
        "k_z": (k_vec, np.nansum(power_k_x_k_z, axis=1)),
    }

    attrs = {}

    for name, (x, powers) in stats.items():
        attrs[f"{name}_avg"], attrs[f"{name}_std"] = _weighted_mean_std(x, powers)

    out_dict = {
        "k_x_f": (["k_x", "f"], power_k_x_f.T),
        "k_y_f": (["k_y", "f"], power_k_y_f.T),
        "k_z_f": (["k_z", "f"], power_k_z_f.T),
        "k_mag_f": (["k_mag", "f"], power_k_mag_f.T),
        "k_x_k_y": (["k_x", "k_y"], power_k_x_k_y.T),
        "k_x_k_z": (["k_x", "k_z"], power_k_x_k_z.T),
        "k_y_k_z": (["k_y", "k_z"], power_k_y_k_z.T),
        "k_perp_k_para": (["k_perp", "k_para"], power_k_perp_k_para.T),
        "k_x": k_vec,
        "k_y": k_vec,
        "k_z": k_vec,
        "k_mag": k_mag_vec,
        "k_perp": k_mag_vec,
        "k_para": k_vec,
        "f": frequencies,
    }

    out = xr.Dataset(out_dict, attrs=attrs)

    # Normalised powers, wave numbers (2 pi / lambda) in 1/m, as irfu-matlab,
    # and frequencies in Hz (none)
    for name in out.data_vars:
        out[name].attrs["UNITS"] = ""

    for name in ["k_x", "k_y", "k_z", "k_mag", "k_perp", "k_para"]:
        out[name].attrs["UNITS"] = "1/m"

    out.f.attrs["UNITS"] = "Hz"

    return out
