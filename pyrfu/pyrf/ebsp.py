#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import bisect
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import fft

# Local imports
from .ts_time import ts_time
from .ts_vec_xyz import ts_vec_xyz
from .resample import resample
from .iso2unix import iso2unix
from .start import start
from .end import end
from .cart2sph import cart2sph
from .calc_fs import calc_fs
from .convert_fac import convert_fac
from .unix2datetime64 import unix2datetime64
from .datetime642iso8601 import datetime642iso8601

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _checksampling(e_xyz, delta_b, b_bgd, full_b, flag_no_resamp):
    assert e_xyz is not None

    fs_e, fs_b = [calc_fs(e_xyz), calc_fs(delta_b)]

    resample_b_options = dict(f_s=fs_b)

    if flag_no_resamp:
        assert fs_e == fs_b
        fs_ = fs_e
    else:
        if fs_b > 1.5 * fs_e:
            e_xyz = resample(e_xyz, delta_b, **resample_b_options)
            b_bgd = resample(b_bgd, delta_b, **resample_b_options)

            fs_ = fs_b
            warnings.warn("Interpolating e to b", UserWarning)
        elif fs_e > 1.5 * fs_b:
            delta_b = resample(delta_b, e_xyz)
            b_bgd = resample(b_bgd, e_xyz)

            fs_ = fs_e
            warnings.warn("Interpolating b to e", UserWarning)
        elif fs_e == fs_b and len(e_xyz) == len(delta_b):
            fs_ = fs_e
        else:
            fs_ = 2 * fs_e

            nt = (np.min([end(e_xyz), end(delta_b)])
                  - np.max([start(e_xyz), start(delta_b)]))
            nt /= (1 / fs_)
            t = np.linspace(np.max([start(e_xyz), start(delta_b)]),
                            np.min([end(e_xyz), end(delta_b)]), int(nt))

            t = ts_time(t)

            e_xyz = resample(e_xyz, t)
            b_bgd = resample(b_bgd, t)
            full_b = resample(full_b, t)
            delta_b = resample(delta_b, t)

            warnings.warn("Interpolating b and e to 2x e sampling",
                          UserWarning)

    return e_xyz, delta_b, b_bgd, full_b, fs_


def _b_elevation(b_x, b_y, b_z, angle_b_elevation_max):
    # Remove the last sample if the total number of samples is odd
    if len(b_x) % 2:
        b_x = b_x[:-1, :]
        b_y = b_y[:-1, :]
        b_z = b_z[:-1, :]

    angle_b_elevation = np.arctan(b_z / np.sqrt(b_x ** 2 + b_y ** 2))
    angle_b_elevation = np.rad2deg(angle_b_elevation)
    idx_b_par_spin_plane = np.abs(angle_b_elevation) < angle_b_elevation_max

    return angle_b_elevation, idx_b_par_spin_plane


def _freq_int(freq_int, delta_b):
    pc12_range, pc35_range, other_range = [False, False, False]

    if isinstance(freq_int, str):
        if freq_int.lower() == "pc12":
            pc12_range = True

            freq_int = [.1, 5]

            delta_t = 1  # local

            tint = np.round([start(delta_b), end(delta_b)])
            tint = list(datetime642iso8601(unix2datetime64(tint)))

        elif freq_int.lower() == "pc35":
            pc35_range = True

            freq_int = [.002, .1]

            delta_t = 60  # local

            tint = 60 * np.array([np.round(start(delta_b) / 60),
                                  np.round(end(delta_b) / 60)])
            tint = datetime642iso8601(unix2datetime64(tint))

        else:
            raise ValueError("Invalid format of interval")

        fs_out = 1 / delta_t

        nt = np.round((iso2unix(tint[1]) - iso2unix(tint[0])) / delta_t)
        nt = nt.astype(int)  # local

        out_time = np.linspace(iso2unix(tint[0]), iso2unix(tint[1]), nt)
        out_time += delta_t / 2
        out_time = out_time[:-1]
    else:
        if freq_int[1] >= freq_int[0]:
            other_range = True

            fs_out = freq_int[1]/5

            delta_t = 1 / fs_out  # local

            nt = np.round((end(delta_b) - start(delta_b)) / delta_t)
            nt = nt.astype(int)  # local

            out_time = np.linspace(start(delta_b), end(delta_b), nt)
            out_time += delta_t / 2
            out_time = out_time[:-1]
        else:
            raise ValueError("FREQ_INT must be [f_min f_max], f_min<f_max")

    any_range = [pc12_range, pc35_range, other_range]

    return any_range, freq_int, fs_out, out_time


def _average_data(data=None, x=None, y=None, av_window=None):
    # average data with time x to time y using window

    dtx, dty = [np.median(np.diff(x)), np.median(np.diff(y))]

    if av_window is None:
        av_window = dty

    dt2 = av_window / 2

    n_data_out = len(y)

    # Pad data with NaNs from each side
    n_point_to_add = int(np.ceil(dt2 / dtx))
    pad_nan = np.zeros((n_point_to_add, data.shape[1])) * np.nan
    data = np.vstack([pad_nan, data, pad_nan])
    pad_time = dtx * np.arange(n_point_to_add)
    x = np.hstack([x[0] - np.flip(pad_time), x, x[-1] + pad_time])

    out = np.zeros((n_data_out, data.shape[1]), dtype="complex128")

    for i, iy in enumerate(y):
        il = bisect.bisect_left(x, iy - dt2)
        ir = bisect.bisect_left(x, iy + dt2)

        out[i, :] = np.nanmean(data[il:ir, :], axis=0)

    return out


def _bb_xxyyzzss(power_bx_plot, power_by_plot, power_bz_plot, power_2b_plot):
    bb_xxyyzzss = np.tile(power_bx_plot, (4, 1, 1))
    bb_xxyyzzss = np.transpose(bb_xxyyzzss, [1, 2, 0])
    bb_xxyyzzss[:, :, 1] = power_by_plot
    bb_xxyyzzss[:, :, 2] = power_bz_plot
    bb_xxyyzzss[:, :, 3] = power_2b_plot
    return np.real(bb_xxyyzzss)


def _ee_xxyyzzss(power_ex_plot, power_ey_plot, power_ez_plot, power_2e_plot):
    ee_xxyyzzss = np.tile(power_ex_plot, (4, 1, 1))
    ee_xxyyzzss = np.transpose(ee_xxyyzzss, [1, 2, 0])
    ee_xxyyzzss[:, :, 1] = power_ey_plot
    ee_xxyyzzss[:, :, 2] = power_ez_plot
    ee_xxyyzzss[:, :, 3] = power_2e_plot
    return np.real(ee_xxyyzzss)


def ebsp(e_xyz, delta_b, full_b, b_bgd, xyz, freq_int, **kwargs):
    """Calculates wavelet spectra of E&B and Poynting flux using wavelets
    (Morlet wavelet). Also computes polarization parameters of B using SVD
    [7]_. SVD is performed on spectral matrices computed from the time series
    of B using wavelets and then averaged over a number of wave periods.

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the wave electric field.
    delta_b : xarray.DataArray
        Time series of the wave magnetic field.
    full_b : xarray.DataArray
        Time series of the high resolution background magnetic field used
        for E.B=0.
    b_bgd : xarray.DataArray
        Time series of the background magnetic field used for field aligned
        coordinates.
    xyz : xarray.DataArray
        Time series of the position time series of spacecraft used for field
        aligned coordinates.
    freq_int : str or array_like
        Frequency interval :
            * "pc12" : [0.1, 5.0]
            * "pc35" : [2e-3, 0.1]
            * [fmin, fmax] : arbitrary interval [fmin,fmax]

    Returns
    -------
    res : xarray.Dataset
        Dataset with :
            * t : xarray.DataArray
                Time.
            * f : xarray.DataArray
                Frequencies.
            * bb_xxyyzzss : xarray.DataArray
                delta_b power spectrum with :
                    * [...,0] : x
                    * [...,1] : y
                    * [...,2] : z
                    * [...,3] : sum
            * ee_xxyyzzss : xarray.DataArray
                E power spectrum with :
                    * [...,0] : x
                    * [...,1] : y
                    * [...,2] : z
                    * [...,3] : sum
            * ee_ss : xarray.DataArray
                E power spectrum (xx+yy spacecraft coordinates, e.g. ISR2).
            * pf_xyz : xarray.DataArray
                Poynting flux (xyz).
            * pf_rtp : xarray.DataArray
                Poynting flux (r, theta, phi) [angles in degrees].
            * dop : xarray.DataArray
                3D degree of polarization.
            * dop2d : xarray.DataArray
                2D degree of polarization in the polarization plane.
            * planarity : xarray.DataArray
                Planarity of polarization.
            * ellipticity : xarray.DataArray
                Ellipticity of polarization ellipse.
            * k : xarray.DataArray
                k-vector (theta, phi FAC) [angles in degrees].

    Other Parameters
    ----------------
    polarization : bool
        Computes polarization parameters. Default False.
    noresamp : bool
        No resampling, E and delta_b are given at the same time line.
        Default False.
    fac : bool
        Uses FAC coordinate system (defined by b0 and optionally xyz),
        otherwise no coordinate system transformation is performed. Default
        False.
    de_dot_b0 : bool
        Computes dEz from delta_b dot B = 0, uses full_b. Default False.
    full_b_db : bool
        delta_b contains DC field. Default False.
    nav : int
        Number of wave periods to average Default 8.
    fac_matrix : numpy.ndarray
        Specify rotation matrix to FAC system Default None.
    m_width_coeff : int or float
        Specify coefficient to multiple Morlet wavelet width by. Default 1.

    See also
    --------
    pyrfu.plot.pl_ebsp : to fill.
    pyrfu.pyrf.convert_fac : Transforms to a field-aligned coordinate.

    Notes
    -----
    This software was developed as part of the MAARBLE (Monitoring, Analyzing
    and Assessing Radiation Belt Energization and Loss) collaborative
    research project which has received funding from the European
    Community's Seventh Framework Programme (FP7-SPACE-2011-1) under grant
    agreement n. 284520.

    References
    ----------
    .. [7] 	Santolík, O., Parrot. M., and  Lefeuvre. F. (2003) Singular value
            decomposition methods for wave propagation analysis,Radio Sci.,
            38(1), 1010, doi : https://doi.org/10.1029/2000RS002523 .

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> # Time interval
    >>> tint_brst = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]
    >>> # Spacecraft index
    >>> mms_id = 3
    >>> # Load spacecraft position
    >>> tint_long = pyrf.extend_tint(tint_brst, [-100, 100])
    >>> r_xyz = mms.get_data("R_gse", tint_long, mms_id)
    >>> # Load background magnetic field, electric field and magnetic field
    fluctuations
    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint_brst, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint_brst, mms_id)
    >>> b_scm = mms.get_data("B_gse_scm_brst_l2", tint_brst, mms_id)
    >>> # Polarization analysis
    >>> options = dict(polarization=True, fac=True)
    >>> polarization = pyrf.ebsp(e_xyz, b_scm, b_xyz, b_xyz, r_xyz,
    >>>                          freq_int=[10, 4000], **options)

    """

    assert isinstance(delta_b, xr.DataArray), "delta_b must be a DataArray"
    assert isinstance(full_b, xr.DataArray), "full_b must be a DataArray"
    assert isinstance(b_bgd, xr.DataArray), "b0 must be a DataArray"
    assert isinstance(xyz, xr.DataArray), "xyz must be a DataArray"

    # Compute magnetic field fluctuations sampling frequency
    fsb = calc_fs(delta_b)

    # Below which we cannot apply E*B=0
    angle_b_elevation_max = 15

    want_ee = e_xyz is not None

    res = dict(t=None, f=None, flagFac=0, bb_xxyyzzss=None, ee_xxyyzzss=None,
               ee_ss=None, pf_xyz=None, pf_rtp=None, dop=None, dop2d=None,
               planarity=None, ellipticity=None, k_tp=None, full_b=full_b,
               b0=b_bgd, r=xyz)

    want_polarization = kwargs.get("polarization", False)

    flag_no_resample = kwargs.get("no_resample", False)
    flag_want_fac = kwargs.get("fac", False)
    flag_de_dot_b0 = kwargs.get("de_dot_b0", False)
    flag_full_b_db = kwargs.get("full_b_db", False)

    m_width_coeff = kwargs.get("m_width_coeff", 1.)

    # Number of wave periods to average
    n_wave_period_to_average = kwargs.get("nav", 8)

    # matrix for rotation to FAC
    fac_matrix = kwargs.get("fac_matrix", None)

    if flag_want_fac and fac_matrix is None:
        if b_bgd is None:
            raise ValueError(
                "ebsp(): at least b0 should be given for option FAC")

        if xyz is None:
            print("convert_fac : assuming s/c position [1 0 0] for estimating "
                  "FAC")
            xyz = [1, 0, 0]
            xyz = ts_vec_xyz(delta_b.time.data,
                             np.tile(xyz, (len(delta_b), 1)))

        xyz = resample(xyz, delta_b, **{"f_s": fsb})

    b_bgd = resample(b_bgd, delta_b, **{"f_s": fsb})

    if flag_full_b_db:
        full_b = delta_b
        res["full_b"] = full_b
        delta_b = delta_b - b_bgd

    if flag_de_dot_b0 and full_b is None:
        raise ValueError("full_b must be given for option de_dot_b0=0")

    any_range, freq_int, out_sampling, out_time = _freq_int(freq_int, delta_b)
    pc12_range, pc35_range, other_range = any_range

    if want_ee:
        # Check the sampling rate
        temp_ = _checksampling(e_xyz, delta_b, b_bgd, full_b, flag_no_resample)
        e_xyz, delta_b, b_bgd, full_b, in_sampling = temp_

    else:
        in_sampling = calc_fs(delta_b)

        e_xyz = None

    if in_sampling / 2 < freq_int[1]:
        raise ValueError("F_MAX must be lower than the Nyquist frequency")

    if want_ee and e_xyz.shape[1] < 3 and not flag_de_dot_b0:
        raise ValueError("E must have all 3 components or flag de_dot_db=0 "
                         "must be given")

    if len(delta_b) % 2:
        delta_b = delta_b[:-1, :]
        b_bgd = b_bgd[:-1, :]

        if fac_matrix is None:
            xyz = xyz[:-1, :]
        else:
            fac_matrix["t"] = fac_matrix["t"][:-1, :]

            fac_matrix["rotMatrix"] = fac_matrix["rotMatrix"][:-1, :, :]

        if want_ee:
            e_xyz = e_xyz[:-1, :]

    in_time = delta_b.time.data.view("i8")*1e-9

    b_x, b_y, b_z = [None, None, None]

    idx_b_par_spin_plane = None

    if flag_de_dot_b0:
        b_x, b_y, b_z = [full_b[:, i].data for i in range(3)]

        # Remove the last sample if the total number of samples is odd
        temp_ = _b_elevation(b_x, b_y, b_z, angle_b_elevation_max)
        angle_b_elevation, idx_b_par_spin_plane = temp_

    # If E has all three components, transform E and B waveforms to a magnetic
    # field aligned coordinate (FAC) and save eisr for computation of e_sum.
    # Otherwise we compute Ez within the main loop and do the transformation
    # to FAC there.

    time_b0 = 0
    if flag_want_fac:
        res["flagFac"] = True

        time_b0 = b_bgd.time.data.view("i8") * 1e-9

        if want_ee:
            if not flag_de_dot_b0:
                eisr2 = e_xyz[:, :2]

                if e_xyz.shape[1] < 3:
                    raise TypeError("E must be a 3D vector to be rotated to "
                                    "FAC")

                if fac_matrix is None:
                    e_xyz = convert_fac(e_xyz, b_bgd, xyz)
                else:
                    e_xyz = convert_fac(e_xyz, fac_matrix)

        if fac_matrix is None:
            delta_b = convert_fac(delta_b, b_bgd, xyz)
        else:
            delta_b = convert_fac(delta_b, fac_matrix)

    # Find the frequencies for an FFT of all data and set important parameters
    nd2 = len(in_time)/2

    freq = in_sampling * np.arange(nd2) / nd2 * .5

    # The frequencies corresponding to FFT
    w_ = np.hstack([0, freq, -np.flip(freq[:-1])])

    morlet_width = 5.36 * m_width_coeff

    # to get proper overlap for Morlet
    freq_number = np.ceil(
        (np.log10(freq_int[1]) - np.log10(freq_int[0])) * 12 * m_width_coeff)

    a_min, a_max = [np.log10(0.5 * in_sampling / freq_int[1]),
                    np.log10(0.5 * in_sampling / freq_int[0])]

    a_number = freq_number
    a_ = np.logspace(a_min, a_max, int(a_number))

    # Maximum frequency
    w_0 = in_sampling / 2

    # Width of the Morlet wavelet
    sigma = morlet_width / w_0

    # Make the FFT of all data
    idx_nan_b = np.isnan(delta_b.data)

    delta_b.data[idx_nan_b] = 0

    swb = fft.fft(delta_b.data, axis=0, workers=os.cpu_count())
    # swb = pyfftw.interfaces.numpy_fft.fft(delta_b, axis=0, threads=n_threads)

    idx_nan_e, idx_nan_eisr2 = [None, None]

    sw_e, sw_eisr2 = [None, None]

    if want_ee:
        print("ebsp ... calculate E and B wavelet transform ... ")

        idx_nan_e = np.isnan(e_xyz.data)

        e_xyz.data[idx_nan_e] = 0

        sw_e = fft.fft(e_xyz.data, axis=0, workers=os.cpu_count())
        # sw_e = pyfftw.interfaces.numpy_fft.fft(e_xyz, axis=0,
        # threads=n_threads)

        if flag_want_fac and not flag_de_dot_b0:
            idx_nan_eisr2 = np.isnan(eisr2.data)

            eisr2.data[idx_nan_eisr2] = 0

            sw_eisr2 = fft.fft(eisr2.data, axis=0, workers=os.cpu_count())
            # sw_eisr2 = pyfftw.interfaces.numpy_fft.fft(eisr2, axis=0,
            # threads=n_threads)
    else:
        print("ebsp ... calculate B wavelet transform ....")

    # Loop through all frequencies
    n_data, n_freq, n_data_out = [len(in_time), len(a_), len(out_time)]

    #
    power_ex_plot = np.zeros((n_data, n_freq), dtype="complex128")
    power_ey_plot = np.zeros((n_data, n_freq), dtype="complex128")
    power_ez_plot = np.zeros((n_data, n_freq), dtype="complex128")
    power_2e_plot = np.zeros((n_data, n_freq), dtype="complex128")
    power_2e_isr2_plot = np.zeros((n_data, n_freq), dtype="complex128")

    power_bx_plot = np.zeros((n_data, n_freq), dtype="complex128")
    power_by_plot = np.zeros((n_data, n_freq), dtype="complex128")
    power_bz_plot = np.zeros((n_data, n_freq), dtype="complex128")
    power_2b_plot = np.zeros((n_data, n_freq), dtype="complex128")

    s_plot_x = np.zeros((n_data, n_freq))
    s_plot_y = np.zeros((n_data, n_freq))
    s_plot_z = np.zeros((n_data, n_freq))

    planarity, ellipticity = [np.zeros((n_data_out, n_freq)) for _ in range(2)]
    dop_3d = np.zeros((n_data_out, n_freq), dtype="complex128")
    dop_2d = np.zeros((n_data_out, n_freq), dtype="complex128")

    the_svd_fac = np.zeros((n_data_out, n_freq))
    phi_svd_fac = np.zeros((n_data_out, n_freq))

    # Get the correct frequencies for the wavelet transform
    frequency_vec = w_0/a_

    censure = np.floor(
        2 * a_ * out_sampling / in_sampling * n_wave_period_to_average)

    for ind_a in range(len(a_)):
        # resample to 1 second sampling for Pc1-2 or 1 minute sampling for
        # Pc3-5 average top frequencies to 1 second/1 minute below will be
        # an average over 8 wave periods. first find where one sample is less
        # than eight wave periods

        if frequency_vec[ind_a] / n_wave_period_to_average > out_sampling:
            av_window = 1 / out_sampling
        else:
            av_window = n_wave_period_to_average / frequency_vec[ind_a]

        # Get the wavelet transform by backward FFT
        w_exp_mat = np.exp(-sigma * sigma * ((a_[ind_a] * w_ - w_0) ** 2) / 2)
        w_exp_mat2 = np.tile(w_exp_mat, (2, 1)).T
        w_exp_mat = np.tile(w_exp_mat, (3, 1)).T

        wb = fft.ifft(np.sqrt(1) * swb * w_exp_mat, axis=0,
                      workers=os.cpu_count())
        # wb = pyfftw.interfaces.numpy_fft.ifft(np.sqrt(1) * swb * w_exp_mat,
        #                                      axis=0, threads=n_threads)
        wb[idx_nan_b] = np.nan

        we, w_eisr2 = [None, None]

        if want_ee:
            we = fft.ifft(np.sqrt(1) * sw_e * w_exp_mat, axis=0,
                          workers=os.cpu_count())
            # arg_ = np.sqrt(1) * sw_e * w_exp_mat
            # we = pyfftw.interfaces.numpy_fft.ifft(arg_, axis=0,
            #                                      threads=n_threads)

            we[idx_nan_e] = np.nan

            if flag_want_fac and not flag_de_dot_b0:
                w_eisr2 = fft.ifft(np.sqrt(1) * sw_eisr2 * w_exp_mat2,
                                   axis=0, workers=os.cpu_count())
                # arg_ = np.sqrt(1) * sw_eisr2 * w_exp_mat2
                # w_eisr2 = pyfftw.interfaces.numpy_fft.ifft(arg_, axis=0,
                #                                            threads=n_threads)
                w_eisr2[idx_nan_eisr2] = np.nan

        new_freq_mat = w_0 / a_[ind_a]
        # Power spectrum of E and Poynting flux

        if want_ee:
            # Power spectrum of E, power = (2*pi)*conj(W).*W./new_freq_mat
            if flag_want_fac and not flag_de_dot_b0:
                sum_power_eisr2 = np.sum(2 * np.pi * (w_eisr2 * np.conj(
                    w_eisr2)) / new_freq_mat, axis=1)
            else:
                sum_power_eisr2 = np.sum(2 * np.pi * (we * np.conj(we)) /
                                         new_freq_mat, axis=1)

            power_2e_isr2_plot[:, ind_a] = sum_power_eisr2

            # Compute Ez from dE * B = 0
            if flag_de_dot_b0:
                we_re, we_im = [np.real(we), np.imag(we)]

                we_z = -(we_re[:, 0] * b_x + we_re[:, 1] * b_y) / b_z - 1j * (
                            we_im[:, 0] * b_x + we_im[:, 1] * b_y) / b_z
                we_z[idx_b_par_spin_plane] = np.nan

                if flag_want_fac:
                    if fac_matrix is None:
                        arg_ = ts_vec_xyz(time_b0,
                                          np.hstack([we[:, :2], we_z]))
                        we = convert_fac(arg_, b_bgd, xyz)
                    else:
                        arg_ = ts_vec_xyz(time_b0,
                                          np.hstack([we[:, :2], we_z]))
                        we = convert_fac(arg_, fac_matrix)

                    we = we[:, 1:]
                else:
                    we = np.hstack([we[:, :2], we_z])

            power_e = 2 * np.pi * (we * np.conj(we)) / new_freq_mat
            power_e = np.vstack([power_e.T, np.sum(power_e, axis=1)]).T

            power_ex_plot[:, ind_a] = power_e[:, 0]
            power_ey_plot[:, ind_a] = power_e[:, 1]
            power_ez_plot[:, ind_a] = power_e[:, 2]
            power_2e_plot[:, ind_a] = power_e[:, 3]

            # Poynting flux calculations, assume E and b units mV/m and nT,
            # get  S in uW/m^2 4pi from wavelets, see A. Tjulins power
            # estimates
            coeff_poynting = 10 / 4 / np.pi * (1/4) * (4 * np.pi)

            s = np.zeros((n_data, 3))

            we_x, we_y, we_z = [we[:, i] for i in range(3)]
            wb_x, wb_y, wb_z = [wb[:, i] for i in range(3)]

            s[:, 0] = coeff_poynting * np.real(
                we_y * np.conj(wb_z) + np.conj(we_y) * wb_z - we_z * np.conj(
                    wb_y) - np.conj(we_z) * wb_y) / new_freq_mat
            s[:, 1] = coeff_poynting * np.real(
                we_z * np.conj(wb_x) + np.conj(we_z) * wb_x - we_x * np.conj(
                    wb_z) - np.conj(we_x) * wb_z) / new_freq_mat
            s[:, 2] = coeff_poynting * np.real(
                we_x * np.conj(wb_y) + np.conj(we_x) * wb_y - we_y * np.conj(
                    wb_x) - np.conj(we_y) * wb_x) / new_freq_mat

            s_plot_x[:, ind_a] = s[:, 0]
            s_plot_y[:, ind_a] = s[:, 1]
            s_plot_z[:, ind_a] = s[:, 2]

        # Power spectrum of B
        power_b = 2 * np.pi * (wb * np.conj(wb)) / new_freq_mat
        power_b = np.vstack([power_b.T, np.sum(power_b, axis=1)]).T

        power_bx_plot[:, ind_a] = power_b[:, 0]
        power_by_plot[:, ind_a] = power_b[:, 1]
        power_bz_plot[:, ind_a] = power_b[:, 2]
        power_2b_plot[:, ind_a] = power_b[:, 3]

        # Polarization parameters
        if want_polarization:
            # Construct spectral matrix and average it
            s_mat = np.zeros((3, 3, n_data), dtype="complex128")

            for i in range(3):
                for j in range(3):
                    s_mat[i, j, :] = 2 * np.pi * (
                                wb[:, i] * np.conj(wb[:, j])) / new_freq_mat

            s_mat = np.transpose(s_mat, [2, 0, 1])

            # Averaged s_mat
            s_mat_avg = np.zeros((n_data_out, 3, 3), dtype="complex128")

            for comp in range(3):
                s_mat_avg[..., comp] = _average_data(s_mat[..., comp], in_time,
                                                     out_time, av_window)

            # Remove data possibly influenced by edge effects
            censure_idx = np.hstack(
                [np.arange(np.min([censure[ind_a], len(out_time)])),
                 np.arange(np.max([0, len(out_time) - censure[ind_a] - 1]),
                           len(out_time))]).astype(int)

            s_mat_avg[censure_idx, ...] = np.nan

            # compute singular value decomposition
            # real matrix which is superposition of real part of spectral
            # matrix over imaginary part
            a_mat, u_mat = [np.zeros((6, 3, n_data_out)) for _ in range(2)]
            w_mat, v_mat = [np.zeros((3, 3, n_data_out)) for _ in range(2)]

            # wSingularValues = zeros(3,n_data2);
            # R = zeros(3,3,n_data2); #spectral matrix in coordinate defined
            # by V axes
            a_mat[:3, ...] = np.real(np.transpose(s_mat_avg, [1, 2, 0]))
            a_mat[3:6, ...] = -np.imag(np.transpose(s_mat_avg, [1, 2, 0]))

            for i in range(n_data_out):
                if np.isnan(a_mat[..., i]).any():
                    u_mat[..., i], w_mat[..., i], v_mat[..., i] = [np.nan,
                                                                   np.nan,
                                                                   np.nan]
                else:
                    u_mat[..., i], w_mat[..., i], v_mat[
                        ..., i] = np.linalg.svd(a_mat[..., i],
                                                full_matrices=False)

            # compute direction of propagation
            sign_kz = np.sign(v_mat[2, 2, :])
            v_mat[2, 2, :] = v_mat[2, 2, :] * sign_kz
            v_mat[1, 2, :] = v_mat[1, 2, :] * sign_kz
            v_mat[0, 2, :] = v_mat[0, 2, :] * sign_kz

            the_svd_fac[:, ind_a] = np.abs(np.squeeze(np.arctan(
                np.sqrt(v_mat[0, 2, :] ** 2 + v_mat[1, 2, :] ** 2)
                / v_mat[2, 2, :])))
            phi_svd_fac[:, ind_a] = np.squeeze(
                np.arctan2(v_mat[1, 2, :], v_mat[0, 2, :]))

            # Calculate polarization parameters
            planarity_local = np.squeeze(
                1 - np.sqrt(w_mat[2, 2, :] / w_mat[0, 0, :]))
            planarity_local[censure_idx] = np.nan

            planarity[:, ind_a] = planarity_local

            # ellipticity: ratio of axes of polarization ellipse axes*sign of
            # polarization

            ellipticity_local = np.squeeze(
                w_mat[1, 1, :] / w_mat[0, 0, :]) * np.sign(
                np.imag(s_mat_avg[:, 0, 1]))
            ellipticity_local[censure_idx] = np.nan

            ellipticity[:, ind_a] = ellipticity_local

            # DOP = sqrt[(3/2.*trace(SM^2)./(trace(SM))^2 - 1/2)];
            # Samson, 1973, JGR
            dop = np.sqrt((3 / 2) * (
                        np.trace(np.matmul(s_mat_avg, s_mat_avg), axis1=1,
                                 axis2=2) / np.trace(s_mat_avg, axis1=1,
                                                     axis2=2) ** 2) - 1 / 2)

            dop[censure_idx] = np.nan
            dop_3d[:, ind_a] = dop

            # DOP in 2D = sqrt[2*trace(rA^2)/trace(rA)^2 - 1)]; Ulrich
            v_mat_new = np.transpose(v_mat, [2, 0, 1])

            s_mat_avg2dim = np.matmul(v_mat_new, np.matmul(s_mat_avg,
                                                           np.transpose(
                                                               v_mat_new,
                                                               [0, 2, 1])))
            s_mat_avg2dim = s_mat_avg2dim[:, :2, :2]
            s_mat_avg = s_mat_avg2dim

            dop2dim = np.sqrt(2 * (
                        np.trace(np.matmul(s_mat_avg, s_mat_avg), axis1=1,
                                 axis2=2) / np.trace(s_mat_avg, axis1=1,
                                                     axis2=2) ** 2) - 1)
            dop2dim[censure_idx] = np.nan
            dop_2d[:, ind_a] = dop

    # set data gaps to NaN and remove edge effects
    censure = np.floor(2*a_)

    for ind_a in range(len(a_)):
        censure_idx = np.hstack(
            [np.arange(np.min([censure[ind_a], len(in_time)])),
             np.arange(np.max([1, len(in_time) - censure[ind_a]]),
                       len(in_time))])

        censure_idx = censure_idx.astype(int)

        power_bx_plot[censure_idx, ind_a] = np.nan
        power_by_plot[censure_idx, ind_a] = np.nan
        power_bz_plot[censure_idx, ind_a] = np.nan
        power_2b_plot[censure_idx, ind_a] = np.nan

        if want_ee:
            power_ex_plot[censure_idx, ind_a] = np.nan
            power_ey_plot[censure_idx, ind_a] = np.nan
            power_ez_plot[censure_idx, ind_a] = np.nan
            power_2e_plot[censure_idx, ind_a] = np.nan

            power_2e_isr2_plot[censure_idx, ind_a] = np.nan

            s_plot_x[censure_idx, ind_a] = np.nan
            s_plot_y[censure_idx, ind_a] = np.nan
            s_plot_z[censure_idx, ind_a] = np.nan

    # remove edge effects from data gaps
    idx_nan_e = np.sum(idx_nan_e, axis=1) > 0
    idx_nan_b = np.sum(idx_nan_b, axis=1) > 0
    idx_nan_eisr2 = np.sum(idx_nan_eisr2, axis=1) > 0

    n_data2 = len(power_2b_plot)
    if pc12_range or other_range:
        censure3 = np.floor(1.8 * a_)
    elif pc35_range:
        censure3 = np.floor(.4 * a_)
    else:
        raise ValueError("Invalid range")

    for i in range(len(idx_nan_b) - 1):
        if idx_nan_b[i] < idx_nan_b[i + 1]:
            for j in range(len(a_)):
                censure_index_front = np.arange(np.max([i - censure3[j], 0]),
                                                i)

                power_bx_plot[censure_index_front, j] = np.nan
                power_by_plot[censure_index_front, j] = np.nan
                power_bz_plot[censure_index_front, j] = np.nan
                power_2b_plot[censure_index_front, j] = np.nan

                s_plot_x[censure_index_front, j] = np.nan
                s_plot_y[censure_index_front, j] = np.nan
                s_plot_z[censure_index_front, j] = np.nan

        if idx_nan_b[i] > idx_nan_b[i + 1]:
            for j in range(len(a_)):
                censure_index_back = np.arange(i, np.min(
                    [i + censure3[j], n_data2]))

                power_bx_plot[censure_index_back, j] = np.nan
                power_by_plot[censure_index_back, j] = np.nan
                power_bz_plot[censure_index_back, j] = np.nan
                power_2b_plot[censure_index_back, j] = np.nan

                s_plot_x[censure_index_back, j] = np.nan
                s_plot_y[censure_index_back, j] = np.nan
                s_plot_z[censure_index_back, j] = np.nan

    n_data3 = len(power_2e_plot)

    for i in range(len(idx_nan_e) - 1):
        if idx_nan_e[i] < idx_nan_e[i + 1]:
            for j in range(len(a_)):
                censure_index_front = np.arange(np.max([i - censure3[j], 1]),
                                                i)

                power_ex_plot[censure_index_front, j] = np.nan
                power_ey_plot[censure_index_front, j] = np.nan
                power_ez_plot[censure_index_front, j] = np.nan
                power_2e_plot[censure_index_front, j] = np.nan

                power_2e_isr2_plot[censure_index_front, j] = np.nan

                s_plot_x[censure_index_front, j] = np.nan
                s_plot_y[censure_index_front, j] = np.nan
                s_plot_z[censure_index_front, j] = np.nan

        elif idx_nan_e[i] > idx_nan_e[i + 1]:
            for j in range(len(a_)):
                censure_index_back = np.arange(i, np.min(
                    [i + censure3[j], n_data3]))

                power_ex_plot[censure_index_back, j] = np.nan
                power_ey_plot[censure_index_back, j] = np.nan
                power_ez_plot[censure_index_back, j] = np.nan
                power_2e_plot[censure_index_back, j] = np.nan

                power_2e_isr2_plot[censure_index_back, j] = np.nan

                s_plot_x[censure_index_back, j] = np.nan
                s_plot_y[censure_index_back, j] = np.nan
                s_plot_z[censure_index_back, j] = np.nan

        else:
            continue

    n_data4 = len(power_2e_isr2_plot)

    for i in range(len(idx_nan_eisr2) - 1):
        if idx_nan_eisr2[i] < idx_nan_eisr2[i + 1]:
            for j in range(len(a_)):
                censure_index_front = np.arange(np.max([i - censure3[j], 0]),
                                                i)

                power_2e_isr2_plot[censure_index_front, j] = np.nan

        elif idx_nan_eisr2[i] > idx_nan_eisr2[i + 1]:
            for j in range(len(a_)):
                censure_index_back = np.arange(i, np.min(
                    [i + censure3[j], n_data4]))

                power_2e_isr2_plot[censure_index_back, j] = np.nan

    power_bx_plot = _average_data(power_bx_plot, in_time, out_time)
    power_by_plot = _average_data(power_by_plot, in_time, out_time)
    power_bz_plot = _average_data(power_bz_plot, in_time, out_time)
    power_2b_plot = _average_data(power_2b_plot, in_time, out_time)

    bb_xxyyzzss = _bb_xxyyzzss(power_bx_plot, power_by_plot, power_bz_plot,
                               power_2b_plot)

    # Output
    res["t"] = unix2datetime64(out_time)
    res["f"] = frequency_vec
    res["bb_xxyyzzss"] = xr.DataArray(bb_xxyyzzss,
                                      coords=[res["t"], res["f"],
                                              ["xx", "yy", "zz", "ss"]],
                                      dims=["time", "frequency", "comp"])

    if want_ee:
        power_ex_plot = _average_data(power_ex_plot, in_time, out_time)
        power_ey_plot = _average_data(power_ey_plot, in_time, out_time)
        power_ez_plot = _average_data(power_ez_plot, in_time, out_time)
        power_2e_plot = _average_data(power_2e_plot, in_time, out_time)

        power_2e_isr2_plot = _average_data(power_2e_isr2_plot,
                                           in_time, out_time)
        power_2e_isr2_plot = np.real(power_2e_isr2_plot)

        s_plot_x = np.real(_average_data(s_plot_x, in_time, out_time))
        s_plot_y = np.real(_average_data(s_plot_y, in_time, out_time))
        s_plot_z = np.real(_average_data(s_plot_z, in_time, out_time))
        s_azimuth, s_elevation, s_r = cart2sph(s_plot_x, s_plot_y, s_plot_z)

        ee_xxyyzzss = _ee_xxyyzzss(power_ex_plot, power_ey_plot,
                                   power_ez_plot, power_2e_plot)

        poynting_xyz = np.tile(s_plot_x, (3, 1, 1))
        poynting_xyz = np.transpose(poynting_xyz, [1, 2, 0])
        poynting_xyz[:, :, 1] = s_plot_y
        poynting_xyz[:, :, 2] = s_plot_z
        poynting_xyz = poynting_xyz.astype(float)

        poynting_r_th_ph = np.tile(s_r, (3, 1, 1))
        poynting_r_th_ph = np.transpose(poynting_r_th_ph, [1, 2, 0])
        poynting_r_th_ph[..., 1] = np.pi / 2 - s_elevation
        poynting_r_th_ph[..., 2] = s_azimuth
        poynting_r_th_ph[..., 1:] = poynting_r_th_ph[..., 1:] * 180 / np.pi
        poynting_r_th_ph = poynting_r_th_ph.astype(float)

        # Output
        res["ee_ss"] = power_2e_isr2_plot.astype(float)

        res["ee_xxyyzzss"] = xr.DataArray(ee_xxyyzzss,
                                          coords=[res["t"], res["f"],
                                                  ["xx", "yy", "zz", "ss"]],
                                          dims=["time", "frequency", "comp"])

        res["pf_xyz"] = xr.DataArray(poynting_xyz,
                                     coords=[res["t"], res["f"],
                                             ["x", "y", "z"]],
                                     dims=["time", "frequency", "comp"])

        res["pf_rtp"] = xr.DataArray(poynting_r_th_ph,
                                     coords=[res["t"], res["f"],
                                             ["rho", "theta", "phi"]],
                                     dims=["time", "frequency", "comp"])

    if want_polarization:
        # Define parameters for which we cannot compute the wave vector
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            ind_low_planarity = planarity < 0.5
            ind_low_ellipticity = np.abs(ellipticity) < .2

        the_svd_fac[ind_low_planarity] = np.nan
        phi_svd_fac[ind_low_planarity] = np.nan

        the_svd_fac[ind_low_ellipticity] = np.nan
        phi_svd_fac[ind_low_ellipticity] = np.nan

        k_th_ph_svd_fac = np.zeros((the_svd_fac.shape[0],
                                    the_svd_fac.shape[1], 2))
        k_th_ph_svd_fac[..., 0] = the_svd_fac
        k_th_ph_svd_fac[..., 1] = phi_svd_fac

        # Output
        res["dop"] = xr.DataArray(np.real(dop_3d),
                                  coords=[res["t"], res["f"]],
                                  dims=["time", "frequency"])

        res["dop2d"] = xr.DataArray(np.real(dop_2d),
                                    coords=[res["t"], res["f"]],
                                    dims=["time", "frequency"])

        res["planarity"] = xr.DataArray(planarity,
                                        coords=[res["t"], res["f"]],
                                        dims=["time", "frequency"])

        res["ellipticity"] = xr.DataArray(ellipticity,
                                          coords=[res["t"], res["f"]],
                                          dims=["time", "frequency"])

        res["k_tp"] = xr.DataArray(k_th_ph_svd_fac,
                                   coords=[res["t"], res["f"],
                                           ["theta", "phi"]],
                                   dims=["time", "frequency", "comp"])

    return res
