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

import multiprocessing as mp
import numpy as np
import xarray as xr
import pyfftw
import numba
import matplotlib.pyplot as plt


@numba.jit(nopython=True, fastmath=True)
def calc_w_w(s_ww, aa, sigma, ww, w0):
    w_w = np.sqrt(1) * s_ww * np.exp(-sigma * sigma * ((aa * ww - w0) ** 2) / 2)
    return w_w


@numba.jit(nopython=True, parallel=True, fastmath=True)
def calc_power_r(power, new_freq_mat):
    power2 = np.absolute((2 * np.pi) * np.conj(power) * power / new_freq_mat)
    return power2


@numba.jit(nopython=True, parallel=True, fastmath=True)
def calc_power_c(power, new_freq_mat):
    power2 = np.sqrt(np.absolute((2 * np.pi) / new_freq_mat)) * power
    return power2


def wavelet(inp, **kwargs):
    """Computes wavelet spectrogram based on fast FFT algorithm.

    Parameters
    ----------
    inp : xarray.DataArray
        Input quantity.

    **kwargs : dict
        Hash table of keyword arguments with :
            * fs : int or float
                Sampling frequency of the input time series.

            * f : list or ndarray
                Vector [f_min f_max], calculate spectra between frequencies f_min and f_max.

            * nf : int or float
                Number of frequency bins.

            * wavelet_width : int or float
                Width of the Morlet wavelet. Default 5.36.

            * linear : float
                Linear spacing between frequencies of df.

            * return_power : bool
                Set to True to return the power, False for complex wavelet transform. Default True.

            * cut_edge : bool
                Set to True to set points affected by edge effects to NaN, False to keep edge
                affect points. Default True

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Wavelet transform of the input.

    """

    # Time bounds
    start_time, end_time = inp.time.data[[0, -1]]
    start_time, end_time = [time.view("i8") * 1e-9 for time in [start_time, end_time]]

    # Time interval
    tint = end_time - start_time

    # Sampling frequency
    fs = len(inp) / tint

    # Unpack time and data
    t, data = [inp.time.data.view("i8") * 1e-9, inp.data]

    # f
    a_min, a_max = [0.01, 2]
    f_min, f_max = [.5 * fs / 10 ** a_max, .5 * fs / 10 ** a_min]

    # nf
    nf = 200

    # wavelet_width
    wavelet_width, delta_f = [5.36, 100]

    return_power, cut_edge, linear_df, plot_flag = [True, True, False, False]

    if kwargs.get("return_power"):
        return_power = kwargs["return_power"]

    if kwargs.get("cut_edge"):
        cut_edge = kwargs["cut_edge"]

    if kwargs.get("fs"):
        if isinstance(kwargs["fs"], (int, float)):
            fs = kwargs["fs"]
        else:
            raise TypeError("fs must be numeric")

    if kwargs.get("nf"):
        if isinstance(kwargs["nf"], (int, float)):
            nf = kwargs["nf"]
        else:
            raise TypeError("nf must be numeric")

    if kwargs.get("linear"):
        linear_df = True
        if isinstance(kwargs["linear"], (int, float)):
            delta_f = kwargs["linear"]
        else:
            raise Warning("Unknown input for linear delta_f set to 100")

    if kwargs.get("wavelet_width"):
        if isinstance(kwargs["wavelet_width"], (int, float)):
            wavelet_width = kwargs["wavelet_width"]
        else:
            raise TypeError("wavelet_width must be numeric")

    if kwargs.get("f"):
        if isinstance(kwargs["f"], (np.ndarray, list)):
            if len(kwargs["f"]) == 2:
                f_min = kwargs["f"][0]
                f_max = kwargs["f"][1]
            else:
                raise IndexError("f should have vector with 2 elements as parameter value")
        else:
            raise TypeError("f must be a list or array")

    if kwargs.get("plot"):
        plot_flag = kwargs["plot"]

    w0, a_number, sigma = [fs / 2, nf, wavelet_width / (fs / 2)]

    if linear_df:
        a_number = np.floor(w0 / delta_f).astype(int)

        f_min, f_max = [delta_f, a_number * delta_f]

        a = w0 / (np.linspace(f_max, f_min, a_number))
    else:
        a_min, a_max = [np.log10(.5 * fs / f_max), np.log10(.5 * fs / f_min)]

        a = np.logspace(a_min, a_max, a_number)

    # Remove the last sample if the total number of samples is odd
    if len(data) / 2 != np.floor(len(data) / 2):
        t, data = [t[:-1], data[:-1, ...]]

    # Check for NaNs
    a[np.isnan(a)] = 0

    # Find the frequencies for an FFT of all data
    nd2, nyq = [len(data) / 2, 1 / 2]

    freq = fs * np.arange(1, nd2 + 1) / nd2 * nyq

    # The frequencies corresponding to FFT
    w = np.hstack([0, freq, -np.flip(freq[:-1])])

    # Get the correct frequencies for the wavelet transform
    new_freq = w0 / a

    if len(inp.shape) == 1:
        out_dict, power2 = [None, np.zeros((len(inp.data), nf))]
    elif len(inp.shape) == 2:
        out_dict, power2 = [{}, None]
    else:
        raise TypeError("Invalid shape of the inp")

    new_freq_mat, _ = np.meshgrid(new_freq, w, sparse=True)

    _, ww = np.meshgrid(a, w, sparse=True)  # Matrix form

    # if scalar add virtual axis
    if len(inp.shape) == 1:
        data = data[:, np.newaxis]

    # go through all the data columns
    for i in range(data.shape[1]):
        # Make the FFT of all data
        data_col = data[:, i]

        # Wavelet transform of the data
        # Forward FFT
        s_w = pyfftw.interfaces.numpy_fft.fft(data_col, threads=mp.cpu_count())

        aa, s_ww = np.meshgrid(a, s_w, sparse=True)  # Matrix form

        # Calculate the FFT of the wavelet transform
        w_w = calc_w_w(s_ww, aa, sigma, ww, w0)

        # Backward FFT
        power = pyfftw.interfaces.numpy_fft.ifft(w_w, axis=0, threads=mp.cpu_count())

        # Calculate the power spectrum
        if return_power:
            power2 = calc_power_r(power, np.tile(new_freq_mat, (len(power), 1)))
        else:
            power2 = calc_power_c(power, np.tile(new_freq_mat, (len(power), 1)))

        if cut_edge:
            censure = np.floor(2 * a).astype(int)

            for j in range(a_number):
                power2[1:censure[j], j] = np.nan

                power2[len(data_col) - censure[j]:len(data_col), j] = np.nan
        else:
            continue

        if len(inp.shape) == 2:
            out_dict[inp.comp.data[i]] = (["time", "frequency"], power2)
        else:
            continue

    if len(inp.shape) == 1:
        out = xr.DataArray(power2, coords=[inp.time.data, new_freq],
                           dims=["time", "frequency"])
    elif len(inp.shape) == 2:
        out = xr.Dataset(out_dict,
                         coords={"time": inp.time.data, "frequency": new_freq})
    else:
        raise TypeError("Invalid shape")

    if plot_flag:
        if isinstance(out, xr.Dataset):
            fig, axs = plt.subplots(3, sharex="all")
            fig.subplots_adjust(hspace=0)
            axs[0].pcolormesh(out.time, out.frequency, out.x.data, cmap="jet")
            axs[0].set_yscale('log')
            axs[0].set_ylabel("f [Hz]")
            axs[1].pcolormesh(out.time, out.frequency, out.y.data, cmap="jet")
            axs[1].set_yscale('log')
            axs[1].set_ylabel("f [Hz]")
            axs[2].pcolormesh(out.time, out.frequency, out.z.data, cmap="jet")
            axs[2].set_yscale('log')
            axs[2].set_ylabel("f [Hz]")
        else:
            fig, axs = plt.subplots(1)
            fig.subplots_adjust(hspace=0)
            axs.pcolormesh(out.time, out.frequency, out.data.T, cmap="jet")
            axs.set_yscale('log')
            axs.set_ylabel("f [Hz]")

    return out
