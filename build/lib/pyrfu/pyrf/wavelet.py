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

"""wavelet.py
@author: Louis Richard
"""

import multiprocessing as mp
import numpy as np
import xarray as xr
import pyfftw
import numba


@numba.jit(nopython=True, fastmath=True)
def calc_w_w(s_ww, scales_mat, sigma, frequencies_mat, f_nyq):
    """faster product using numba"""
    w_w = s_ww * np.exp(-sigma * sigma
                        * ((scales_mat * frequencies_mat - f_nyq) ** 2) / 2)
    w_w = w_w * np.sqrt(1)
    return w_w


@numba.jit(nopython=True, parallel=True, fastmath=True)
def calc_power_r(power, new_freq_mat):
    """faster product using numba"""
    power2 = np.absolute((2 * np.pi) * np.conj(power) * power / new_freq_mat)
    return power2


@numba.jit(nopython=True, parallel=True, fastmath=True)
def calc_power_c(power, new_freq_mat):
    """faster product using numba"""
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
                Vector [f_min f_max], calculate spectra between frequencies
                f_min and f_max.

            * nf : int or float
                Number of frequency bins.

            * wavelet_width : int or float
                Width of the Morlet wavelet. Default 5.36.

            * linear : float
                Linear spacing between frequencies of df.

            * return_power : bool
                Set to True to return the power, False for complex wavelet
                transform. Default True.

            * cut_edge : bool
                Set to True to set points affected by edge effects to NaN,
                False to keep edge affect points. Default True

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Wavelet transform of the input.

    """

    # Time bounds
    start_time = inp.time.data[0].view("i8") * 1e-9
    end_time = inp.time.data[1].view("i8") * 1e-9

    # Time interval
    tint = end_time - start_time

    # Sampling frequency
    f_s = len(inp) / tint

    # Unpack time and data
    time, data = [inp.time.data.view("i8") * 1e-9, inp.data]

    # f
    scale_min, scale_max = [0.01, 2]
    f_min, f_max = [.5 * f_s / 10 ** scale_max, .5 * f_s / 10 ** scale_min]

    # nf
    n_freqs = 200

    # wavelet_width
    wavelet_width, delta_f = [5.36, 100]

    return_power, cut_edge, linear_df = [True, True, False]

    if kwargs.get("return_power"):
        return_power = kwargs["return_power"]

    if kwargs.get("cut_edge"):
        cut_edge = kwargs["cut_edge"]

    if kwargs.get("fs"):
        assert isinstance(kwargs["fs"], (int, float)), "fs must be numeric"
        f_s = kwargs["fs"]

    if kwargs.get("nf"):
        assert isinstance(kwargs["nf"], (int, float)), "nf must be numeric"
        n_freqs = kwargs["nf"]

    if kwargs.get("linear"):
        linear_df = True
        if isinstance(kwargs["linear"], (int, float)):
            delta_f = kwargs["linear"]
        else:
            raise Warning("Unknown input for linear delta_f set to 100")

    if kwargs.get("wavelet_width"):
        assert isinstance(kwargs["wavelet_width"], (int, float))
        wavelet_width = kwargs["wavelet_width"]

    if kwargs.get("f"):
        assert isinstance(kwargs["f"], (np.ndarray, list))
        assert len(kwargs["f"]) == 2
        f_min, f_max = kwargs["f"]

    f_nyq, scale_number, sigma = [f_s / 2, n_freqs, wavelet_width / (f_s / 2)]

    if linear_df:
        scale_number = np.floor(f_nyq / delta_f).astype(int)

        f_min, f_max = [delta_f, scale_number * delta_f]

        scales = f_nyq / (np.linspace(f_max, f_min, scale_number))
    else:
        scale_min = np.log10(.5 * f_s / f_max)
        scale_max = np.log10(.5 * f_s / f_min)
        scales = np.logspace(scale_min, scale_max, scale_number)

    # Remove the last sample if the total number of samples is odd
    if len(data) / 2 != np.floor(len(data) / 2):
        time, data = [time[:-1], data[:-1, ...]]

    # Check for NaNs
    scales[np.isnan(scales)] = 0

    # Find the frequencies for an FFT of all data
    freq = f_s * .5 * np.arange(1, 1 + len(data) / 2) / (len(data) / 2)

    # The frequencies corresponding to FFT
    frequencies = np.hstack([0, freq, -np.flip(freq[:-1])])

    # Get the correct frequencies for the wavelet transform
    new_freq = f_nyq / scales

    if len(inp.shape) == 1:
        out_dict, power2 = [None, np.zeros((len(inp.data), n_freqs))]
    elif len(inp.shape) == 2:
        out_dict, power2 = [{}, None]
    else:
        raise TypeError("Invalid shape of the inp")

    new_freq_mat, _ = np.meshgrid(new_freq, frequencies, sparse=True)

    _, frequencies_mat = np.meshgrid(scales, frequencies, sparse=True)

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

        scales_mat, s_w_mat = np.meshgrid(scales, s_w, sparse=True)

        # Calculate the FFT of the wavelet transform
        w_w = calc_w_w(s_w_mat, scales_mat, sigma, frequencies_mat, f_nyq)

        # Backward FFT
        power = pyfftw.interfaces.numpy_fft.ifft(w_w, axis=0,
                                                 threads=mp.cpu_count())

        # Calculate the power spectrum
        if return_power:
            power2 = calc_power_r(power,
                                  np.tile(new_freq_mat, (len(power), 1)))
        else:
            power2 = calc_power_c(power,
                                  np.tile(new_freq_mat, (len(power), 1)))

        if cut_edge:
            censure = np.floor(2 * scales).astype(int)

            for j in range(scale_number):
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

    return out
