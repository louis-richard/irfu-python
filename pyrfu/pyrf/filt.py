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

import xarray as xr
import numpy as np

from scipy import signal


# noinspection PyTupleAssignmentBalance
def filt(inp, f_min=0., f_max=1., n=-1):
    """Filters input quantity.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the variable to filter.

    f_min : float
        Lower limit of the frequency range.

    f_max : float
        Upper limit of the frequency range.

    n : int
        Order of the elliptic filter.

    Returns
    -------
    out : xarray.DataArray
        Time series of the filtered signal.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2017-07-18T13:03:34.000", "2017-07-18T13:07:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic and electric fields

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

    Convert E to field aligned coordinates

    >>> e_xyzfac = pyrf.convert_fac(e_xyz, b_xyz, [1,0,0])

    Bandpass filter E waveform

    >>> e_xyzfac_hf = pyrf.filt(e_xyzfac, 4, 0, 3)
    >>> e_xyzfac_lf = pyrf.filt(e_xyzfac, 0, 4, 3)

    """

    fs = 1 / (np.median(np.diff(inp.time)).astype(int) * 1e-9)

    # Data of the input
    inp_data = inp.data

    f_min, f_max = [f_min / (fs / 2), f_max / (fs / 2)]

    if f_max > 1:
        f_max = 1

    # Parameters of the elliptic filter. fact defines the width between stopband and passband
    r_p, r_s, fact = [0.5, 60, 1.1]

    if f_min == 0:
        b1, a1 = [None] * 2
        b2, a2 = [None] * 2

        if n == -1:
            n, f_max = signal.ellipord(f_max, np.min([f_max * fact, 0.9999]), r_p, r_s)

        b, a = signal.ellip(n, r_p, r_s, f_max, btype="lowpass")
    elif f_max == 0:
        b1, a1 = [None] * 2
        b2, a2 = [None] * 2

        if n == -1:
            n, f_min = signal.ellipord(f_min, np.min([f_min * fact, 0.9999]), r_p, r_s)

        b, a = signal.ellip(n, r_p, r_s, f_min, btype="highpass")
    else:
        b, a = [None] * 2

        if n == -1:
            n, f_max = signal.ellipord(f_max, np.min([f_max * 1.3, 0.9999]), r_p, r_s)

        b1, a1 = signal.ellip(n, r_p, r_s, f_max)

        if n == -1:
            n, f_min = signal.ellipord(f_min, f_min * .75, r_p, r_s)

        b2, a2 = signal.ellip(n, r_p, r_s, f_min)

    try:
        n_c = inp_data.shape[1]
    except IndexError:
        n_c = 1
        inp_data = inp_data[:, np.newaxis]

    out_data = np.zeros(inp_data.shape)

    if f_min != 0 and f_max != 0:
        for i_col in range(n_c):
            out_data[:, i_col] = signal.filtfilt(b1, a1, inp_data[:, i_col])
            out_data[:, i_col] = signal.filtfilt(b2, a2, out_data[:, i_col])
    else:
        for i_col in range(n_c):
            out_data[:, i_col] = signal.filtfilt(b, a, inp_data[:, i_col])

    if n_c == 1:
        out_data = out_data[:, 0]

    out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims, attrs=inp.attrs)

    return out
