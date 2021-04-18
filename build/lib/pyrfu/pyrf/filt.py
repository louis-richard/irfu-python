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

"""filt.py
@author: Louis Richard
"""

import xarray as xr
import numpy as np

from scipy import signal


# noinspection PyTupleAssignmentBalance
def filt(inp, f_min=0., f_max=1., order=-1):
    """Filters input quantity.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the variable to filter.

    f_min : float
        Lower limit of the frequency range.

    f_max : float
        Upper limit of the frequency range.

    order : int
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

    f_samp = 1 / (np.median(np.diff(inp.time)).astype(int) * 1e-9)

    # Data of the input
    inp_data = inp.data

    f_min, f_max = [f_min / (f_samp / 2), f_max / (f_samp / 2)]

    f_max = np.min([f_max, 1.])

    # Parameters of the elliptic filter. fact defines the width between
    # stopband and passband
    # r_pass, r_stop, fact = [0.5, 60, 1.1]

    if f_min == 0:
        num1, den1, num2, den2 = [None] * 4

        if order == -1:
            order, f_max = signal.ellipord(f_max,
                                           np.min([f_max * 1.1, 0.9999]),
                                           .5, 60)

        num, den = signal.ellip(order, .5, 60, f_max, btype="lowpass")
    elif f_max == 0:
        num1, den1, num2, den2 = [None] * 4

        if order == -1:
            order, f_min = signal.ellipord(f_min,
                                           np.min([f_min * 1.1, 0.9999]),
                                           .5, 60)

        num, den = signal.ellip(order, .5, 60, f_min, btype="highpass")
    else:
        num, den = [None] * 2

        if order == -1:
            order, f_max = signal.ellipord(f_max,
                                           np.min([f_max * 1.3, 0.9999]),
                                           .5, 60)

        num1, den1 = signal.ellip(order, .5, 60, f_max)

        if order == -1:
            order, f_min = signal.ellipord(f_min, f_min * .75, .5, 60)

        num2, den2 = signal.ellip(order, .5, 60, f_min)

    if len(inp_data.shape) == 1:
        inp_data = inp_data[:, np.newaxis]

    out_data = np.zeros(inp_data.shape)

    for i_col in range(inp_data.shape[1]):
        if f_min != 0 and f_max != 0:
            out_data[:, i_col] = signal.filtfilt(num1, den1,
                                                 inp_data[:, i_col])
            out_data[:, i_col] = signal.filtfilt(num2, den2,
                                                 out_data[:, i_col])
        else:
            out_data[:, i_col] = signal.filtfilt(num, den, inp_data[:, i_col])

    if inp_data.shape[1] == 1:
        out_data = out_data[:, 0]

    out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims,
                       attrs=inp.attrs)

    return out
