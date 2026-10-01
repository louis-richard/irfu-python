#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import signal

# local imports
from pyrfu.pyrf.calc_fs import calc_fs

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


# noinspection PyTupleAssignmentBalance
def _ellip_coefficients(f_min, f_max, order):
    # Filters are designed as second-order sections (sos) rather than
    # transfer-function polynomials (b, a), which are numerically unstable
    # for high orders and low normalised cutoffs.
    sos1, sos2 = [None] * 2

    # fact defines the width between stopband and passband
    r_p, r_s, fact = 0.5, 60, 1.1

    if f_min == 0:
        if order == -1:
            order, f_max = signal.ellipord(
                f_max,
                np.min([f_max * fact, 0.9999]),
                r_p,
                r_s,
            )

        sos1 = signal.ellip(order, r_p, r_s, f_max, btype="lowpass", output="sos")
    elif f_max == 0:
        if order == -1:
            order, f_min = signal.ellipord(
                f_min,
                np.min([f_min * fact, 0.9999]),
                r_p,
                r_s,
            )

        sos1 = signal.ellip(order, r_p, r_s, f_min, btype="highpass", output="sos")
    else:
        if order == -1:
            order1, f_max = signal.ellipord(
                f_max,
                np.min([f_max * 1.3, 0.9999]),
                0.5,
                60,
            )
            order2, f_min = signal.ellipord(f_min, f_min * 0.75, 0.5, 60)
        else:
            order1, order2 = [order, order]

        sos1 = signal.ellip(order1, 0.5, 60, f_max, btype="lowpass", output="sos")
        sos2 = signal.ellip(order2, 0.5, 60, f_min, btype="highpass", output="sos")

    return sos1, sos2


def _sos_padlen(sos):
    # Same padding as MATLAB's filtfilt (and the former (b, a) implementation):
    # 3 * filter order. Odd orders have one section with b2 = a2 = 0.
    n_trailing = min(np.sum(sos[:, 2] == 0), np.sum(sos[:, 5] == 0))
    return 3 * (2 * len(sos) - n_trailing)


def filt(inp, f_min: float = 0.0, f_max: float = 1.0, order: int = -1):
    r"""Filters input quantity.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the variable to filter.
    f_min : float, Optional
        Lower limit of the frequency range. Default is 0. (Highpass filter).
    f_max : float, Optional
        Upper limit of the frequency range. Default is 1. (Highpass filter).
    order : int, Optional
        Order of the elliptic filter. Default is -1.

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

    assert isinstance(inp, xr.DataArray), "inp must be a xarray.DataArray"

    # Data of the input
    inp_data = inp.data.astype(np.float64)

    assert isinstance(f_min, (int, float)), "f_min must be int or float"
    assert isinstance(f_max, (int, float)), "f_max must be int or float"
    assert isinstance(order, (int, float)), "order must be int or float"

    if f_min == 0.0 and f_max == 0.0:
        raise ValueError("f_min and f_max cannot both be 0.0!")

    # Calculate the sampling frequency and normalize the cutoff frequencies
    # to the Nyquist frequency
    f_samp = calc_fs(inp)
    f_min, f_max = [f_min / (f_samp / 2.0), f_max / (f_samp / 2.0)]

    if f_min >= 1.0:
        raise ValueError("f_min must be smaller than the Nyquist frequency!")

    # An upper cutoff at or above the Nyquist frequency is no cutoff at all
    # (and Wn = 1 is invalid for the elliptic filter design) -> highpass.
    if f_max >= 1.0:
        f_max = 0.0

        if f_min == 0.0:
            raise ValueError("f_max >= Nyquist frequency with f_min = 0 does nothing!")

    # Parameters of the elliptic filter. fact defines the width between
    # stopband and passband
    # r_pass, r_stop, fact = [0.5, 60, 1.1]

    sos1, sos2 = _ellip_coefficients(f_min, f_max, int(order))

    if len(inp_data.shape) == 1:
        inp_data = inp_data[:, np.newaxis]
    elif len(inp_data.shape) == 3:
        inp_data = inp_data.reshape(inp_data.shape[0], -1)
    elif len(inp_data.shape) != 2:
        raise ValueError("inp must be 1D, 2D or 3D")

    # use odd padding with padlen = 3 * order for consistency with MATLAB
    out_data = signal.sosfiltfilt(
        sos1, inp_data, axis=0, padtype="odd", padlen=_sos_padlen(sos1)
    )

    if sos2 is not None:
        out_data = signal.sosfiltfilt(
            sos2, out_data, axis=0, padtype="odd", padlen=_sos_padlen(sos2)
        )

    if inp.ndim == 1:
        out_data = out_data[:, 0]
    elif len(inp.shape) == 3:
        out_data = out_data.reshape(inp.shape)

    out = xr.DataArray(
        out_data,
        coords=inp.coords,
        dims=inp.dims,
        attrs=inp.attrs,
    )

    return out
