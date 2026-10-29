#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import signal
from xarray.core.dataarray import DataArray

# Local imports
from .calc_fs import calc_fs

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Elliptic filter of irfu-matlab irf_lowpass: order, passband ripple (dB) and
# stopband attenuation (dB)
_ORDER, _R_PASS, _R_STOP = 4, 0.1, 60


def lowpass(
    inp,
    f_cut: float,
    f_s: Optional[float] = None,
    fhz: Optional[float] = None,
):
    r"""Low-pass filter, as irfu-matlab irf_lowpass: the linear trend is
    removed, the rest filtered forward and backward (zero phase) with a
    4th-order elliptic filter (0.1 dB ripple, 60 dB attenuation), and the
    trend added back.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable (scalar, vector or tensor).
    f_cut : float
        Cutoff frequency in Hz, between 0 and f_s / 2.
    f_s : float, Optional
        Sampling frequency of inp in Hz. Default is the sampling frequency of
        inp (``calc_fs``).
    fhz : float, Optional
        Former name of f_s, deprecated (FutureWarning; removed in 3.0).

    Returns
    -------
    out : xarray.DataArray
        Time series of the filtered data, with the attributes of inp.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.
    ValueError
        If f_cut is not between 0 and the Nyquist frequency, if both f_s and
        fhz are given, if inp has NaN or infinite values (they spread to the
        whole output: remove or interpolate them first), or if inp has 12
        samples or fewer.

    Notes
    -----
    The filter is applied in second-order sections, with the padding of
    MATLAB's filtfilt (12 samples, three times the order). The (b, a) form
    used before loses precision as f_cut decreases relative to the Nyquist
    frequency, and is unstable below about 3e-5 of it (e.g., 0.1 Hz at
    8192 Hz); its padding was 15 samples, which changed the first and last
    samples.

    """

    if not isinstance(inp, DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    if fhz is not None:
        warnings.warn(
            "lowpass: fhz is deprecated and will be removed in pyrfu 3.0, use f_s",
            FutureWarning,
            stacklevel=2,
        )

        if f_s is not None:
            raise ValueError("give f_s only (fhz is its deprecated name)")

        f_s = fhz

    if f_s is None:
        f_s = calc_fs(inp)

    if not 0 < f_cut < f_s / 2:
        raise ValueError(
            f"f_cut must be between 0 and the Nyquist frequency {f_s / 2} Hz, "
            f"not {f_cut} Hz"
        )

    data = np.asarray(inp.data, dtype=np.float64)

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "inp has NaN or infinite values, which spread to the whole output: "
            "remove or interpolate them first"
        )

    pad_len = 3 * _ORDER

    if len(data) <= pad_len:
        raise ValueError(f"inp must have more than {pad_len} samples")

    # Remove trend
    data_detrend = signal.detrend(data, type="linear", axis=0)
    rest = data - data_detrend

    # Elliptic filter, in second-order sections
    sos = signal.ellip(_ORDER, _R_PASS, _R_STOP, f_cut / (f_s / 2), output="sos")

    # Filter data, forward and backward, with the padding of MATLAB's filtfilt
    out_data = signal.sosfiltfilt(
        sos, data_detrend, axis=0, padtype="odd", padlen=pad_len
    )

    out = xr.DataArray(
        out_data + rest, coords=inp.coords, dims=inp.dims, attrs=inp.attrs
    )

    return out
