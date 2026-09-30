#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party import
import numba
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


# No fastmath: it lets LLVM assume there are no NaNs, so np.nanmean would not
# skip them (NaN windows at data gaps and in the cone of influence)
@numba.jit(cache=True, fastmath=False, nopython=True, parallel=True)
def _compress_cwt_1d(cwt, idxs, nc):
    nf = cwt.shape[1]

    cwt_c = np.zeros((len(idxs), nf))

    for i in numba.prange(len(idxs)):
        idx = idxs[i]
        for j in range(nf):
            # Block of nc time steps starting at idx
            x_data = cwt[idx : idx + nc, j]
            cwt_c[i, j] = np.nanmean(x_data)

    return cwt_c


def compress_cwt(cwt, nc: int = 100):
    r"""Compress the wavelet transform averaging over blocks of nc time steps.

    Parameters
    ----------
    cwt : xarray.Dataset
        Wavelet transform to compress.
    nc : int, Optional
        Number of time steps for averaging. The time series is split into
        len(time) // nc consecutive blocks; the remaining samples at the end are
        dropped. NaNs are ignored in the averages. Default is 100.

    Returns
    -------
    cwt_t : numpy.ndarray
        Times of the centres of the blocks.
    cwt_x : ndarray
        Compressed wavelet transform of the first component of the field.
    cwt_y : ndarray
        Compressed wavelet transform of the second component of the field.
    cwt_z : ndarray
        Compressed wavelet transform of the third component of the field.

    """

    assert isinstance(cwt, xr.Dataset), "cwt must be an xarray.Dataset"

    # First time step of each block
    indices = np.arange(len(cwt.time.data) // nc, dtype=np.int64) * nc

    # Time at the centre of each block
    times = cwt.time.data
    cwt_t = times[indices] + (times[indices + nc - 1] - times[indices]) / 2

    cwt_x = _compress_cwt_1d(cwt.x.data, indices, nc=nc)
    cwt_y = _compress_cwt_1d(cwt.y.data, indices, nc=nc)
    cwt_z = _compress_cwt_1d(cwt.z.data, indices, nc=nc)

    return cwt_t, cwt_x, cwt_y, cwt_z
