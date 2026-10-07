#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from scipy import ndimage

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

WEIGHTS = {
    5: np.array([0.1, 0.25, 0.3, 0.25, 0.1]),
    7: np.array([0.07, 0.15, 0.18, 0.2, 0.18, 0.15, 0.07]),
}


def waverage(inp, f_sampl: float = None, n_pts: int = 7):
    r"""Computes weighted average.

    Port of irf_waverage.m: the data are put on a regular time grid at the
    sampling frequency, and each point is replaced by the weighted average of
    the n_pts points centred on it. Missing points (gaps, NaNs and zeros) are
    left out and the weights renormalised; the average is 0 if all the points
    are missing.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input data.
    f_sampl : float, Optional
        Sampling frequency. Default is from the first two time steps.
    n_pts : int, Optional
        Number of point to average over 5 ot 7. Default is 7

    Returns
    -------
    out : xarray.DataArray
        Weighted averaged of inp, at the times of inp.

    """

    if n_pts not in WEIGHTS:
        raise ValueError("n_pts must be 5 or 7")

    if len(inp) <= 1:
        return inp.copy()

    t_sec = (inp.time.data - inp.time.data[0]) / np.timedelta64(1, "s")

    if f_sampl is None:
        f_sampl = 1 / t_sec[1]

    n_data = int(np.round(t_sec[-1] * f_sampl))
    delta_t = t_sec[-1] / n_data

    # Data on the regular time grid, missing points set to zero
    inp_data = inp.data.reshape(len(inp), -1).astype(np.float64)
    indices = np.round(t_sec / delta_t).astype(np.int64)

    out = np.zeros((n_data + 1, inp_data.shape[1]))
    out[indices, :] = inp_data
    out[np.isnan(out)] = 0  # set NaNs to zeros

    # Weighted sums of the points and of the weights of the non-missing points
    weights = WEIGHTS[n_pts]
    options = {"axis": 0, "mode": "constant", "cval": 0.0}
    num = ndimage.convolve1d(out, weights, **options)
    den = ndimage.convolve1d((out != 0).astype(np.float64), weights, **options)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 1e-12, num / den, 0.0)

    # Make sure we do return matrix of the same size
    return inp.copy(data=out[indices, :].reshape(inp.shape))
