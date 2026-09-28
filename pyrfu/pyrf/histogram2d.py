#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.pyrf.resample import resample

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2026"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


SCALES = ["linlin", "loglin", "linlog", "loglog"]


def _bin_edges(data, bins, bin_range, log):
    r"""Computes the bin edges along one dimension.

    Parameters
    ----------
    data : numpy.ndarray
        Values to be histogrammed along this dimension.
    bins : int or array_like
        Number of bins, or explicit bin edges (returned unchanged).
    bin_range : array_like, shape(2,) or None
        Leftmost and rightmost edges. If None, the extrema of `data` are used.
    log : bool
        If True, the bins are logarithmically spaced.

    Returns
    -------
    edges : numpy.ndarray
        Bin edges, shape (bins + 1,).

    """
    # Explicit bin edges are used as they are.
    if not isinstance(bins, (int, np.integer)):
        return np.asarray(bins, dtype=float)

    if bin_range is None:
        low, upp = np.min(data), np.max(data)
    else:
        low, upp = bin_range

    if log:
        assert low > 0, "logarithmic bins require a strictly positive range"
        return np.geomspace(low, upp, bins + 1)

    return np.linspace(low, upp, bins + 1)


def _bin_centers(edges, log):
    r"""Computes the bin centers from the bin edges.

    Parameters
    ----------
    edges : numpy.ndarray
        Bin edges, shape (n + 1,).
    log : bool
        If True, returns the geometric center of the bins, otherwise the
        arithmetic one.

    Returns
    -------
    centers : numpy.ndarray
        Bin centers, shape (n,).

    """
    if log:
        return np.sqrt(edges[:-1] * edges[1:])

    return 0.5 * (edges[:-1] + edges[1:])


def histogram2d(
    inp1, inp2, bins=100, y_range=None, weights=None, density=True, scale="linlin"
):
    r"""Computes 2d histogram of inp2 vs inp1 with nbins number of bins.

    Parameters
    ----------
    inp1 : xarray.DataArray
        Time series of the x coordinates of the points to be histogrammed.
    inp2 : xarray.DataArray
        Time series of the y coordinates of the points to be histogrammed.
    bins : int or array_like or [int, int] or [array, array], Optional
        Number of bins along both dimensions, number of bins along each
        dimension, or explicit bin edges. Default is ``bins=100``.
    y_range : array_like, shape(2,2), Optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
        will be considered outliers and not tallied in the histogram.
    weights : array_like, shape(N,), Optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
        Weights are normalized to 1 if `density` is True. If `density` is
        False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.
    density : bool, Optional
        If False, returns the number of samples in each bin. If True, the
        default, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_area``.
    scale : {"linlin", "loglin", "linlog", "loglog"}, Optional
        Spacing of the bins along the x and y dimensions, in this order.
        Default is ``scale="linlin"`` (linear bins along both dimensions).
        A logarithmically spaced dimension only bins strictly positive
        values, the others are discarded.

    Returns
    -------
    out : xarray.DataArray
        2D map of the density of ``inp2`` vs ``inp1``. The bin edges, the
        scale and the normalization are stored in the attributes.

    Notes
    -----
    Samples for which either coordinate is not finite (or is not strictly
    positive along a logarithmic dimension) are discarded, as is the
    corresponding weight.

    The bin centers are the arithmetic means of the bin edges along a linear
    dimension and their geometric means along a logarithmic one, so that they
    are at the center of the bins as they are displayed.

    The first dimension of the output (``x_bins``) is the one of ``inp1``.
    ``xarray`` plots the *last* dimension along the x axis, so use
    ``out.plot(x="x_bins")`` (or ``out.T.plot()``) to get ``inp2`` vs
    ``inp1``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_id = np.arange(1, 5)

    Load magnetic field and electric field

    >>> r_mms = [mms.get_data("r_gse", tint, i) for i in mms_id]
    >>> b_mms = [mms.get_data("b_gse_fgm_srvy_l2", tint, i) for i in mms_id]

    Compute current density, etc

    >>> j_xyz, _, b_xyz, _, _, _ = pyrf.c_4_j(r_mms, b_mms)

    Compute magnitude of B and J

    >>> b_mag = pyrf.norm(b_xyz)
    >>> j_mag = pyrf.norm(j_xyz)

    Histogram of J vs B

    >>> h2d_b_j = pyrf.histogram2d(b_mag, j_mag)

    Same histogram with logarithmically spaced bins along both dimensions

    >>> h2d_b_j = pyrf.histogram2d(b_mag, j_mag, scale="loglog")

    """

    assert scale in SCALES, f"scale must be one of {SCALES}"

    log_x, log_y = scale[:3] == "log", scale[3:] == "log"

    # resample inp2 with respect to inp1
    if len(inp2) != len(inp1) or not np.array_equal(inp2.time.data, inp1.time.data):
        inp2 = resample(inp2, inp1)

    x_data, y_data = np.asarray(inp1.data), np.asarray(inp2.data)

    # Discard the samples that cannot be binned, numpy would otherwise fail to
    # compute the range of the histogram.
    keep = np.logical_and(np.isfinite(x_data), np.isfinite(y_data))

    if log_x:
        keep = np.logical_and(keep, x_data > 0)

    if log_y:
        keep = np.logical_and(keep, y_data > 0)

    if not np.all(keep):
        logging.info(
            "Discarding %d/%d samples in histogram2d",
            np.sum(~keep),
            len(keep),
        )

    x_data, y_data = x_data[keep], y_data[keep]

    if weights is not None:
        weights = np.asarray(weights)[keep]

    # A 0-d array is a number of bins
    if isinstance(bins, np.ndarray) and bins.ndim == 0:
        bins = bins.item()

    # Split the bins and the ranges between the two dimensions. As in
    # np.histogram2d, any length-2 bins (including a numpy array) is
    # [bins_x, bins_y].
    if isinstance(bins, (list, tuple, np.ndarray)) and len(bins) == 2:
        bins_x, bins_y = bins
    else:
        bins_x, bins_y = bins, bins

    range_x, range_y = (None, None) if y_range is None else y_range

    x_edges = _bin_edges(x_data, bins_x, range_x, log_x)
    y_edges = _bin_edges(y_data, bins_y, range_y, log_y)

    h2d, x_edges, y_edges = np.histogram2d(
        x_data,
        y_data,
        bins=[x_edges, y_edges],
        density=density,
        weights=weights,
    )

    x_bins = _bin_centers(x_edges, log_x)
    y_bins = _bin_centers(y_edges, log_y)

    out = xr.DataArray(
        h2d,
        coords=[x_bins, y_bins],
        dims=["x_bins", "y_bins"],
        attrs={
            "scale": scale,
            "density": density,
            "n_samples": int(np.sum(keep)),
            "x_edges": x_edges,
            "y_edges": y_edges,
        },
    )

    return out
