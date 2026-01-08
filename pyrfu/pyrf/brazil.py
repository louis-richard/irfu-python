#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.pyrf.histogram2d import histogram2d
from pyrfu.pyrf.optimize_nbins_2d import optimize_nbins_2d

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2025"
__license__ = "MIT"
__version__ = "2.4.14"
__status__ = "Prototype"


def brazil(
    beta_para: np.ndarray,
    p_aniso: np.ndarray,
    bins: list = None,
    threshold: int = 9,
    **kwargs,
):
    """
    Computes 2D histogram and PDF (Brazil plot style) for plasma data.

    Parameters
    ----------
    beta_para : np.ndarray
        Parallel beta values (must be positive and finite).
    p_aniso : np.ndarray
        Temperature anisotropy values (must be positive and finite).
    bins : list, optional
        Bin edges or number of bins for the histogram. If None, optimized
        bin count is used.
    threshold : int, optional
        Minimum count threshold for masking low-counts (default is 9).

    Returns
    -------
    n : xarray.DataArray
        2D histogram counts with low-counts masked (NaN for <9).
    h : xarray.DataArray
        2D probability density function with low-counts masked.
    """
    # Valid data mask
    valid = (
        np.isfinite(beta_para) & (beta_para > 0) & np.isfinite(p_aniso) & (p_aniso > 0)
    )
    beta_para = beta_para[valid]
    p_aniso = p_aniso[valid]

    log_beta = np.log10(beta_para)
    log_aniso = np.log10(p_aniso)

    if bins is None:
        bins = optimize_nbins_2d(log_beta, log_aniso, **kwargs)

    # Compute histogram edges
    _, x_edges, y_edges = np.histogram2d(log_beta, log_aniso, bins=bins, density=True)

    # Compute bin centers in linear space
    x_centers = 10 ** (x_edges[:-1] + np.diff(x_edges) / 2)
    y_centers = 10 ** (y_edges[:-1] + np.diff(y_edges) / 2)

    # Histogram counts (not normalized)
    n = histogram2d(beta_para, p_aniso, bins=[10**x_edges, 10**y_edges], density=False)
    n = n.assign_coords({"x_bins": x_centers, "y_bins": y_centers})
    n.data[n.data == 0] = np.nan  # mask zero counts

    # Histogram PDF (normalized)
    h = histogram2d(beta_para, p_aniso, bins=[10**x_edges, 10**y_edges], density=True)
    h = h.assign_coords({"x_bins": x_centers, "y_bins": y_centers})
    h.data[n.data < threshold] = np.nan  # apply count threshold mask to PDF

    return n, h
