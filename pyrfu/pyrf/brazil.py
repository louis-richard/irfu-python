#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.pyrf.optimize_nbins_2d import optimize_nbins_2d

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
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
    beta_para : array_like or xarray.DataArray
        Parallel beta values. Only the samples where both inputs are positive
        and finite are used.
    p_aniso : array_like or xarray.DataArray
        Temperature anisotropy values, paired sample by sample with
        `beta_para` (same length).
    bins : int or list, optional
        Number of bins, or bin edges, of log10(beta_para) and log10(p_aniso),
        as `bins` in numpy.histogram2d. If None, the number of bins is
        optimized with :func:`pyrfu.pyrf.optimize_nbins_2d`.
    threshold : int, optional
        Bins with fewer counts are masked in the PDF (default is 9).
    **kwargs
        Keyword arguments passed to :func:`pyrfu.pyrf.optimize_nbins_2d`.

    Returns
    -------
    n : xarray.DataArray
        2D histogram counts, NaN in the empty bins.
    h : xarray.DataArray
        2D probability density function per unit beta_para and p_aniso, NaN
        in the bins with fewer than `threshold` counts (and in the empty
        ones).

    Raises
    ------
    ValueError
        If `beta_para` and `p_aniso` have different lengths.

    Notes
    -----
    The bins are logarithmically spaced; the coordinates ``x_bins`` and
    ``y_bins`` are their geometric centers, and the edges are in the
    attributes ``x_edges`` and ``y_edges``.

    """
    beta_para = np.asarray(beta_para, dtype=np.float64)
    p_aniso = np.asarray(p_aniso, dtype=np.float64)

    if beta_para.shape != p_aniso.shape:
        raise ValueError(
            f"beta_para and p_aniso must have the same shape, got "
            f"{beta_para.shape} and {p_aniso.shape}"
        )

    # Valid data mask
    valid = (
        np.isfinite(beta_para) & (beta_para > 0) & np.isfinite(p_aniso) & (p_aniso > 0)
    )
    log_beta = np.log10(beta_para[valid])
    log_aniso = np.log10(p_aniso[valid])

    if bins is None:
        bins = optimize_nbins_2d(log_beta, log_aniso, **kwargs)

    # Counts in logarithmic bins
    counts, x_edges, y_edges = np.histogram2d(log_beta, log_aniso, bins=bins)
    x_edges, y_edges = [10**x_edges, 10**y_edges]

    # Geometric bin centers
    x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
    y_centers = np.sqrt(y_edges[:-1] * y_edges[1:])

    # Probability density per unit beta_para and p_aniso
    bin_area = np.outer(np.diff(x_edges), np.diff(y_edges))
    pdf = counts / np.sum(counts) / bin_area

    attrs = {"x_edges": x_edges, "y_edges": y_edges, "n_samples": int(np.sum(valid))}
    n = xr.DataArray(
        np.where(counts > 0, counts, np.nan),
        coords=[x_centers, y_centers],
        dims=["x_bins", "y_bins"],
        attrs=attrs,
    )
    h = xr.DataArray(
        np.where((counts >= threshold) & (counts > 0), pdf, np.nan),
        coords=[x_centers, y_centers],
        dims=["x_bins", "y_bins"],
        attrs=dict(attrs),
    )

    return n, h
