#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _bin_stat(inp0, inp1, bins, func):
    # Pairs with a non-finite x or y are left out
    finite = np.isfinite(inp0.data) & np.isfinite(inp1.data)

    if not np.any(finite):
        raise ValueError("inp0 and inp1 have no finite pair of values")

    x_data, y_data = [inp0.data[finite], inp1.data[finite]]

    # Same edges as numpy.histogram: bins uniform bins from min(x) to max(x)
    # (min(x) - 0.5 to max(x) + 0.5 if x is constant)
    x_edge = np.histogram_bin_edges(x_data, bins)

    # Bin i holds x_edge[i] <= x < x_edge[i + 1], the last one also max(x)
    idx_bin = np.clip(np.digitize(x_data, x_edge) - 1, 0, bins - 1)

    y_stat, y_std = [np.full(bins, np.nan), np.full(bins, np.nan)]

    for i in range(bins):
        y_bin = y_data[idx_bin == i]

        if y_bin.size:
            y_stat[i], y_std[i] = [func(y_bin), np.std(y_bin)]

    out_dict = {
        "data": (["bins"], y_stat),
        "sigma": (["bins"], y_std),
        "bins": (x_edge[:-1] + x_edge[1:]) / 2,
    }

    return xr.Dataset(out_dict)


def mean_bins(inp0, inp1, bins: int = 10):
    r"""Computes mean of values of y corresponding to bins of x.

    Parameters
    ----------
    inp0 : xarray.DataArray
        Time series of the quantity x that defines the bins.
    inp1 : xarray.DataArray
        Time series of the quantity y to compute the binned mean of, paired
        with inp0 sample by sample.
    bins : int, Optional
        Number of bins. Default is 10.

    Returns
    -------
    out : xarray.Dataset
        Dataset with :
            * bins : xarray.DataArray
                bin values of the x variable.
            * data : xarray.DataArray
                Mean values of y corresponding to each bin of x.
            * sigma : xarray.DataArray
                Standard deviation.

    Raises
    ------
    ValueError
        If no pair of inp0 and inp1 values is finite.

    Notes
    -----
    The bins are those of numpy.histogram: uniform from min(x) to max(x),
    each closed on the left and the last one also on the right, so that
    every sample falls in a bin. Pairs with a non-finite x or y are left out.
    Empty bins are NaN.

    Examples
    --------
    >>> import numpy
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_list = numpy.arange(1,5)

    Load magnetic field and electric field

    >>> r_mms, b_mms = [[] * 4 for _ in range(2)]
    >>> for mms_id in range(1, 5):
    >>> 	r_mms.append(mms.get_data("R_gse", tint, mms_id))
    >>> 	b_mms.append(mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id))
    >>>

    Compute current density, etc

    >>> j_xyz, _, b_xyz, _, _, _ = pyrf.c_4_j(r_mms, b_mms)

    Compute magnitude of B and J

    >>> b_mag = pyrf.norm(b_xyz)
    >>> j_mag = pyrf.norm(j_xyz)

    Mean value of J for 10 bins of B

    >>> m_b_j = pyrf.mean_bins(b_mag, j_mag)

    """

    assert isinstance(inp0, xr.DataArray), "inp0 must be xarray.DataArray"
    assert isinstance(inp1, xr.DataArray), "inp1 must be xarray.DataArray"

    assert inp0.ndim == 1, "inp0 must be a scalar"
    assert inp1.ndim == 1, "inp1 must be a scalar"

    out = _bin_stat(inp0, inp1, bins, np.mean)

    return out
