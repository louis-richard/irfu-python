#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .mean_bins import _bin_stat

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def median_bins(inp0, inp1, bins: int = 10):
    r"""Computes median of values of y corresponding to bins of x

    Parameters
    ----------
    inp0 : xarray.DataArray
        Time series of the quantity x that defines the bins.
    inp1 : xarray.DataArray
        Time series of the quantity y to compute the binned median of, paired
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
                Median values of y corresponding to each bin of x.
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

    Median value of J for 10 bins of B

    >>> med_b_j = pyrf.median_bins(b_mag, j_mag)

    """

    assert isinstance(inp0, xr.DataArray), "inp0 must be xarray.DataArray"
    assert isinstance(inp1, xr.DataArray), "inp1 must be xarray.DataArray"

    assert inp0.ndim == 1, "inp0 must be a scalar"
    assert inp1.ndim == 1, "inp1 must be a scalar"

    out = _bin_stat(inp0, inp1, bins, np.median)

    return out
