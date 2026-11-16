#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.mms._internal.vdf_bins import _check_same_bins, _like


def vdf_counts(vdf, vdf_err):
    r"""Counts of a distribution from its Poisson error.

    With Poisson statistics, the error of a bin of N counts is f / sqrt(N), so
    that N = (f / delta f)^2, as in irfu-matlab (``PDist.movmean``,
    ``PDist.find_low_counts``).

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution, e.g. ``mms.get_data("pdi_fpi_brst_l2", ...)``.
    vdf_err : xarray.Dataset
        Its error distribution, at the same times and in the same units, e.g.
        ``mms.get_data("pderri_fpi_brst_l2", ...)``.

    Returns
    -------
    counts : xarray.Dataset
        Distribution of the counts (``UNITS`` "counts"): (f / delta f)^2, 0
        where f is 0 (FPI gives delta f = 0 there), NaN where delta f is 0 and f
        is not, or where either is NaN.

    Raises
    ------
    TypeError
        If `vdf` or `vdf_err` is not a xarray.Dataset.
    ValueError
        If `vdf_err` does not have the times, the shape and the units of `vdf`.

    See Also
    --------
    pyrfu.mms.vdf_one_count, pyrfu.mms.vdf_error, pyrfu.pyrf.average_vdf

    Notes
    -----
    The counts of the FPI L2 distributions are close to whole numbers:
    (f / delta f)^2 is 1.0005 for a single count of DES and 1.0001 of DIS
    (2017-07-11), and grows slightly faster than N for more counts.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> err_i = mms.get_data("pderri_fpi_brst_l2", tint, 1)
    >>> counts_i = mms.vdf_counts(vdf_i, err_i)

    """
    _check_same_bins(vdf, vdf_err, "vdf_err", units=True)

    f_data = np.asarray(vdf.data.data, dtype=np.float64)
    f_err = np.asarray(vdf_err.data.data, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        counts = (f_data / f_err) ** 2

    counts = np.where(f_data == 0.0, 0.0, counts)
    counts = np.where(np.isfinite(counts), counts, np.nan)

    return _like(vdf, counts, "counts")
