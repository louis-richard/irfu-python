#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.mms._internal.vdf_bins import _check_same_bins, _like, _values


def vdf_counts(vdf, vdf_err):
    r"""Counts of a distribution from its Poisson error.

    With Poisson statistics, the error of a bin of N counts is f / sqrt(N), so
    that N = (f / delta f)^2, as in irfu-matlab (``PDist.movmean``,
    ``PDist.find_low_counts``).

    Parameters
    ----------
    vdf : xarray.Dataset or xarray.DataArray
        Skymap distribution, e.g. ``mms.get_data("pdi_fpi_brst_l2", ...)``, or
        a spectrum (xarray.DataArray).
    vdf_err : xarray.Dataset or xarray.DataArray
        Its error distribution (spectrum), at the same times and in the same
        units, e.g. ``mms.get_data("pderri_fpi_brst_l2", ...)``.

    Returns
    -------
    counts : xarray.Dataset or xarray.DataArray
        Distribution (spectrum) of the counts (``UNITS`` "counts"): (f / delta f)^2, 0
        where f is 0 (FPI gives delta f = 0 there), NaN where delta f is 0 and f
        is not, or where either is NaN.

    Raises
    ------
    TypeError
        If `vdf` and `vdf_err` are not both xarray.Dataset or both
        xarray.DataArray.
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

    The FPI fast survey distributions are lossy compressed (attribute
    ``compressionloss`` of `get_dist`): their counts are not Poisson where
    they are low, and do not go below a floor of about 2 per bin (Barrie et
    al., 2017; see `get_dist`). The counts of the bins with fewer true counts
    are then wrong; those of the burst distributions, mostly lossless, are
    not.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> err_i = mms.get_data("pderri_fpi_brst_l2", tint, 1)
    >>> counts_i = mms.vdf_counts(vdf_i, err_i)

    """
    _check_same_bins(vdf, vdf_err, "vdf_err", units=True)

    f_data = np.asarray(_values(vdf), dtype=np.float64)
    f_err = np.asarray(_values(vdf_err), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        counts = (f_data / f_err) ** 2

    counts = np.where(f_data == 0.0, 0.0, counts)
    counts = np.where(np.isfinite(counts), counts, np.nan)

    return _like(vdf, counts, "counts")
