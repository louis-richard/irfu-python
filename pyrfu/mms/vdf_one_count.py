#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.mms._internal.vdf_bins import _check_same_bins, _like


def vdf_one_count(vdf, counts):
    r"""One-count level of a distribution: the distribution of one count in
    each bin, f / N.

    For a measured distribution, it equals delta f^2 / f (irfu-matlab
    Example_MMS_ePSD_1count_level). For the time average of M distributions
    (``pyrf.average_vdf``) with their counts summed (``method="sum"``), it is
    the level of one count recorded in the window: the mean of the
    distributions of one count of the samples weighted by their counts,
    divided by M.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (or its time average).
    counts : xarray.Dataset
        Its counts (``vdf_counts``, or their sum over the same windows), at
        the same times.

    Returns
    -------
    one_count : xarray.Dataset
        One-count level, in the units of `vdf`; NaN where there are no counts.

    Raises
    ------
    TypeError
        If `vdf` or `counts` is not a xarray.Dataset.
    ValueError
        If `counts` does not have the times and the shape of `vdf`.

    See Also
    --------
    pyrfu.mms.vdf_counts, pyrfu.mms.vdf_error, pyrfu.pyrf.average_vdf

    Notes
    -----
    The one-count level of a bin changes with the bin and the time (by about
    25 % for DES and 10 % for DIS at a given energy, 2017-07-11); it is defined
    only where there are counts. Averaged over the angles, it is within 1 % of
    the DIS ``avgf1counts`` of the FPI L2 files and 0.75 of the DES one
    (v3.4.0, 2023-04-24, at every energy).

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> err_i = mms.get_data("pderri_fpi_brst_l2", tint, 1)
    >>> counts_i = mms.vdf_counts(vdf_i, err_i)
    >>> one_count_i = mms.vdf_one_count(vdf_i, counts_i)

    One-count level of the average of 5 distributions

    >>> vdf_avg = pyrf.average_vdf(vdf_i, 5)
    >>> counts_sum = pyrf.average_vdf(counts_i, 5, method="sum")
    >>> one_count_avg = mms.vdf_one_count(vdf_avg, counts_sum)

    """
    _check_same_bins(vdf, counts, "counts")

    f_data = np.asarray(vdf.data.data, dtype=np.float64)
    n_counts = np.asarray(counts.data.data, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        one_count = np.where(n_counts > 0.0, f_data / n_counts, np.nan)

    return _like(vdf, one_count, vdf.data.attrs.get("UNITS"))
