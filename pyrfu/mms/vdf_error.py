#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.mms._internal.vdf_bins import _check_same_bins, _like, _units, _values


def vdf_error(vdf, counts):
    r"""Poisson error of a distribution from its counts, f / sqrt(N).

    Applied to the time average of M distributions (``pyrf.average_vdf``) and
    their counts summed over the same windows (``method="sum"``), it gives
    the error of the average, f_avg / sqrt(N_sum).

    Parameters
    ----------
    vdf : xarray.Dataset or xarray.DataArray
        Skymap distribution (or its time average), or a spectrum, e.g. the
        FEEPS omni-directional intensity (`pyrfu.mms.feeps_omni`).
    counts : xarray.Dataset or xarray.DataArray
        Its counts (``vdf_counts``, or their sum over the same windows; for
        FEEPS, the "counts" of `pyrfu.mms.feeps_omni`), at the same times.

    Returns
    -------
    vdf_err : xarray.Dataset
        Error of the distribution, in the units of `vdf`; 0 where there are no
        counts, as the FPI L2 error distributions.

    Raises
    ------
    TypeError
        If `vdf` and `counts` are not both xarray.Dataset or both
        xarray.DataArray.
    ValueError
        If `counts` does not have the times and the shape of `vdf`.

    See Also
    --------
    pyrfu.mms.vdf_counts, pyrfu.mms.vdf_one_count, pyrfu.pyrf.average_vdf

    Notes
    -----
    The FPI fast survey distributions are lossy compressed (attribute
    ``compressionloss`` of `get_dist`): their counts are not Poisson where
    they are low, and do not go below a floor of about 2 per bin (Barrie et
    al., 2017; see `get_dist`). The errors of the bins with fewer true counts
    are then wrong; those of the burst distributions, mostly lossless, are
    not.

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> err_i = mms.get_data("pderri_fpi_brst_l2", tint, 1)
    >>> counts_i = mms.vdf_counts(vdf_i, err_i)
    >>> vdf_avg = pyrf.average_vdf(vdf_i, 5)
    >>> err_avg = mms.vdf_error(vdf_avg, pyrf.average_vdf(counts_i, 5, method="sum"))

    """
    _check_same_bins(vdf, counts, "counts")

    f_data = np.asarray(_values(vdf), dtype=np.float64)
    n_counts = np.asarray(_values(counts), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_err = np.where(n_counts > 0.0, f_data / np.sqrt(n_counts), 0.0)

    f_err = np.where(np.isnan(f_data) | np.isnan(n_counts), np.nan, f_err)

    return _like(vdf, f_err, _units(vdf))
