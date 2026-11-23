#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from xarray import Dataset

# Local imports
from ..pyrf.average_vdf import _average_windows


def vdf_movmean(vdf, n_pts: int, counts: Dataset | None = None):
    r"""Moving average of a skymap distribution in time, on its own times.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the skymap distribution.
    n_pts : int
        Number of distributions of the window (odd): each distribution is
        averaged with its (n_pts - 1) / 2 neighbours on each side, fewer at
        the ends of the time series.
    counts : xarray.Dataset, Optional
        Counts of `vdf`, at its times and on its bins (e.g. from
        `vdf_counts`). The bins with fewer than 2 counts summed over the
        window (one count or none) are set to 0. Default is None: no
        removal.

    Returns
    -------
    out : xarray.Dataset
        Time series of the averaged distribution, at the times of `vdf`, with
        the attributes of each window as `pyrfu.pyrf.average_vdf`: the
        energy widths averaged, the compression indicator (the largest) and
        the FPI quality flags (combined) of the window, and the accumulation
        half widths (``delta_time_minus/plus``) spanning it.

    Raises
    ------
    TypeError
        If `vdf` or `counts` is not a xarray.Dataset, or `n_pts` is not an
        integer.
    ValueError
        If `n_pts` is not odd or not between 1 and the number of
        distributions, or `counts` does not have the times and the shape of
        `vdf`.

    Notes
    -----
    The NaN bins have no data and are left out of the mean; a bin without
    data in a window is NaN (see :ref:`api-mms-vdf-nan`). The bins are
    averaged as they are: with alternating energy tables (FPI burst), the
    channel i of one table is averaged with the channel i of the other, at
    another energy, and the energies are averaged too; use `vdf_einterp`
    first to average on one table. `pyrfu.pyrf.average_vdf` averages over
    consecutive blocks instead, at a lower time resolution.

    See Also
    --------
    pyrfu.pyrf.average_vdf, pyrfu.mms.vdf_einterp, pyrfu.mms.vdf_counts

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 3)
    >>> err_i = mms.get_data("pderri_fpi_brst_l2", tint, 3)

    Average over 5 distributions, without the bins of one count or none

    >>> vdf_i5 = mms.vdf_movmean(vdf_i, 5, counts=mms.vdf_counts(vdf_i, err_i))

    """

    if not isinstance(vdf, Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    if not isinstance(n_pts, (int, np.integer)) or isinstance(n_pts, bool):
        raise TypeError("n_pts must be an integer")

    n_vdf = len(vdf.time)

    if not 1 <= n_pts <= n_vdf or n_pts % 2 == 0:
        raise ValueError(
            f"n_pts must be odd and between 1 and the number of distributions "
            f"({n_vdf})"
        )

    if counts is not None:
        if not isinstance(counts, Dataset):
            raise TypeError("counts must be a xarray.Dataset")

        if counts.data.shape != vdf.data.shape or not np.array_equal(
            counts.time.data, vdf.time.data
        ):
            raise ValueError("counts must have the times and the shape of vdf")

    # Windows centred on each distribution, shrinking at the ends
    half = n_pts // 2
    centres = np.arange(n_vdf)
    bounds = [(max(0, i - half), min(n_vdf, i + half + 1)) for i in centres]

    return _average_windows(vdf, bounds, centres, vdf.time.data, "mean", counts)
