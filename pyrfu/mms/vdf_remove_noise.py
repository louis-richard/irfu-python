#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import ndimage

# Local imports
from pyrfu.mms._internal.vdf_bins import _check_same_bins, _is_skymap


def vdf_remove_noise(vdf, counts, n_pts, n_thresh: float):
    r"""Sets to 0 the bins of a skymap whose neighbourhood has few counts.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the skymap distribution.
    counts : xarray.Dataset
        Counts of `vdf`, at its times and on its bins (e.g. from
        `vdf_counts`); NaN counts count as 0.
    n_pts : array_like of int
        Size of the neighbourhood of each bin (odd numbers): the number of
        times, energies, azimuths and polar angles, centred on the bin.
    n_thresh : float
        The bins whose neighbourhood has fewer counts are set to 0.

    Returns
    -------
    out : xarray.Dataset
        Copy of `vdf` with the bins of few counts in their neighbourhood set
        to 0.

    Raises
    ------
    TypeError
        If `vdf` and `counts` are not both xarray.Dataset.
    ValueError
        If `vdf` is not a skymap, `counts` does not have its times and shape,
        or `n_pts` is not four odd positive integers.

    Notes
    -----
    The counts are summed over the neighbourhood of each bin: around it in
    time, energy and polar angle, with fewer bins at the edges, and all
    around in azimuth, which wraps around. An isolated count (noise) has
    few counts around it, a population of particles many. The NaN bins have
    no data and stay NaN (see :ref:`api-mms-vdf-nan`).

    The method suits populations with many counts per bin. In tenuous
    plasma the particles themselves are in bins of few counts, and are
    removed with the noise: with neighbourhoods of 3 x 3 x 3 x 3 bins, the
    ion density of the FPI burst distributions changes by -0.2 %, -1.3 % and
    -2.8 % (thresholds 2, 5 and 10 counts) in the magnetosheath (10.6
    cm^-3, MMS1, 2015-10-16 13:06), but by -50 %, -85 % and -88 % in the
    plasma sheet (0.41 cm^-3, MMS3, 2017-07-11 22:34). Compare the moments
    before and after.

    See Also
    --------
    pyrfu.mms.vdf_counts, pyrfu.mms.vdf_find_low_counts,
    pyrfu.mms.vdf_movmean

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2015-10-16T13:06:00.000", "2015-10-16T13:06:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> err_i = mms.get_data("pderri_fpi_brst_l2", tint, 1)
    >>> counts_i = mms.vdf_counts(vdf_i, err_i)

    Bins with fewer than 5 counts in 3 times, 3 energies and 3 x 3 angles

    >>> vdf_clean = mms.vdf_remove_noise(vdf_i, counts_i, [3, 3, 3, 3], 5)

    """

    if not isinstance(vdf, xr.Dataset) or not isinstance(counts, xr.Dataset):
        raise TypeError("vdf and counts must be xarray.Dataset")

    if not _is_skymap(vdf):
        raise ValueError("vdf must be a skymap")

    _check_same_bins(vdf, counts, "counts")

    n_pts = np.asarray(n_pts)

    if (
        n_pts.shape != (4,)
        or not np.issubdtype(n_pts.dtype, np.integer)
        or np.any(n_pts < 1)
        or np.any(n_pts % 2 == 0)
    ):
        raise ValueError("n_pts must be four odd positive integers")

    # Counts of the neighbourhood: zeros beyond the edges, the azimuths
    # wrapping around
    summed = np.nan_to_num(np.asarray(counts.data.data, dtype=np.float64))

    for axis, (size, mode) in enumerate(
        zip(n_pts, ["constant"] * 2 + ["wrap", "constant"])
    ):
        summed = ndimage.correlate1d(summed, np.ones(size), axis=axis, mode=mode)

    data = np.asarray(vdf.data.data, dtype=np.float64).copy()
    data[(summed < n_thresh) & np.isfinite(data)] = 0.0

    out = vdf.copy()
    out["data"] = vdf.data.copy(data=data)

    return out
