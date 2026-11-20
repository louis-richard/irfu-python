#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from typing import Literal

# 3rd party imports
import numpy as np
from xarray import Dataset

# Local imports
from pyrfu._core.time import _seconds, _timedelta
from pyrfu.pyrf.ts_skymap import ts_skymap


def average_vdf(
    vdf, n_pts, method: Literal["mean", "sum"] = "mean", counts: Dataset | None = None
):
    r"""Time averages the velocity distribution functions over `n_pts` in time.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the velocity distribution function.
    n_pts : int or None
        Number of points (samples) of the averaging window. If None, the
        distributions are averaged over the entire time interval, and the
        output is a skymap with a single time step (at the middle of the
        interval) which can be used as input of the other routines.
    method : {'mean', 'sum'}, Optional
        Method for averaging. Use 'sum' for counts. Default is 'mean'. The
        NaN bins have no data and are left out; a bin without data in a
        window is NaN, for both methods (see :ref:`api-mms-vdf-nan`).
    counts : xarray.Dataset, Optional
        Counts of `vdf`, at its times and on its bins (e.g. from
        :func:`pyrfu.mms.vdf_counts`). The bins with fewer than 2 counts
        summed over the window (one count or none: below 1.5, the counts of
        FPI being whole numbers to 0.05 %) are set to 0, as irfu-matlab
        ``PDist.movmean(..., 'RemoveOneCounts', counts)``; NaN counts count as
        0. Default is None: no removal.

    Returns
    -------
    vdf_avg : xarray.Dataset
        Time series of the time averaged velocity distribution function. The
        energy widths (``delta_energy_minus`` and ``delta_energy_plus``, time
        x energy) are averaged over each window, and the accumulation half
        widths (``delta_time_minus`` and ``delta_time_plus``) span it, when
        `vdf` has them. The compression indicator (``compressionloss``, 0
        lossless, 1 lossy) of each window is its largest: an average is lossy
        if one of its distributions is. The FPI quality flags (``errorflags``)
        of each window are those of its distributions combined (bitwise or):
        an average has the flags of each of its distributions.

    Raises
    ------
    TypeError
        If `vdf` or `counts` is not a xarray.Dataset, or `n_pts` is not an
        integer or None.
    ValueError
        If `n_pts` is not odd or not between 1 and the number of
        distributions, or `counts` does not have the times and the shape of
        `vdf`.

    Notes
    -----
    The one-count removal keeps the bins with data only: a bin without data
    in the window stays NaN (see :ref:`api-mms-vdf-nan`). With lossy
    compressed distributions (FPI fast survey, ``compressionloss``), the
    counts of the low bins do not go below about 2 (see
    :func:`pyrfu.mms.vdf_counts`), so that few bins are removed.

    """

    # Check input type
    if not isinstance(vdf, Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    if n_pts is not None and (not isinstance(n_pts, int) or isinstance(n_pts, bool)):
        raise TypeError("n_pts must be an integer or None")

    # No more points than distributions
    if n_pts is not None and not 1 <= n_pts <= len(vdf.time):
        raise ValueError(
            f"n_pts must be between 1 and the number of distributions ({len(vdf.time)})"
        )

    if n_pts is not None and n_pts % 2 == 0:
        raise ValueError("The number of distributions to be averaged must be an odd")

    if method not in ["mean", "sum"]:
        raise NotImplementedError("method not implemented feel free to do it!!")

    if counts is not None:
        if not isinstance(counts, Dataset):
            raise TypeError("counts must be a xarray.Dataset")

        if counts.data.shape != vdf.data.shape or not np.array_equal(
            counts.time.data, vdf.time.data
        ):
            raise ValueError("counts must have the times and the shape of vdf")

    n_vdf = len(vdf.time.data)
    times = vdf.time.data

    if n_pts is None:
        # Single window spanning the whole interval, time tag at the middle.
        bounds = [(0, n_vdf)]
        avg_inds = np.array([n_vdf // 2])
        time_avg = np.array([times[0] + (times[-1] - times[0]) / 2])
    else:
        pad_value = n_pts // 2
        avg_inds = np.arange(pad_value, n_vdf - pad_value, n_pts, dtype=int)
        bounds = [(i - pad_value, i + pad_value + 1) for i in avg_inds]
        time_avg = times[avg_inds]

    func = np.nanmean if method == "mean" else np.nansum

    # Mean or sum over the samples with data of each bin, NaN without
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data_avg = np.stack(
            [func(vdf.data.data[i_l:i_r, ...], axis=0) for i_l, i_r in bounds]
        )

    no_data = np.stack(
        [~np.any(np.isfinite(vdf.data.data[i_l:i_r]), axis=0) for i_l, i_r in bounds]
    )
    data_avg[no_data] = np.nan

    # One-count removal: fewer than 2 counts in the window (irfu-matlab
    # PDist.movmean RemoveOneCounts), the bins with data only
    if counts is not None:
        n_counts = np.nan_to_num(np.asarray(counts.data.data, dtype=np.float64))
        summed = np.stack([np.sum(n_counts[i_l:i_r], axis=0) for i_l, i_r in bounds])
        data_avg[(summed < 1.5) & ~no_data] = 0.0
    energy_avg = np.stack(
        [np.nanmean(vdf.energy.data[i_l:i_r], axis=0) for i_l, i_r in bounds]
    )
    phi_avg = np.stack(
        [np.nanmean(vdf.phi.data[i_l:i_r], axis=0) for i_l, i_r in bounds]
    )

    # Pass energy tables explicitly as ts_skymap would otherwise read
    # energy[1, :], which fails for a single time step.
    vdf_avg = ts_skymap(
        time_avg,
        data_avg,
        energy_avg,
        phi_avg,
        vdf.theta.data,
        energy0=vdf.attrs["energy0"],
        energy1=vdf.attrs["energy1"],
        esteptable=vdf.attrs["esteptable"][avg_inds],
    )
    vdf_avg.attrs = {**vdf.attrs, **vdf_avg.attrs}

    vdf_avg.time.attrs = vdf.time.attrs
    for k in vdf:
        vdf_avg[k].attrs = vdf[k].attrs

    # Energy widths (time, energy) averaged over each window; left as they
    # are when missing (get_dist leaves them out) or per channel (energy,)
    for k in ["delta_energy_minus", "delta_energy_plus"]:
        if vdf.attrs.get(k) is None:
            continue

        widths = np.asarray(vdf.attrs[k])

        if widths.ndim == 2 and len(widths) == n_vdf:
            vdf_avg.attrs[k] = np.stack(
                [np.nanmean(widths[i_l:i_r], axis=0) for i_l, i_r in bounds]
            )

    # Compression indicator (per time): lossy if one of the distributions is
    loss = vdf.attrs.get("compressionloss")

    if isinstance(loss, np.ndarray) and loss.ndim == 1 and len(loss) == n_vdf:
        vdf_avg.attrs["compressionloss"] = np.array(
            [np.max(loss[i_l:i_r]) for i_l, i_r in bounds]
        )

    # Quality flags (per time): those of each distribution of the window
    flags = vdf.attrs.get("errorflags")

    if isinstance(flags, np.ndarray) and flags.ndim == 1 and len(flags) == n_vdf:
        vdf_avg.attrs["errorflags"] = np.array(
            [np.bitwise_or.reduce(flags[i_l:i_r]) for i_l, i_r in bounds]
        )

    # Half widths of the accumulations: from the start of the first sample of
    # each window to the end of the last
    if "delta_time_minus" in vdf.attrs and "delta_time_plus" in vdf.attrs:
        t_start = np.array([times[i_l] for i_l, _ in bounds]) - _timedelta(
            [vdf.attrs["delta_time_minus"][i_l] for i_l, _ in bounds]
        )
        t_stop = np.array([times[i_r - 1] for _, i_r in bounds]) + _timedelta(
            [vdf.attrs["delta_time_plus"][i_r - 1] for _, i_r in bounds]
        )
        vdf_avg.attrs["delta_time_minus"] = _seconds(time_avg, t_start)
        vdf_avg.attrs["delta_time_plus"] = _seconds(t_stop, time_avg)

    return vdf_avg
