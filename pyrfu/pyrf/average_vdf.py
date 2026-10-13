#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 3rd party imports
import numpy as np
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.pyrf.ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _seconds(values):
    r"""Durations in seconds as timedelta64[ns]."""
    return (np.asarray(values, dtype=np.float64) * 1e9).astype("timedelta64[ns]")


def average_vdf(vdf, n_pts, method: str = "mean"):
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
        Method for averaging. Use 'sum' for counts. Default is 'mean'.

    Returns
    -------
    vdf_avg : xarray.Dataset
        Time series of the time averaged velocity distribution function.

    """
    # Check input type
    if not isinstance(vdf, Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    if n_pts is not None and not isinstance(n_pts, int):
        raise TypeError("n_pts must be an integer or None")

    if n_pts is not None and n_pts % 2 == 0:
        raise ValueError("The number of distributions to be averaged must be an odd")

    if method not in ["mean", "sum"]:
        raise NotImplementedError("method not implemented feel free to do it!!")

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

    vdf_avg = np.stack(
        [func(vdf.data.data[i_l:i_r, ...], axis=0) for i_l, i_r in bounds]
    )
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
        vdf_avg,
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

    for k in ["delta_energy_minus", "delta_energy_plus"]:
        vdf_avg.attrs[k] = np.stack(
            [np.nanmean(vdf.attrs[k][i_l:i_r], axis=0) for i_l, i_r in bounds]
        )

    # Half widths of the accumulations: from the start of the first sample of
    # each window to the end of the last
    if "delta_time_minus" in vdf.attrs and "delta_time_plus" in vdf.attrs:
        t_start = np.array([times[i_l] for i_l, _ in bounds]) - _seconds(
            [vdf.attrs["delta_time_minus"][i_l] for i_l, _ in bounds]
        )
        t_stop = np.array([times[i_r - 1] for _, i_r in bounds]) + _seconds(
            [vdf.attrs["delta_time_plus"][i_r - 1] for _, i_r in bounds]
        )
        vdf_avg.attrs["delta_time_minus"] = (time_avg - t_start) / np.timedelta64(
            1, "s"
        )
        vdf_avg.attrs["delta_time_plus"] = (t_stop - time_avg) / np.timedelta64(1, "s")

    return vdf_avg
