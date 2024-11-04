#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 3rd party imports
import numpy as np
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.pyrf.ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def average_vdf(vdf, n_pts, method: str = "mean"):
    r"""Time averages the velocity distribution functions over `n_pts` in time.

    Parameters
    ----------
    vdf : xarray.DataArray
        Time series of the velocity distribution function.
    n_pts : int
        Number of points (samples) of the averaging window.
    method : {'mean', 'sum'}, Optional
        Method for averaging. Use 'sum' for counts. Default is 'mean'.

    Returns
    -------
    vdf_avg : xarray.DataArray
        Time series of the time averaged velocity distribution function.

    """
    # Check input type
    if not isinstance(vdf, Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    if not isinstance(n_pts, int):
        raise TypeError("n_pts must be an integer")

    if n_pts % 2 == 0:
        raise ValueError("The number of distributions to be averaged must be an odd")

    n_vdf = len(vdf.time.data)
    times = vdf.time.data

    pad_value = np.floor(n_pts / 2)
    avg_inds = np.arange(pad_value, n_vdf - pad_value, n_pts, dtype=int)
    time_avg = times[avg_inds]

    energy_avg = np.zeros((len(avg_inds), vdf.data.shape[1]))
    phi_avg = np.zeros((len(avg_inds), vdf.data.shape[2]))
    vdf_avg = np.zeros((len(avg_inds), *vdf.data.shape[1:]))

    for i, avg_ind in enumerate(avg_inds):
        l_bound = int(avg_ind - pad_value)
        r_bound = int(avg_ind + pad_value)
        if method == "mean":
            vdf_avg[i, ...] = np.nanmean(vdf.data.data[l_bound:r_bound, ...], axis=0)
        elif method == "sum":
            vdf_avg[i, ...] = np.nansum(vdf.data.data[l_bound:r_bound, ...], axis=0)
        else:
            raise NotImplementedError("method not implemented feel free to do it!!")

        energy_avg[i, ...] = np.nanmean(vdf.energy.data[l_bound:r_bound, ...], axis=0)
        phi_avg[i, ...] = np.nanmean(vdf.phi.data[l_bound:r_bound, ...], axis=0)

    vdf_avg = ts_skymap(time_avg, vdf_avg, energy_avg, phi_avg, vdf.theta.data)
    vdf_avg.attrs = vdf.attrs

    vdf_avg.time.attrs = vdf.time.attrs
    for k in vdf:
        vdf_avg[k].attrs = vdf[k].attrs

    vdf_avg.attrs["energy0"] = vdf.attrs["energy0"]
    vdf_avg.attrs["energy1"] = vdf.attrs["energy1"]
    vdf_avg.attrs["esteptable"] = vdf.attrs["esteptable"][: len(avg_inds)]
    vdf_avg.attrs["delta_energy_minus"] = vdf.attrs["delta_energy_minus"][avg_inds]
    vdf_avg.attrs["delta_energy_plus"] = vdf.attrs["delta_energy_plus"][avg_inds]

    return vdf_avg
