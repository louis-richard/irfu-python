#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import warnings

# 3rd party imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Local imports
from ..pyrf import datetime642iso8601, time_clip
from .plot_spectr import plot_spectr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _time_avg(vdf, tint):
    if not tint:
        tint = list(datetime642iso8601(vdf.time.data[[0, -1]]))
        warnings.warn("Averages the entire time series", UserWarning)

    if len(tint) == 1:
        idx = bisect.bisect_left(vdf.time.data, np.datetime64(tint[0]))
        vdf_data = vdf.data.data[idx, ...]
        vdf_ener = vdf.energy.data[idx, ...]
        vdf_azim = vdf.phi.data[idx, ...]
        vdf_thet = vdf.theta.data

    elif len(tint) == 2:
        vdf = time_clip(vdf, tint)
        vdf_data = np.nanmean(vdf.data.data, axis=0)
        vdf_ener = np.nanmean(vdf.energy.data, axis=0)
        vdf_azim = np.nanmean(vdf.phi.data, axis=0)
        vdf_thet = vdf.theta.data
    else:
        raise TypeError("Invalid time interval format")

    vdf_new = xr.DataArray(vdf_data, coords=[vdf_ener, vdf_azim, vdf_thet],
                           dims=["energy", "phi", "theta"])

    return vdf_new


def _energy_avg(vdf, en_range):
    if not en_range:
        en_range = vdf.energy.data[[0, -1]]
        warnings.warn("Averages the entire energy range", UserWarning)
    else:
        en_range[0] = np.max(vdf.energy.data[0], en_range[0])
        en_range[1] = np.max(vdf.energy.data[-1], en_range[-1])

    idx = np.where(np.logical_and(vdf.energy.data > en_range[0],
                                  vdf.energy.data < en_range[1]))[0]
    assert idx, "Energy range is not covered by the instrument"

    out_data = np.nanmean(vdf.data[idx, ...], axis=0)

    out = xr.DataArray(out_data, coords=[vdf.phi.data, vdf.theta.data],
                       dims=["phi", "theta"])
    return out


def _check_units(vdf):
    if vdf.attrs["UNITS"] == "s^3/m^6":
        y_label = "PSD [s$^3$ m$^{-6}$]"
    elif vdf.attrs["UNITS"] == "1/(cm^2 s sr keV)":
        y_label = "Intensity [(cm$^2$ s sr keV)$^{-1}$]"
    else:
        raise ValueError("Invalid units")

    return y_label


def plot_ang_ang(vdf, tint: list = None, en_range: list = None):
    r"""Creates colormap of the phase space density or the differential
    particle flux, as a function of the azimuthal and elevation angles.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution.
    tint : list of str, Optional
        Time interval. If the time interval contains only one element,
        uses the distribution at the closest time. If the time interval
        contains two elements, time average the distribution. If None,
        uses the entire timeline for averaging. Default is None.
    en_range : list of float, Optional
        Energy range. If None uses the entire energy range.

    Returns
    -------
    f : matplotlib.figure.Figure
        Figure
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis
    cax : matplotlib.axes._axes.Axes
        Colorbar axis

    """

    # Average over the selected time interval
    vdf_c = _time_avg(vdf, tint)

    # Average over the selected energy range
    vdf_avg = _energy_avg(vdf_c, en_range)

    f, ax = plt.subplots(1, figsize=(9, 9))
    f.subplots_adjust(left=.1, right=.85, bottom=.1, top=.9)
    ax, cax = plot_spectr(ax, vdf_avg, cscale="log")
    ax.set_xlabel("$\\phi$ [deg.]")
    ax.set_ylabel("$\\theta$ [deg.]")
    cax.set_ylabel(_check_units(vdf))
    ax.set_title(f"{en_range[0]:3.0f} keV < $E$ < {en_range[1]:3.0f} keV")

    return f, ax, cax
