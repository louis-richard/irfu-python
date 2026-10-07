#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import warnings

import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
# (from the modules, as pyrfu.plot is imported while pyrfu.pyrf is loading)
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.time_clip import time_clip
from .plot_spectr import plot_spectr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
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

    vdf_new = xr.DataArray(
        vdf_data,
        coords=[vdf_ener, vdf_azim, vdf_thet],
        dims=["energy", "phi", "theta"],
    )

    return vdf_new


def _energy_avg(vdf, en_range):
    energy = vdf.energy.data
    e_min, e_max = np.nanmin(energy), np.nanmax(energy)

    if en_range is None:
        en_range = np.array([e_min, e_max])
        warnings.warn("Averages the entire energy range", UserWarning)
    else:
        # Clamp to the energy range of the instrument (without changing the
        # caller's list)
        en_range = np.array([max(e_min, en_range[0]), min(e_max, en_range[1])])

    idx = np.where(np.logical_and(energy >= en_range[0], energy <= en_range[1]))[0]

    if idx.size == 0:
        raise ValueError("Energy range is not covered by the instrument")

    out_data = np.nanmean(vdf.data[idx, ...], axis=0)

    out = xr.DataArray(
        out_data,
        coords=[vdf.phi.data, vdf.theta.data],
        dims=["phi", "theta"],
    )
    return out, en_range


def _check_units(vdf):
    # The units are an attribute of the data variable of the skymap
    units = vdf.data.attrs.get("UNITS", "")
    psd_units = {f"s^3/{u}^6": f"s$^3$ {u}$^{{-6}}$" for u in ["m", "cm", "km"]}

    if units in psd_units:
        y_label = f"PSD [{psd_units[units]}]"
    elif units == "1/(cm^2 s sr keV)":
        y_label = "Intensity [(cm$^2$ s sr keV)$^{-1}$]"
    elif units == "keV/(cm^2 s sr keV)":
        y_label = "DEF [keV (cm$^2$ s sr keV)$^{-1}$]"
    else:
        y_label = units

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
        Energy range (inclusive), in the units of the energies of vdf, clamped
        to the energy range of the instrument. If None uses the entire energy
        range.

    Returns
    -------
    f : matplotlib.figure.Figure
        Figure
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis
    cax : matplotlib.axes._axes.Axes
        Colorbar axis

    Raises
    ------
    ValueError
        If no energy channel is within en_range.

    Notes
    -----
    Averaging FPI burst distributions over a time interval mixes the two
    alternating energy tables; rebin them to 64 energies first (see
    :func:`pyrfu.mms.vdf_to_e64`).

    """

    # Average over the selected time interval
    vdf_c = _time_avg(vdf, tint)

    # Average over the selected energy range
    vdf_avg, en_range = _energy_avg(vdf_c, en_range)
    en_units = vdf.energy.attrs.get("UNITS", "keV")

    f, ax = plt.subplots(1, figsize=(9, 9))
    f.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9)
    ax, cax = plot_spectr(ax, vdf_avg, cscale="log")
    ax.set_xlabel("$\\phi$ [deg.]")
    ax.set_ylabel("$\\theta$ [deg.]")
    cax.set_ylabel(_check_units(vdf))
    ax.set_title(
        f"{en_range[0]:.3g} {en_units} $\\leq E \\leq$ {en_range[1]:.3g} {en_units}"
    )

    return f, ax, cax
