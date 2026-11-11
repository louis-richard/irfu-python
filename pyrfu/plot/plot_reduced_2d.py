#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from matplotlib import colors

# Local imports
from .colorbar import colorbar as _add_colorbar


def plot_reduced_2d(ax, f2d, clim: list = None):
    r"""Plot the 2D recuded distribution `f2d` onto the axis `ax`.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis.
    f2d : xarray.DataArray
        2D reduced distribution.
    clim : list of floats, Optional
        Coloraxis limits.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis.
    cax : matplotlib.axes._axes.Axes
        Colorbar axis.

    """

    if clim is None:
        clim = [1e-13, 1e-8]

    clim_log = np.log10(clim)
    n_levs = int(np.round(clim_log[1]) - np.round(clim_log[0])) + 1

    im = ax.pcolormesh(
        f2d.vx.data,
        f2d.vy.data,
        np.transpose(f2d.data),
        norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
        cmap="Spectral_r",
        rasterized=True,
        shading="auto",
    )

    c_lines = ax.contour(
        f2d.vx.data,
        f2d.vy.data,
        np.transpose(f2d.data),
        levels=np.logspace(clim_log[0], clim_log[1], n_levs),
        norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
        cmap="viridis",
    )

    vx_lim = np.max(abs(f2d.vx.data))
    vy_lim = np.max(abs(f2d.vy.data))

    ax.set_xlim(np.array([-vx_lim, vx_lim]))
    ax.set_ylim(np.array([-vy_lim, vy_lim]))
    ax.set_aspect("equal")

    cax = _add_colorbar(im, ax, width=0.01, where="top")
    im.colorbar.add_lines(c_lines)

    return ax, cax
