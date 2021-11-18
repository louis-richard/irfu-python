#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


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

    im = ax.pcolormesh(f2d.vx.data, f2d.vy.data, np.transpose(f2d.data),
                       norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
                       cmap="Spectral_r", rasterized=True, shading="auto")

    c_lines = ax.contour(f2d.vx.data, f2d.vy.data, np.transpose(f2d.data),
                         levels=np.logspace(clim_log[0], clim_log[1], n_levs),
                         norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
                         cmap="viridis")

    vx_lim = np.max(abs(f2d.vx.data))
    vy_lim = np.max(abs(f2d.vy.data))

    ax.set_xlim(np.array([-vx_lim, vx_lim]))
    ax.set_ylim(np.array([-vy_lim, vy_lim]))
    ax.set_aspect("equal")

    f = plt.gcf()
    pos = ax.get_position()
    cax = f.add_axes([pos.x0, pos.y0 + pos.height + .01, pos.width, .01])
    cbar = f.colorbar(mappable=im, cax=cax, ax=ax, orientation="horizontal")
    cbar.add_lines(c_lines)
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position("top")

    return ax, cax
