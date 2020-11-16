#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_spectr.py

@author : Louis RICHARD
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import colors
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use("seaborn-whitegrid")
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
sns.set_context("paper")
# plt.rc('lines', linewidth=1)


def plot_spectr(ax=None, inp=None, yscale="", cscale="", clim=None, cmap="", cbar=True, **kwargs):
    """Plot a spectrogram using pcolormesh.

    Parameters
    ----------
    ax : axes
        Target axis to plot. If None create a new figure.

    inp : DataArray
        Input 2D data to plot.

    yscale : str
        Y-axis flag. Default is "" (linear).

    cscale : str
        C-axis flag. Default is "" (linear).

    clim : list
        C-axis bounds. Default is None (autolim).

    cmap : str
        Colormap. Default is jet.

    cbar : bool
        Flag for colorbar. Set to False to hide.

    Returns
    -------
    fig : figure
        to fill.

    axs : axes
        to fill.

    caxs : caxes
        Only if cbar is True.

    """

    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = plt.gcf()

    if cscale == "log":
        if clim is not None and isinstance(clim, list):
            norm = colors.LogNorm(vmin=clim[0], vmax=clim[1])
            vmin = clim[0]
            vmax = clim[1]
        else:
            norm = colors.LogNorm()
            vmin = None
            vmax = None
    else:
        if clim is not None and isinstance(clim, list):
            norm = None
            vmin = clim[0]
            vmax = clim[1]
        else:
            norm = None
            vmin = None
            vmax = None

    if not cmap:
        cmap = "jet"

    t, y = [inp.coords[inp.dims[0]], inp.coords[inp.dims[1]]]

    im = ax.pcolormesh(t, y, inp.data.T, norm=norm, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    if yscale == "log":
        ax.set_yscale("log")

    if cbar:
        if "pad" in kwargs:
            pad = kwargs["pad"]
        else:
            pad = 0.01

        pos = ax.get_position()
        cax = fig.add_axes([pos.x0+pos.width+pad, pos.y0, 0.01, pos.height])
        fig.colorbar(mappable=im, cax=cax, ax=ax)

        return ax, cax
    else:
        return ax
