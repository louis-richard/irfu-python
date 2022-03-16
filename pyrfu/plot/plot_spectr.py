#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.11"
__status__ = "Prototype"

plt.style.use("seaborn-ticks")


def plot_spectr(axis, inp, yscale: str = "linear", cscale: str = "linear",
                clim: list = None, cmap: str = "", colorbar: bool = True,
                **kwargs):
    r"""Plot a spectrogram using pcolormesh.

    Parameters
    ----------
    axis : matplotlib.pyplot.subplotsaxes
        Target axis to plot. If None create a new figure.
    inp : xarray.DataArray
        Input 2D data to plot.
    yscale : {"linear", "log"}, Optional
        Y-axis scaling. Default is "" (linear).
    cscale : {"linear", "log"}, Optional
        C-axis scaling. Default is "" (linear).
    clim : list, Optional
        C-axis bounds. Default is None (autolim).
    cmap : str, Optional
        Colormap. Default is "jet".
    colorbar : bool, Optional
        Flag for colorbar. Set to False to hide.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments.

    Returns
    -------
    fig : figure
        Figure with axis.
    axs : matplotlib.pyplot.subplotsaxes
        Axis with spectrum.
    caxs : matplotlib.pyplot.subplotsaxes
        Only if colorbar is True.

    """

    if axis is None:
        fig, axis = plt.subplots(1)
    else:
        fig = plt.gcf()

    if not cmap:
        cmap = "jet"

    if cscale == "log":
        if clim is not None and isinstance(clim, list):
            options = dict(norm=mcolors.LogNorm(vmin=clim[0], vmax=clim[1]),
                           cmap=cmap)
        else:
            options = dict(norm=mcolors.LogNorm(), cmap=cmap)
    else:
        if clim is not None and isinstance(clim, list):
            options = dict(cmap=cmap, vmin=clim[0], vmax=clim[1])
        else:
            options = dict(cmap=cmap, vmin=None, vmax=None)

    x_data, y_data = [inp.coords[inp.dims[0]], inp.coords[inp.dims[1]]]

    image = axis.pcolormesh(x_data.data, y_data.data, inp.data.T,
                            rasterized=True, shading="auto", **options)

    if x_data.dtype == '<M8[ns]':
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    if yscale == "log":
        axis.set_yscale("log")

    axis.set_axisbelow(False)

    if colorbar:
        if kwargs.get("pad"):
            pad = kwargs["pad"]
        else:
            pad = 0.01

        pos = axis.get_position()
        cax = fig.add_axes([pos.x0+pos.width+pad, pos.y0, 0.01, pos.height])
        fig.colorbar(mappable=image, cax=cax, ax=axis)
        cax.set_axisbelow(False)

        out = (axis, cax)
    else:
        out = axis

    return out
