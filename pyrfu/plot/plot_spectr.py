#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.colors as mcolors
import matplotlib.dates as mdates

# 3rd party imports
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.11"
__status__ = "Prototype"

plt.style.use("seaborn-ticks")


def plot_spectr(
    axis,
    inp,
    yscale: str = "linear",
    cscale: str = "linear",
    clim: list = None,
    cmap: str = "",
    colorbar: str = "right",
    **kwargs,
):
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
    colorbar : str, Optional
        Location of the colorbar with respect to the axis.
        Set to "none" to hide.

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
            options = {
                "norm": mcolors.LogNorm(vmin=clim[0], vmax=clim[1]),
                "cmap": cmap,
            }
        else:
            options = {"norm": mcolors.LogNorm(), "cmap": cmap}
    else:
        if clim is not None and isinstance(clim, list):
            options = {"cmap": cmap, "vmin": clim[0], "vmax": clim[1]}
        else:
            options = {"cmap": cmap, "vmin": None, "vmax": None}

    x_data, y_data = [inp.coords[inp.dims[0]], inp.coords[inp.dims[1]]]

    image = axis.pcolormesh(
        x_data.data,
        y_data.data,
        inp.data.T,
        rasterized=True,
        shading="auto",
        **options,
    )

    if x_data.dtype == "<M8[ns]":
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    if yscale == "log":
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))

    axis.set_axisbelow(False)
    axis.set_ylim(inp[inp.dims[1]].data[[0, -1]])

    if colorbar.lower() == "right":
        if kwargs.get("pad"):
            pad = kwargs["pad"]
        else:
            pad = 0.01

        pos = axis.get_position()
        cax = fig.add_axes(
            [pos.x0 + pos.width + pad, pos.y0, 0.01, pos.height],
        )
        plt.colorbar(mappable=image, cax=cax, ax=axis, orientation="vertical")

        cax.yaxis.set_ticks_position(colorbar.lower())
        cax.yaxis.set_label_position(colorbar.lower())

        cax.set_axisbelow(False)

        if cscale == "log":
            cax.yaxis.set_major_locator(
                mticker.LogLocator(base=10.0, numticks=4),
            )
        else:
            cax.yaxis.set_major_locator(mticker.MaxNLocator(4))

        out = (axis, cax)
    elif colorbar.lower() == "top":
        if kwargs.get("pad"):
            pad = kwargs["pad"]
        else:
            pad = 0.01

        pos = axis.get_position()
        cax = fig.add_axes(
            [pos.x0, pos.y0 + pos.height + pad, pos.width, 0.01],
        )
        plt.colorbar(
            mappable=image,
            cax=cax,
            ax=axis,
            orientation="horizontal",
        )

        cax.xaxis.set_ticks_position(colorbar.lower())
        cax.xaxis.set_label_position(colorbar.lower())

        cax.set_axisbelow(False)

        if cscale == "log":
            cax.xaxis.set_major_locator(
                mticker.LogLocator(base=10.0, numticks=4),
            )
        else:
            cax.xaxis.set_major_locator(mticker.MaxNLocator(4))

        out = (axis, cax)
    elif colorbar.lower() == "none":
        out = axis
    else:
        raise NotImplementedError("colorbar must be: 'right', 'top', or 'none'")

    return out
