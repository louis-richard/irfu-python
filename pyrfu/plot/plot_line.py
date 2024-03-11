#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def plot_line(axis, inp, **kwargs):
    r"""Line plot of time series.

    Parameters
    ----------
    axis : matplotlib.axes._axes.Axes
        Single axis where to plot inp. If None creates a new figure with a single axis.
    inp : xarray.DataArray
        Time series to plot

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments control the line properties. See matplotlib.lines.Line2D
        for reference.

    Returns
    -------
    axs : matplotlib.axes._axes.Axes
        Axis with matplotlib.lines.Line2D.

    """

    if axis is None:
        _, axis = plt.subplots(1)
    else:
        if not isinstance(axis, mpl.axes.Axes):
            raise TypeError("axis must be a matplotlib.axes._axes.Axes")

    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be an xarray.DataArray object!")

    if inp.data.ndim < 3:
        data = inp.data
    elif inp.data.ndim == 3:
        data = np.reshape(
            inp.data,
            (inp.shape[0], inp.shape[1] * inp.shape[2]),
        )
    else:
        raise NotImplementedError(
            f"plot_line cannot handle {inp.data.ndim} dimensional data"
        )

    time = inp.time
    axis.plot(time, data, **kwargs)

    if time.dtype == "<M8[ns]":
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    axis.grid(True, which="major", linestyle="-", linewidth="0.2", c="0.5")
    axis.yaxis.set_major_locator(mticker.MaxNLocator(4))

    return axis
