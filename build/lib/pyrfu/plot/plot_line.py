#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

plt.style.use("seaborn-ticks")
color = ["tab:blue", "tab:green", "tab:red", "k"]
plt.rc('axes', prop_cycle=mpl.cycler(color=color))


def plot_line(axis, inp, **kwargs):
    r"""Line plot of time series.

    Parameters
    ----------
    axis : matplotlib.pyplot.subplotsaxes
        Axis
    inp : xarray.DataArray
        Time series to plot

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments control the Line2D properties.

    Returns
    -------
    axs :
        Axes.

    """

    if axis is None:
        _, axis = plt.subplots(1)

    if len(inp.shape) == 3:
        data = np.reshape(inp.data,
                          (inp.shape[0], inp.shape[1] * inp.shape[2]))
    else:
        data = inp.data

    time = inp.time
    axis.plot(time, data, **kwargs)

    if time.dtype == '<M8[ns]':
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    axis.grid(True, which="major", linestyle="-", linewidth="0.5", c="0.5")

    return axis
