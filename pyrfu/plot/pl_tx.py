#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def pl_tx(axis, inp_list, comp, **kwargs):
    r"""Line plot of 4 spacecraft time series.

    Parameters
    ----------
    axis : matplotlib.axes._subplots.AxesSubplot
        Axis
    inp_list : list of xarray.DataArray
        Time series to plot
    comp: int
        Index of the column to plot.

    Other Parameters
    ----------------
    kwargs : dict
        Hash table of plot options.

    """

    if axis is None:
        _, axis = plt.subplots(1)

    colors = ["blue", "green", "red", "k"]

    for inp, color in zip(inp_list, colors):
        if len(inp.shape) == 3:
            data = np.reshape(
                inp.data,
                (inp.shape[0], inp.shape[1] * inp.shape[2]),
            )
        elif len(inp.shape) == 1:
            data = inp.data[:, np.newaxis]
        else:
            data = inp.data

        time = inp.time
        axis.plot(time, data[:, comp], color=color, **kwargs)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
    axis.grid(True, which="major", linestyle="-", linewidth="0.5", c="0.5")

    return axis
