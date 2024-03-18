#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import List

# 3rd party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def pl_tx(
    axis: Axes, inp_list: List[DataArray], comp: int = 0, colors: str = "mms", **kwargs
) -> Axes:
    r"""Line plot of 4 spacecraft time series.

    Parameters
    ----------
    axis : matplotlib.axes._axes.Axes
        Axis
    inp_list : list of xarray.DataArray
        Time series to plot
    comp: int, Optional
        Index of the column to plot. Default is 0.
    colors: {'cluster', 'mms'}, Optional
        Color cycle to use. Default uses MMS

    Other Parameters
    ----------------
    kwargs : dict
        Hash table of plot options.

    Returns
    -------
    axis : matplotlib.axes._axes.Axes
        Axis with matplotlib.lines.Line2D.

    Raises
    ------
    NotImplementedError: if invalid color style of inp_list.ndim > 3

    """

    if axis is None:
        _, axis = plt.subplots(1)

    if colors.lower() not in ["cluster", "mms"]:
        raise NotImplementedError("Unknown color cycle")

    for i, inp in enumerate(inp_list):
        if inp.ndim == 1:
            data = inp.data[:, np.newaxis]
        elif inp.ndim == 2:
            data = inp.data
        elif inp.ndim == 3:
            data = np.reshape(
                inp.data,
                (inp.shape[0], inp.shape[1] * inp.shape[2]),
            )
        else:
            raise NotImplementedError("inp.ndim > 3 not implemented")

        time = inp.time
        axis.plot(time, data[:, comp], color=f"{colors}:{colors}{i + 1:d}", **kwargs)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
    axis.grid(True, which="major", linestyle="-", linewidth="0.5", c="0.5")

    return axis
