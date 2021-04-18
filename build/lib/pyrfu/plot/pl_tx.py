#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("seaborn-ticks")
colors_ = ["tab:blue", "tab:green", "tab:red", "k"]
plt.rc('axes', prop_cycle=cycler.cycler(color=colors_))


def pl_tx(axis, inp_list, comp, **kwargs):
    r"""Line plot of 4 spacecraft time series.

    Parameters
    ----------
    axis : to fill
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

    for inp in inp_list:
        if len(inp.shape) == 3:
            data = np.reshape(inp.data,
                              (inp.shape[0], inp.shape[1] * inp.shape[2]))
        elif len(inp.shape) == 1:
            data = inp.data[:, np.newaxis]
        else:
            data = inp.data

        time = inp.time
        axis.plot(time, data[:, comp], **kwargs)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
    axis.grid(True, which="major", linestyle="-", linewidth="0.5", c="0.5")

    return axis
