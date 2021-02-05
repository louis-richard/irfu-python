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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cycler import cycler
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
# date_form = mdates.ConciseDateFormatter(locator)
date_form = mdates.DateFormatter("%H:%M:%S")
plt.style.use("seaborn-whitegrid")
sns.set_context("paper")

color = ["k", "tab:blue", "tab:red", "tab:green"]
# color = np.array([[0, 0, 0], [213, 94, 0], [0, 158, 115], [86, 180, 233]]) / 255

default_cycler = cycler(color=color)
plt.rc('axes', prop_cycle=default_cycler)


def pl_tx(axis, inp_list, comp, **kwargs):
    """Line plot of 4 spacecraft time series.

    Parameters
    ----------
    axis : to fill
        Axis

    inp_list : list of xarray.DataArray
        Time series to plot

    comp: int
        Index of the column to plot.

    kwargs : dict
        Hash table of plot options.

    """

    if axis is None:
        _, axis = plt.subplots(1)

    for inp in inp_list:
        if len(inp.shape) == 3:
            data = np.reshape(inp.data, (inp.shape[0], inp.shape[1] * inp.shape[2]))
        elif len(inp.shape) == 1:
            data = inp.data[:, np.newaxis]
        else:
            data = inp.data

        time = inp.time
        axis.plot(time, data[:, comp], **kwargs)

    axis.xaxis.set_major_formatter(date_form)
    axis.grid(True, which="major", linestyle="-", linewidth="0.5", c="0.5")

    return axis
