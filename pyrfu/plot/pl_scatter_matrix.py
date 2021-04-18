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

"""pl_scatter_matrix.py
@author: Louis Richard
"""


import warnings
import xarray as xr
import matplotlib.pyplot as plt

from ..pyrf import histogram2d
from . import plot_spectr


def pl_scatter_matrix(inp1, inp2, pdf: bool = False, cmap: str = "jet"):
    """Produces a scatter plot of each components of field inp1 with respect
    to every component of field inp2. If pdf is set to True, the scatter
    plot becomes a 2d histogram.

    Parameters
    ----------
    inp1 : xarray.DataArray
        First time series (x-axis).

    inp2 : xarray.DataArray
        Second time series (y-axis).

    pdf : bool
        Flag to plot the 2d histogram. If False the figure is a scatter plot.
        If True the figure is a 2d histogram.

    cmap : str
        Colormap. Default : "jet"

    Returns
    -------
    fig : matplotlib.pyplot.figure
        to fill.

    axs : matplotlib.pyplot.axes
        to fill.

    caxs : matplotlib.pyplot.colorbar
        Only if pdf is True

    """

    if inp1 is None:
        raise ValueError("pl_scatter_matrix requires at least one argument")

    if inp2 is None:
        inp2 = inp1
        warnings.warn("inp2 is empty assuming that inp2=inp1", UserWarning)

    assert isinstance(inp1, xr.DataArray) and isinstance(inp2, xr.DataArray)

    if not pdf:
        fig, axs = plt.subplots(3, 3, sharex="all", sharey="all",
                                figsize=(16, 9))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9,
                            hspace=0.05, wspace=0.05)

        for i in range(3):
            for j in range(3):
                axs[j, i].scatter(inp1[:, i].data, inp2[:, j].data, marker="+")

        out = (fig, axs)
    else:
        fig, axs = plt.subplots(3, 3, sharex="all", sharey="all",
                                figsize=(16, 9))
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9,
                            hspace=0.05, wspace=0.3)

        caxs = [[None]*3]*3

        for i in range(3):
            for j in range(3):
                hist_ = histogram2d(inp1[:, i], inp2[:, j])
                axs[j, i], caxs[j][i] = plot_spectr(axs[j, i], hist_,
                                                    cmap=cmap, cscale="log")
                axs[j, i].grid()

        out = (fig, axs, caxs)

    return out
