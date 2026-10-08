#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.histogram2d import histogram2d
from ..pyrf.resample import resample
from .plot_spectr import plot_spectr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def pl_scatter_matrix(
    inp1,
    inp2: xr.DataArray = None,
    pdf: bool = False,
    cmap: str = "jet",
):
    r"""Produces a scatter plot of each components of field inp1 with respect
    to every component of field inp2. If pdf is set to True, the scatter
    plot becomes a 2d histogram.

    Parameters
    ----------
    inp1 : xarray.DataArray
        First time series (x-axis).
    inp2 : xarray.DataArray
        Second time series (y-axis).
    pdf : bool, Optional
        Flag to plot the 2d histogram. If False the figure is a scatter plot.
        If True the figure is a 2d histogram.
    cmap : str, Optional
        Colormap. Default : "jet"

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure with time series plots.
    axs : matplotlib.axes._subplots.AxesSubplot
        Axes.
    caxs : list of list of matplotlib.axes.Axes
        Colorbar axes, in the same layout as axs. Only if pdf is True.

    """

    if inp2 is None:
        inp2 = inp1
        warnings.warn("inp2 is empty assuming that inp2=inp1", UserWarning)

    if not isinstance(inp1, xr.DataArray) or not isinstance(inp2, xr.DataArray):
        raise TypeError("inp1 and inp2 must be xarray.DataArray")

    # Pair the samples at the same times (as histogram2d does)
    if not np.array_equal(inp1.time.data, inp2.time.data):
        inp2 = resample(inp2, inp1)

    if not pdf:
        fig, axs = plt.subplots(
            3,
            3,
            sharex="all",
            sharey="all",
            figsize=(16, 9),
        )
        fig.subplots_adjust(
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            hspace=0.05,
            wspace=0.05,
        )

        for i in range(3):
            for j in range(3):
                axs[j, i].scatter(
                    inp1[:, i].data,
                    inp2[:, j].data,
                    marker="+",
                )

        out = (fig, axs)
    else:
        fig, axs = plt.subplots(
            3,
            3,
            sharex="all",
            sharey="all",
            figsize=(16, 9),
        )
        fig.subplots_adjust(
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            hspace=0.05,
            wspace=0.3,
        )

        caxs = [[None] * 3 for _ in range(3)]

        for i in range(3):
            for j in range(3):
                hist_ = histogram2d(inp1[:, i], inp2[:, j])
                axs[j, i], caxs[j][i] = plot_spectr(
                    axs[j, i],
                    hist_,
                    cmap=cmap,
                    cscale="log",
                )
                axs[j, i].grid()

        out = (fig, axs, caxs)

    return out
