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

"""plot_spectr.py
@author: Louis Richard
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

plt.style.use("seaborn-ticks")


def plot_spectr(axis, inp, yscale: str = "", cscale: str = "",
                clim: list = None, cmap: str = "", cbar: bool = True,
                **kwargs):
    r"""Plot a spectrogram using pcolormesh.

    Parameters
    ----------
    axis : axes
        Target axis to plot. If None create a new figure.

    inp : xarray.DataArray
        Input 2D data to plot.

    yscale : str
        Y-axis flag. Default is "" (linear).

    cscale : str
        C-axis flag. Default is "" (linear).

    clim : list
        C-axis bounds. Default is None (autolim).

    cmap : str
        Colormap. Default is jet.

    cbar : bool
        Flag for colorbar. Set to False to hide.

    Other Parameters
    ----------------
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    fig : figure
        to fill.

    axs : axes
        to fill.

    caxs : caxes
        Only if cbar is True.

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

    image = axis.pcolormesh(x_data, y_data, inp.data.T, rasterized=True,
                            **options)

    if x_data.dtype == '<M8[ns]':
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    if yscale == "log":
        axis.set_yscale("log")

    if cbar:
        if kwargs.get("pad"):
            pad = kwargs["pad"]
        else:
            pad = 0.01

        pos = axis.get_position()
        cax = fig.add_axes([pos.x0+pos.width+pad, pos.y0, 0.01, pos.height])
        fig.colorbar(mappable=image, cax=cax, ax=axis)

        out = (axis, cax)
    else:
        out = axis

    return out
