#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def plot_surf(axis, x, y, z, c, cmap, norm, cax_pos: str = "bottom"):
    r"""Plots surface.

    Parameters
    ----------
    axis : matplotlib.axes._subplots.Axes3DSubplot
        Axis to plot.
    x : ndarray
        X axis meshgrid.
    y : ndarray
        Y axis meshgrid.
    z : ndarray
        Z axis meshgrid.
    c : ndarray
        C axis meshgrid.
    cmap : matplotlib.colors.ListedColormap
        Colormap
    norm : matplotlib.colors.Normalize
        Normalization.
    cax_pos : {"bottom", "top", "left", "right"}, Optional
        Position of the colorbar with respect to the axis. Default is "bottom".

    Returns
    -------
    axis : matplotlib.axes._subplots.Axes3DSubplot
        Axis with surface.
    caxis : matplotlib.axes._axes.Axes
        Colorbar axis.

    """

    axis.plot_surface(x, y, z, facecolors=cmap(norm(c)))

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(np.array([]))

    f = plt.gcf()
    pos = axis.get_position()

    if cax_pos == "bottom":
        caxis = f.add_axes([pos.x0, pos.y0 - .01, pos.width, .01])
        f.colorbar(mappable=mappable, cax=caxis, ax=axis,
                   orientation="horizontal")
    elif cax_pos == "top":
        caxis = f.add_axes([pos.x0, pos.y0 + pos.height + .01, pos.width, .01])
        f.colorbar(mappable=mappable, cax=caxis, ax=axis,
                   orientation="horizontal")
        caxis.xaxis.set_ticks_position("top")
        caxis.xaxis.set_label_position("top")

    elif cax_pos == "left":
        caxis = f.add_axes([pos.x0 - .01, pos.y0, .01, pos.height])
        f.colorbar(mappable=mappable, cax=caxis, ax=axis)
        caxis.yaxis.set_ticks_position("left")
        caxis.yaxis.set_label_position("left")

    elif cax_pos == "right":
        caxis = f.add_axes([pos.x0 + pos.width + .01, pos.y0, .01, pos.height])
        f.colorbar(mappable=mappable, cax=caxis, ax=axis)
    else:
        raise ValueError("Invalid caxis position")

    return axis, caxis
