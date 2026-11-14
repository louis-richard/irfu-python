#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from typing import Literal

from matplotlib.cm import ScalarMappable

# Local imports
from .colorbar import colorbar as _add_colorbar


def plot_surf(
    axis,
    x,
    y,
    z,
    c,
    cmap,
    norm,
    cax_pos: Literal["bottom", "top", "left", "right"] = "bottom",
):
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

    mappable = ScalarMappable(cmap=cmap, norm=norm)

    if cax_pos not in ["bottom", "top", "left", "right"]:
        raise ValueError("Invalid caxis position")

    # The gap of colorbar on the four sides
    caxis = _add_colorbar(mappable, axis, 0.01, 0.01, cax_pos)

    return axis, caxis
