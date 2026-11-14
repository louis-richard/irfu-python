#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from typing import Literal

import matplotlib as mpl
import numpy as np

# Local imports
from .colorbar import colorbar as _add_colorbar


def plot_projection(
    axis,
    v_x,
    v_y,
    f_mat,
    vlim: float = 1e3,
    clim: list | None = None,
    cmap: str | None = None,
    colorbar: Literal["right", "top", "none"] = "right",
):
    r"""Plot the projection of the distribution.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to plot.
    v_x : ndarray
        X axis velocity grid, in km/s (plotted in 10^3 km/s).
    v_y : ndarray
        Y axis velocity grid, in km/s (plotted in 10^3 km/s).
    f_mat : ndarray
        Projected distribution, plotted as its base-10 logarithm.
    vlim : float, Optional
        Maximum velocity of the axes, in km/s. Default is 1000.
    clim : list, Optional
        Limits of the colour scale, [min, max] of log10(f_mat). Default is
        the range of the data.
    cmap : str, Optional
        Colormap. Default is matplotlib's default (``image.cmap``, set by the
        pyrfu style).
    colorbar : {"right", "top", "none"}, Optional
        Location of the colorbar. Default is "right".

    Returns
    -------
    axis : matplotlib.axes.Axes
        Axis.
    caxis : matplotlib.axes.Axes or None
        Colorbar axis, None if colorbar is "none".

    Raises
    ------
    TypeError
        If `cmap` is not a string.
    NotImplementedError
        If `colorbar` is not "right", "top" or "none".

    """

    if clim is None:
        clim = [None, None]

    if not cmap or isinstance(cmap, str):
        # get_cmap(None) is the image.cmap colormap; the stub omits None
        colormap = mpl.colormaps.get_cmap(cmap)  # type: ignore[arg-type]
    else:
        raise TypeError(
            "cmap must be a string. "
            "To add a custom colormap use mpl.colormaps.register(custom)."
        )

    image = axis.pcolormesh(
        v_x / 1e3,
        v_y / 1e3,
        np.log10(f_mat.T),
        cmap=colormap,
        vmin=clim[0],
        vmax=clim[1],
    )
    axis.set_xlim([-vlim / 1e3, vlim / 1e3])
    axis.set_ylim([-vlim / 1e3, vlim / 1e3])
    axis.set_aspect("equal")

    if colorbar in ["top", "right"]:
        caxis = _add_colorbar(image, axis, width=0.01, where=colorbar)
    elif colorbar == "none":
        caxis = None
    else:
        raise NotImplementedError("invalid position")

    return axis, caxis
