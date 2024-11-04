#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def plot_projection(
    axis,
    v_x,
    v_y,
    f_mat,
    vlim: float = 1e3,
    clim: list = None,
    cmap: str = None,
    colorbar: str = "right",
):
    r"""Plot the projection of the distribution.

    Parameters
    ----------
    axis : matplotlib.pyplot.subplotsaxes
        Axis to plot.
    v_x : ndarray
        X axis velocity grid.
    v_y : ndarray
        Y axis velocity grid.
    f_mat : ndarray
        Projected distribution.
    vlim : float, Optional
        Maximum velocity to limit axis. Default is vlim = 1000 km/s.
    clim : list, Optional
        Caxes limit. Default is clim = [-18, -13] (assume to be in SI units)
    colorbar : str, Optional
        Location of the colorbar. Default is cbar_pos = "top".
    cmap : str, Optional
        Colormap. Default is cmap = "jet".

    Returns
    -------
    axis : axis
        Axis
    caxis : axis
        Colorbar axis.

    """

    if clim is None:
        clim = [None, None]

    if not cmap or isinstance(cmap, str):
        cmap = mpl.colormaps.get_cmap(cmap)
    else:
        raise TypeError(
            "cmap must be a string. "
            "To add a custom colormap use mpl.colormaps.register(custom)."
        )

    image = axis.pcolormesh(
        v_x / 1e3,
        v_y / 1e3,
        np.log10(f_mat.T),
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    axis.set_xlim([-vlim / 1e3, vlim / 1e3])
    axis.set_ylim([-vlim / 1e3, vlim / 1e3])
    axis.set_aspect("equal")

    f = plt.gcf()
    pos = axis.get_position()

    if colorbar == "top":
        caxis = f.add_axes(
            [pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.01],
        )
        f.colorbar(
            mappable=image,
            cax=caxis,
            ax=axis,
            orientation="horizontal",
        )
        caxis.xaxis.set_ticks_position("top")
        caxis.xaxis.set_label_position("top")
    elif colorbar == "right":
        caxis = f.add_axes(
            [pos.x0 + pos.width + 0.01, pos.y0, 0.01, pos.height],
        )
        f.colorbar(mappable=image, cax=caxis, ax=axis)
    elif colorbar == "none":
        caxis = None
    else:
        raise NotImplementedError("invalid position")

    return axis, caxis
