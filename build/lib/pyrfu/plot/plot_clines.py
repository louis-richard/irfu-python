#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party import
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from matplotlib.colorbar import ColorbarBase

# Local imports
from .plot_line import plot_line

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def plot_clines(axis, inp, yscale="log", cscale="log", cmap="jet", **kwargs):
    r"""Plot lines with color associated to the level.

    Parameters
    ----------
    axis :
        Axes
    inp : xarray.DataArray
        Time series as an energy spectrum to plot.
    yscale : str, Optional
        Scale of the yaxis. Default is "log"
    cscale : str, Optional
        Scale of the colormap. Default is "log".
    cmap : str, Optional
        Colormap. Default is "jet"
    kwargs : dict
        Plot options.

    Returns
    -------
    axis :
        Updated axis
    cbl :
        Colorbar associated

    Other Parameters
    ----------------
    See pyrfu.plot.plot_line

    """

    pad = .01
    c_map = get_cmap(name=cmap)

    for i, c in enumerate(c_map(np.linspace(0, 1, len(inp.energy.data)))):
        plot_line(axis, inp[:, i], color=c, **kwargs)

    pos = axis.get_position()
    f = plt.gcf()
    cax = f.add_axes([pos.x0 + pos.width + pad, pos.y0, pad, pos.height])

    if cscale == "log":
        norm = LogNorm(vmin=inp.energy.data[0], vmax=inp.energy.data[-1])
    else:
        raise NotImplementedError

    ColorbarBase(cax, cmap=c_map, norm=norm, orientation="vertical")

    if yscale == "log":
        axis.set_yscale("log")

    cax.set_axisbelow(False)

    return axis, cax
