#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib.pyplot as plt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def colorbar(mappable, axis, pad: float = 0.01):
    r"""Add colorbar to ax corresponding to im.

    Parameters
    ----------
    mappable : matplotlib.collections.QuadMesh
        The quadrilateral mesh mappable described by this colorbar.
    axis : matplotlib.pyplot.subplotsaxes
        Axis of plot.
    pad : float, Optional
        Shift the colorbar with respect to the axis.

    Returns
    -------
    cax : matplotlib.colorbar.Colorbar
        Colorbar added to the plot.

    """

    pos = axis.get_position()
    fig = plt.gcf()
    cax = fig.add_axes([pos.x0 + pos.width + pad, pos.y0, pad, pos.height])
    fig.colorbar(mappable, cax)

    return cax
