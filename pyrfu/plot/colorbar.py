#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def colorbar(mappable, axis, pad: float = 0.01, width: float = 0.03):
    r"""Add colorbar to ax corresponding to im.

    Parameters
    ----------
    mappable : matplotlib.collections.QuadMesh
        The quadrilateral mesh mappable described by this colorbar.
    axis : matplotlib.pyplot.subplotsaxes
        Axis of plot.
    pad : float, Optional
        Shift the colorbar with respect to the axis.
    width : float, Optional
        Width of the colorbar.

    Returns
    -------
    cax : matplotlib.colorbar.Colorbar
        Colorbar added to the plot.

    """

    pos = axis.get_position()
    fig = axis.get_figure()
    cax = fig.add_axes([pos.x0 + pos.width + pad, pos.y0, width, pos.height])
    fig.colorbar(mappable, cax)

    return cax
