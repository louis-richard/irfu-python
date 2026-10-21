#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def colorbar(mappable, axis, pad: float = 0.01, width: float = 0.03):
    r"""Add a colorbar of `mappable` on the right of `axis`.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The mappable described by this colorbar, e.g. the
        matplotlib.collections.QuadMesh returned by pcolormesh.
    axis : matplotlib.axes.Axes
        Axis of the plot.
    pad : float, Optional
        Gap between the axis and the colorbar, in figure coordinates. Default
        is 0.01.
    width : float, Optional
        Width of the colorbar, in figure coordinates. Default is 0.03.

    Returns
    -------
    cax : matplotlib.axes.Axes
        Axis of the colorbar, added to the figure of `axis`.

    """

    pos = axis.get_position()
    fig = axis.get_figure()
    cax = fig.add_axes([pos.x0 + pos.width + pad, pos.y0, width, pos.height])
    fig.colorbar(mappable, cax)

    return cax
