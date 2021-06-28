#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from matplotlib import dates

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def span_tint(axs, tint, ymin: float = 0, ymax:float = 1, **kwargs):
    r"""Add a vertical span (rectangle) across the time Axes.

    The rectangle spans from tint[0] to tint[1] horizontally, and, by default,
    the whole y-axis vertically. The y-span can be set using ymin (default: 0)
    and ymax (default: 1) which are in axis units; e.g. ymin = 0.5 always
    refers to the middle of the y-axis regardless of the limits set by
    set_ylim

    Parameters
    ----------
    axs : list of matplotlib.pyplot.subplotsaxes
        Axes to span.
    tint : list of str
        Time interval to span
    ymin : float
        Lower y-coordinate of the span, in y-axis units (0-1). Default ymin=0.
    ymax : float, Optional
        Upper y-coordinate of the span, in y-axis units (0-1). Default ymax=1.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments control the Polygon properties.

    Returns
    -------
    axs : list
        List of matplotlib.axes._subplots.AxesSubplot spanned..

    """

    for axis in axs:
        t_start, t_stop = [dates.datestr2num(tint[0]),
                           dates.datestr2num(tint[1])]
        axis.axvspan(t_start, t_stop, ymin, ymax, **kwargs)

    return axs
