# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
colorbar.py

@author : Louis RICHARD
"""

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def colorbar(im, ax, pad=0.01):
    """Add colorbar to ax corresponding to im.

    Parameters
    ----------
    im : to fill
        to fill.

    ax : matplotlib.pyplot.subplotsaxes
        Axis of plot.

    pad : float, optional


    Returns
    -------
    cax : coloraxis

    """
    pos = ax.get_position()
    fig = plt.gcf()
    cax = fig.add_axes([pos.x0 + pos.width + pad, pos.y0, pad, pos.height])
    fig.colorbar(im, cax)

    return cax
