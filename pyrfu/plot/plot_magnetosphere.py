#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from matplotlib.patches import Wedge

# Local imports
from ..pyrf import magnetosphere

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _add_earth(ax=None, **kwargs):
    theta1, theta2 = 90., 270.
    nightside_ = Wedge((0., 0.), 1., theta1, theta2, fc="k", ec="k", **kwargs)
    dayside_ = Wedge((0., 0.), 1., theta2, theta1, fc="w", ec="k", **kwargs)
    for wedge in [nightside_, dayside_]:
        ax.add_artist(wedge)
    return [nightside_, dayside_]


def plot_magnetosphere(ax, tint, colors: list = None):
    r"""Plot magnetopause, bow shock and earth.

    Parameters
    ----------
    ax : matplotlib.pyplot.subplotsaxes
        Axis to plot.
    tint : list of str
        Time interval.
    colors : list, Optional
        Colors of the magnetopause and the bow show.
        Default use ["tab:blue", "tab:red"]

    Returns
    -------
    ax : matplotlib.pyplot.subplotsaxes
        Axis.

    """

    # Compute Magnetopause
    if colors is None:
        colors = ["tab:blue", "tab:red"]

    x_mp, y_mp = magnetosphere("mp_shue1998", tint)

    # Compute bow show
    x_bs, y_bs = magnetosphere("bs", tint)

    # Plot
    ax.plot(np.hstack([x_mp, np.flip(x_mp)]),
            np.hstack([y_mp, np.flip(-y_mp)]),
            color=colors[0], label="Magnetopause")
    ax.plot(np.hstack([x_bs, np.flip(x_bs)]),
            np.hstack([y_bs, np.flip(-y_bs)]),
            color=colors[1], label="Bow Shock")
    _add_earth(ax)

    return ax
