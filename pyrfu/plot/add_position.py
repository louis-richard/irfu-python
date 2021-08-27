#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from matplotlib.dates import num2date

# Local imports
from ..pyrf import t_eval

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def add_position(ax, r_xyz, spine: float = 20, position: str = "top",
                 fontsize: float = 10):
    r"""Add extra axes to plot spacecraft position.

    Parameters
    ----------
    ax : matplotlib.pyplot.subplotsaxes
        Reference axis.
    r_xyz : xarray.DataArray
        Time series of the spacecraft position.
    spine : float, Optional
        Relative position of the axes. Default is spines=20.
    position : str, Optional
        Axis position wtr to the reference axis. Default is position="top".
    fontsize : float, Optional
        xticks label font size. Default is 10.

    Returns
    -------
    axr : matplotlib.pyplot.subplotsaxes
        Twin axis with spacecraft position as x-axis label.

    """

    x_lim = ax.get_xlim()

    t_ticks = [t_.replace(tzinfo=None) for t_ in num2date(ax.get_xticks())]
    t_ticks = np.array(t_ticks).astype("<M8[ns]")
    r_ticks = t_eval(r_xyz, t_ticks).data

    ticks_labels = []
    for ticks_ in r_ticks:
        ticks_labels.append(
            f"{ticks_[0]:3.2f}\n{ticks_[1]:3.2f}\n{ticks_[2]:3.2f}")

    axr = ax.twiny()
    axr.spines[position].set_position(("outward", spine))
    axr.xaxis.set_ticks_position(position)
    axr.xaxis.set_label_position(position)
    axr.set_xticks(t_ticks)
    axr.set_xticklabels(ticks_labels, fontsize=fontsize)
    axr.set_xlim(x_lim)

    return axr
