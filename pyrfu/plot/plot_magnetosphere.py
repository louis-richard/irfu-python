#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 - 2021 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""plot_magnetosphere.py
@author: Louis Richard
"""

import numpy as np

from matplotlib.patches import Wedge

from ..pyrf import magnetosphere


def _add_earth(ax=None, **kwargs):
    theta1, theta2 = 90., 270.
    nightside_ = Wedge((0., 0.), 1., theta1, theta2, fc="k", ec="k", **kwargs)
    dayside_ = Wedge((0., 0.), 1., theta2, theta1, fc="w", ec="k", **kwargs)
    for wedge in [nightside_, dayside_]:
        ax.add_artist(wedge)
    return [nightside_, dayside_]


def plot_magnetosphere(ax, b_z, d_p, colors=None):
    r"""Plot magnetopause, bow shock and earth.

    Parameters
    ----------
    ax : axis
        Axis to plot.
    b_z : float
        IMF B_Z in GSM.
    d_p : float
        Solar wind dynamic pressure.
    colors : list, optional
        Colors of the magnetopause and the bow show.

    Returns
    -------
    ax : axis
        Axis.

    """

    # Compute Magnetopause
    if colors is None:
        colors = ["tab:blue", "tab:red"]

    x_mp, y_mp = magnetosphere("mp_shue1998", d_p, b_z)

    # Compute bow show
    x_bs, y_bs = magnetosphere("bs", d_p, b_z)

    # Plot
    ax.plot(np.hstack([x_mp, np.flip(x_mp)]),
            np.hstack([y_mp, np.flip(-y_mp)]),
            color=colors[0], label="Magnetopause")
    ax.plot(np.hstack([x_bs, np.flip(x_bs)]),
            np.hstack([y_bs, np.flip(-y_bs)]),
            color=colors[1], label="Bow Shock")
    _add_earth(ax)

    return ax
