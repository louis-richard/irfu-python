#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""set_rlabel.py
@author: Louis Richard
"""

import numpy as np

from ..pyrf import t_eval
from astropy.time import Time
from matplotlib.dates import num2date


def set_rlabel(ax, r_xyz, spines: list = None, position: str = "top"):
    r"""Add extra axes to plot spacecraft position.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Reference axis

    r_xyz : xarray.DataArray
        Time series of the spacecraft position

    spines : list, optional
        Relative position of the axes. Default is spines=.[60, 35, 10]

    position : str, optional
        Axis position wtr to the reference axis. Default is position="top"

    Returns
    -------
    axr : list
        List of the three new axes.

    """

    if spines is None:
        spines = [60, 35, 10]

    x_lim = ax.get_xlim()

    t_ticks = Time(num2date(ax.get_xticks()), format="datetime").datetime64
    r_ticks = np.transpose(t_eval(r_xyz, t_ticks).data)

    r_ticks_labels = [[f"{x:3.2f}" for x in ticks_] for ticks_ in r_ticks]

    axr = [ax.twiny() for _ in range(3)]

    for ax, ticks_labels, spine in zip(axr, r_ticks_labels, spines):
        ax.spines[position].set_position(("outward", spine))
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        ax.set_xticks(t_ticks)
        ax.set_xticklabels(ticks_labels)
        ax.set_xlim(x_lim)

    return axr
