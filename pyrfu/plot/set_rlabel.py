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

from astropy.time import Time
from matplotlib.dates import num2date
from ..pyrf import t_eval


def set_rlabel(ax, r_xyz, spine: float = 20, position: str = "top",
               fontsize: float = 10):
    r"""Add extra axes to plot spacecraft position.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Reference axis.

    r_xyz : xarray.DataArray
        Time series of the spacecraft position.

    spine : float, optional
        Relative position of the axes. Default is spines=20.

    position : str, optional
        Axis position wtr to the reference axis. Default is position="top".

    fontsize : float, optional
        xticks label font size. Default is 10.

    Returns
    -------
    axr : list
        List of the three new axes.

    """

    x_lim = ax.get_xlim()

    t_ticks = Time(num2date(ax.get_xticks()), format="datetime").datetime64
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
