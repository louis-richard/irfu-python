#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2021 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""span_tint.py
@author: Louis Richard
"""

from dateutil import parser as date_parser


def span_tint(axs, tint, ymin=0, ymax=1, **kwargs):
    r"""Add a vertical span (rectangle) across the time Axes.

    The rectangle spans from tint[0] to tint[1] horizontally, and, by default,
    the whole y-axis vertically. The y-span can be set using ymin (default: 0)
    and ymax (default: 1) which are in axis units; e.g. ymin = 0.5 always
    refers to the middle of the y-axis regardless of the limits set by
    set_ylim

    Parameters
    ----------
    axs : list
        List of matplotlib.axes._subplots.AxesSubplot to span.

    tint : list
        Time interval to span

    ymin : float
        Lower y-coordinate of the span, in y-axis units (0-1). Default ymin=0.

    ymax : float, optional
        Upper y-coordinate of the span, in y-axis units (0-1). Default ymax=1.


    Returns
    -------


    """
    for axis in axs:
        t_start, t_stop = [date_parser.parse(tint[0]),
                           date_parser.parse(tint[1])]
        axis.axvspan(t_start, t_stop, ymin, ymax, **kwargs)

    return axs
