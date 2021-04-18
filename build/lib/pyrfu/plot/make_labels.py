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

"""make_labels.py
@author: Louis Richard
"""

import string


def make_labels(axs, pos, pad=0):
    r"""Add subplots labels to axes

    Parameters
    ----------
    axs : numpy.ndarray
        Array of subplots axes.

    pos : list or numpy.ndarray
        Position of the text in the axis.

    pad : int
        (Option) Offset in axis counter.

    Returns
    -------
    axs : numpy.ndarray
        Array of subplots axes with labels.

    """

    lbl = string.ascii_lowercase[pad:len(axs) + pad]

    for label, axis in zip(lbl, axs):
        axis.text(pos[0], pos[1], "({})".format(label),
                  transform=axis.transAxes)

    return axs
