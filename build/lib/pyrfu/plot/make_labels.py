#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import string

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def make_labels(axs, pos, pad: float = 0):
    r"""Add subplots labels to axes

    Parameters
    ----------
    axs : ndarray
        Array of subplots axes.
    pos : array_like
        Position of the text in the axis.
    pad : int, Optional
        Offset in axis counter.

    Returns
    -------
    axs : ndarray
        Array of subplots axes with labels.

    """

    lbl = string.ascii_lowercase[pad:len(axs) + pad]

    for label, axis in zip(lbl, axs):
        if "proj" in axis.properties():
            axis.text2D(pos[0], pos[1], "({})".format(label),
                        transform=axis.transAxes,
                        bbox=dict(boxstyle="square", ec=(1., 1., 1.),
                                  fc=(1., 1., 1.)))
        else:
            axis.text(pos[0], pos[1], "({})".format(label),
                      transform=axis.transAxes,
                      bbox=dict(boxstyle="square", ec=(1., 1., 1.),
                                fc=(1., 1., 1.)))

    return axs
