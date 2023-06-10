#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import string

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def make_labels(axs, pos, pad: float = 0, **kwargs):
    r"""Add subplots labels to axes

    Parameters
    ----------
    axs : ndarray
        Array of subplots axes.
    pos : array_like
        Position of the text in the axis.

    Returns
    -------
    axs : ndarray
        Array of subplots axes with labels.

    """

    lbl = string.ascii_lowercase[pad : len(axs) + pad]

    for label, axis in zip(lbl, axs):
        if "proj" in axis.properties():
            axis.text2D(
                pos[0],
                pos[1],
                f"({label})",
                transform=axis.transAxes,
                **kwargs,
            )
        else:
            axis.text(
                pos[0],
                pos[1],
                f"({label})",
                transform=axis.transAxes,
                **kwargs,
            )

    return axs
