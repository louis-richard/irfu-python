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


def make_labels(axs, pos, pref=None, suff=None, pad: int = 0, **kwargs):
    r"""Add subplots labels to axes

    Parameters
    ----------
    axs : ndarray
        Array of subplots axes.
    pos : array_like
        Position of the text in the axis.
    pref : str, Optional
        Prefix to append to the label.
    suff : str, Optional
        Suffix to append to the label.
    pad : int, Optional
        Number of labels to skip.

    Returns
    -------
    axs : ndarray
        Array of subplots axes with labels.

    """

    if len(axs) + pad > 26:
        raise ValueError("Number of subplots exceeds the number of labels available.")

    lbl = string.ascii_lowercase[pad : len(axs) + pad]

    if pref is not None:
        lbl = [f"{pref}{lbl[i]}" for i in range(len(lbl))]

    if suff is not None:
        lbl = [f"{lbl[i]}{suff}" for i in range(len(lbl))]

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
