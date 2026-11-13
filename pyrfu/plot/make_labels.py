#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import string
from collections.abc import Sequence

# 3rd party imports
import numpy as np


def make_labels(axs, pos, pref=None, suff=None, pad: int = 0, **kwargs):
    r"""Add subplots labels to axes

    Parameters
    ----------
    axs : array_like of matplotlib.axes.Axes
        Subplots axes, a list or an array (e.g., the 2-D array of
        plt.subplots(2, 2), labelled row by row).
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
    axs : array_like of matplotlib.axes.Axes
        Subplots axes with labels (`axs` as given).

    """

    # A 2-D array of axes raised AttributeError (its rows taken as axes)
    axs_flat = np.ravel(axs)

    if len(axs_flat) + pad > 26:
        raise ValueError("Number of subplots exceeds the number of labels available.")

    lbl: Sequence[str] = string.ascii_lowercase[pad : len(axs_flat) + pad]

    if pref is not None:
        lbl = [f"{pref}{lbl[i]}" for i in range(len(lbl))]

    if suff is not None:
        lbl = [f"{lbl[i]}{suff}" for i in range(len(lbl))]

    for label, axis in zip(lbl, axs_flat):
        if axis.name == "3d":
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
