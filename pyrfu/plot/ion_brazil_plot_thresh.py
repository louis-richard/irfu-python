#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Literal

# 3rd party imports
import numpy as np

# Local imports
from ..models.ion_anisotropy_thresh import ion_anisotropy_thresh


def ion_brazil_plot_thresh(ax, legend: bool = True, colors: list | None = None):
    r"""Add the thresholds for the electromagnetic ion temperature anisotropy driven
    instabilities from linear Vlasov theory [1]_ [2]_ . The thresholds are given at
    :math:`\gamma = 10^{-2} \omega_{ci}` . Also changes the scale and limits of the
    axis to make it in form of a Brazil plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where to plot the thresholds in a Brazil plot format.
    legend : bool, Optional
        Add a legend of the four thresholds at the top of the figure of `ax`.
        Default is True.
    colors : list of str, Optional
        Colors of the proton cyclotron, mirror mode, parallel firehose and
        oblique firehose thresholds. Default is ["deeppink", "m", "b", "c"].

    References
    ----------
    .. [1]  Daniel Verscharen et al 2016 ApJ 831 128
    .. [2]  Bennett A. Maruca et al 2012 ApJ 748 137

    """

    if colors is None:
        colors = ["deeppink", "m", "b", "c"]

    # Proton - electron plasma thresholds [1]_: proton cyclotron, mirror-mode,
    # parallel firehose and oblique firehose instabilities
    thresholds: list[
        tuple[
            Literal[
                "proton-cyclotron", "mirror", "parallel-firehose", "oblique-firehose"
            ],
            str,
        ]
    ] = [
        ("proton-cyclotron", "Proton cyclotron"),
        ("mirror", "Mirror Mode"),
        ("parallel-firehose", "Parallel Firehose"),
        ("oblique-firehose", "Oblique Firehose"),
    ]
    beta_para = np.logspace(-2, 4)

    lines = []

    for i, (instability, _) in enumerate(thresholds):
        (line,) = ax.loglog(
            beta_para,
            ion_anisotropy_thresh(beta_para, instability, "10^-2"),
            color=colors[i],
            linestyle="--",
            lw=1.5,
        )
        lines.append(line)

    ax.set_xlim([1e-2, 5e3])
    ax.set_ylim([0.3, 1e1])

    f = ax.get_figure()
    if legend:
        f.legend(
            handles=lines,
            labels=[label for _, label in thresholds],
            loc="upper center",
            borderaxespad=0.1,
            ncol=4,
            frameon=False,
            handlelength=1.0,
        )
