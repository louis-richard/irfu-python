#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from pyrfu.models.ion_anisotropy_thresh import ion_anisotropy_thresh


def ion_brazil_plot_thresh(ax, legend: bool = True, colors: list = None):
    r"""Add the thresholds for the electromagnetic ion temperature anisotropy driven
    instabilities from linear Vlasov theory [1]_ [2]_ . The thresholds are given at
    :math:`\gamma = 10^{-2} \omega_{ci}` . Also changes the scale and limits of the
    axis to make it in form of a Brazil plot.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis where to plot the thresholds in a Brazil plot format.


    References
    ----------
    .. [1]  Daniel Verscharen et al 2016 ApJ 831 128
    .. [2]  Bennett A. Maruca et al 2012 ApJ 748 137

    """

    if colors is None:
        colors = ["deeppink", "m", "b", "c"]

    # Proton - electron plasma thresholds [1]_
    # Proton cyclotron instability
    (l0,) = ax.loglog(
        np.logspace(-2, 4),
        ion_anisotropy_thresh(np.logspace(-2, 4), "proton-cyclotron", "10^-2"),
        color=colors[0],
        linestyle="--",
        lw=1.5,
    )

    # Mirror-mode instability
    (l1,) = ax.loglog(
        np.logspace(-2, 4),
        ion_anisotropy_thresh(np.logspace(-2, 4), "mirror", "10^-2"),
        color=colors[1],
        linestyle="--",
        lw=1.5,
    )

    # Parallel firehose
    (l2,) = ax.loglog(
        np.logspace(-2, 4),
        ion_anisotropy_thresh(np.logspace(-2, 4), "parallel-firehose", "10^-2"),
        color=colors[2],
        linestyle="--",
        lw=1.5,
    )

    # Oblique firehose
    (l3,) = ax.loglog(
        np.logspace(-2, 4),
        ion_anisotropy_thresh(np.logspace(-2, 4), "oblique-firehose", "10^-2"),
        color=colors[3],
        linestyle="--",
        lw=1.5,
    )
    ax.set_xlim([1e-2, 5e3])
    ax.set_ylim([0.3, 1e1])

    f = plt.gcf()
    if legend:
        f.legend(
            handles=[l0, l1, l2, l3],  # The line objects
            labels=[
                "Proton cyclotron",
                "Mirror Mode",
                "Parallel Firehose",
                "Oblique Firehose",
            ],
            loc="upper center",
            borderaxespad=0.1,
            ncol=4,
            frameon=False,
            handlelength=1.0,
        )
