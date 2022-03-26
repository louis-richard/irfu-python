#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

plt.style.use("seaborn-ticks")
colors = ["tab:blue", "tab:green", "tab:red", "k"]
markers = ["s", "d", "o", "^"]


def mms_pl_config(r_mms):
    r"""Plot spacecraft configuaration with three 2d plots of the position in
    Re and one 3d plot of the relative position of the spacecraft:

    Parameters
    ----------
    r_mms : list of xarray.DataArray
        Time series of the spacecraft position

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure with MMS configuration plot.
    axs : matplotlib.pyplot.subplotsaxes
        Axes in the figure.

    """

    r_earth = 6378.136
    r_xyz = np.vstack([np.mean(r_xyz, 0) for r_xyz in r_mms])
    r_xyz = np.mean(r_xyz, 0)
    delta_r = r_xyz - np.tile(r_xyz, (4, 1))

    fig = plt.figure(figsize=(9, 9))
    gs0 = fig.add_gridspec(3, 3, hspace=0.3, left=0.1, right=0.9,
                           bottom=0.1, top=0.9)

    gs00 = gs0[0, :].subgridspec(1, 3, wspace=0.35)
    gs10 = gs0[1:, :].subgridspec(1, 1, wspace=0.35)

    axs0 = fig.add_subplot(gs00[0])
    axs1 = fig.add_subplot(gs00[1])
    axs2 = fig.add_subplot(gs00[2])
    axs3 = fig.add_subplot(gs10[0], projection='3d')

    earth = plt.Circle((0, 0), 1, color='k', clip_on=False)

    x_lbs = ["$X$ [$r_earth$]", "$Y$ [$r_earth$]", "$X$ [$r_earth$]"]
    y_lbs = ["$Z$ [$r_earth$]", "$Z$ [$r_earth$]", "$Y$ [$r_earth$]"]

    axs_ = [axs0, axs1, axs2]
    idxs_, idys_ = [[0, 1, 0], [2, 1, 1]]

    for ax, idx_, idy_, x_lb, y_lb in zip(axs_, idxs_, idys_, x_lbs, y_lbs):
        for i, marker in enumerate(markers):
            ax.scatter(r_xyz[i, idx_] / r_earth, r_xyz[i, idy_] / r_earth,
                       marker=marker)

        ax.add_artist(earth)
        ax.set_xlim([20, -20])
        ax.set_ylim([-20, 20])
        ax.set_aspect("equal")
        ax.set_xlabel(x_lb)
        ax.set_ylabel(y_lb)

    axs3.view_init(elev=13, azim=-20)

    for i, marker in enumerate(markers):
        options = dict(s=50, marker=marker)
        axs3.scatter(delta_r[i, 0], delta_r[i, 1], delta_r[i, 2], **options)

        options = dict(color=colors[i], marker=marker, zdir='z', zs=-30)
        axs3.plot([delta_r[i, 0]] * 2, [delta_r[i, 1]] * 2, **options)
        axs3.plot([delta_r[i, 0]] * 2, [delta_r[i, 2]] * 2, **options)
        axs3.plot([delta_r[i, 1]] * 2, [delta_r[i, 2]] * 2, **options)

        options = dict(color="k", linestyle="--", linewidth=.5)
        axs3.plot([delta_r[i, 0]] * 2, [delta_r[i, 1]] * 2,
                  [-30, delta_r[i, 2]], **options)
        axs3.plot([delta_r[i, 0]] * 2, [-30, delta_r[i, 1]],
                  [delta_r[i, 2]] * 2, **options)
        axs3.plot([-30, delta_r[i, 0]], [delta_r[i, 1]] * 2,
                  [delta_r[i, 2]] * 2, **options)

    for idx_0, idx_1 in zip([0, 1, 2, 0, 1, 2], [1, 2, 0, 3, 3, 3]):
        axs3.plot(delta_r[[idx_0, idx_1], 0], delta_r[[idx_0, idx_1], 1],
                  delta_r[[idx_0, idx_1], 2], 'k-')

    axs3.set_xlim([-30, 30])
    axs3.set_ylim([30, -30])
    axs3.set_zlim([-30, 30])
    axs3.set_xlabel("$\\Delta X$ [km]")
    axs3.set_ylabel("$\\Delta Y$ [km]")
    axs3.set_zlabel("$\\Delta Z$ [km]")

    axs3.legend(["MMS1", "MMS2", "MMS3", "MMS4"], frameon=False)

    axs = [axs0, axs1, axs2, axs3]

    return fig, axs
