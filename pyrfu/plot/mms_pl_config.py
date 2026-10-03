#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np

# Local imports
from ..constants import R_E

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

colors = ["tab:blue", "tab:green", "tab:red", "k"]
markers = ["s", "d", "o", "^"]


def mms_pl_config(r_mms):
    r"""Plot spacecraft configuration with three 2d plots of the position in
    Re and one 3d plot of the relative position of the spacecraft.

    Parameters
    ----------
    r_mms : list of xarray.DataArray
        Time series of the spacecraft position [km], one per spacecraft. The
        positions are averaged over time.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with MMS configuration plot.
    axs : list of matplotlib.axes.Axes
        Axes in the figure: X-Z, Y-Z and X-Y positions [R_E] and 3d position
        relative to the center of the tetrahedron [km].

    """

    # Mean position of each spacecraft and relative to the center [km]
    r_xyz = np.vstack([np.mean(r_xyz.data, axis=0) for r_xyz in r_mms])
    delta_r = r_xyz - np.mean(r_xyz, axis=0)

    fig = plt.figure(figsize=(9, 9))
    gs0 = fig.add_gridspec(
        3,
        3,
        hspace=0.3,
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
    )

    gs00 = gs0[0, :].subgridspec(1, 3, wspace=0.35)
    gs10 = gs0[1:, :].subgridspec(1, 1, wspace=0.35)

    axs0 = fig.add_subplot(gs00[0])
    axs1 = fig.add_subplot(gs00[1])
    axs2 = fig.add_subplot(gs00[2])
    axs3 = fig.add_subplot(gs10[0], projection="3d")

    x_lbs = ["$X$ [$R_E$]", "$Y$ [$R_E$]", "$X$ [$R_E$]"]
    y_lbs = ["$Z$ [$R_E$]", "$Z$ [$R_E$]", "$Y$ [$R_E$]"]

    axs_ = [axs0, axs1, axs2]
    idxs_, idys_ = [[0, 1, 0], [2, 2, 1]]

    # Keep +-20 R_E unless a spacecraft is further out
    lim_re = max(20.0, 1.1 * np.max(np.abs(r_xyz)) / R_E)

    for ax, idx_, idy_, x_lb, y_lb in zip(axs_, idxs_, idys_, x_lbs, y_lbs):
        for i, marker in enumerate(markers):
            ax.scatter(
                r_xyz[i, idx_] / R_E,
                r_xyz[i, idy_] / R_E,
                color=colors[i],
                marker=marker,
            )

        ax.add_artist(plt.Circle((0, 0), 1, color="k", clip_on=False))
        ax.set_xlim([lim_re, -lim_re])
        ax.set_ylim([-lim_re, lim_re])
        ax.set_aspect("equal")
        ax.set_xlabel(x_lb)
        ax.set_ylabel(y_lb)

    axs3.view_init(elev=13, azim=-20)

    lim_km = 1.2 * np.max(np.abs(delta_r))
    lim_km = lim_km if lim_km > 0 else 1.0

    for i, marker in enumerate(markers):
        options = {"s": 50, "marker": marker, "color": colors[i]}
        axs3.scatter(delta_r[i, 0], delta_r[i, 1], delta_r[i, 2], **options)

        # Projections on the walls where the dashed lines end
        options = {"color": colors[i], "marker": marker, "linestyle": ""}
        axs3.plot([delta_r[i, 0]], [delta_r[i, 1]], [-lim_km], **options)
        axs3.plot([delta_r[i, 0]], [lim_km], [delta_r[i, 2]], **options)
        axs3.plot([-lim_km], [delta_r[i, 1]], [delta_r[i, 2]], **options)

        options = {"color": "k", "linestyle": "--", "linewidth": 0.5}
        axs3.plot(
            [delta_r[i, 0]] * 2,
            [delta_r[i, 1]] * 2,
            [-lim_km, delta_r[i, 2]],
            **options,
        )
        axs3.plot(
            [delta_r[i, 0]] * 2,
            [lim_km, delta_r[i, 1]],
            [delta_r[i, 2]] * 2,
            **options,
        )
        axs3.plot(
            [-lim_km, delta_r[i, 0]],
            [delta_r[i, 1]] * 2,
            [delta_r[i, 2]] * 2,
            **options,
        )

    for idx_0, idx_1 in zip([0, 1, 2, 0, 1, 2], [1, 2, 0, 3, 3, 3]):
        axs3.plot(
            delta_r[[idx_0, idx_1], 0],
            delta_r[[idx_0, idx_1], 1],
            delta_r[[idx_0, idx_1], 2],
            "k-",
        )

    axs3.set_xlim([-lim_km, lim_km])
    axs3.set_ylim([lim_km, -lim_km])
    axs3.set_zlim([-lim_km, lim_km])
    axs3.set_xlabel(r"$\Delta X$ [km]")
    axs3.set_ylabel(r"$\Delta Y$ [km]")
    axs3.set_zlabel(r"$\Delta Z$ [km]")

    axs3.legend(axs3.collections[:4], ["MMS1", "MMS2", "MMS3", "MMS4"], frameon=False)

    axs = [axs0, axs1, axs2, axs3]

    return fig, axs
