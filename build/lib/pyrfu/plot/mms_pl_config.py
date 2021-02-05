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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use("seaborn-whitegrid")
date_form = mdates.DateFormatter("%H:%M:%S")
sns.set_context("paper")

plt.rc("lines", linewidth=1)
colors = np.array([[0, 0, 0], [213, 94, 0], [0, 158, 115], [86, 180, 233]]) / 255
markers = ["s", "d", "o", "^"]


def mms_pl_config(r_mms):
    """Plot spacecraft configuaration with three 2d plots of the position in re and one 3d plot of
    the relative position of the spacecraft:

    Parameters
    ----------
    r_mms : list of xarray.DataArray
        Time series of the spacecraft position

    Returns
    -------
    fig : matplotlib.pyplot.figure
        to fill.

    axs : matplotlib.pyplot.subplotsaxes
        to fill.

    """

    r_earth = 6378.136
    r_xyz = np.vstack([np.mean(r_xyz, 0) for r_xyz in r_mms])
    r_xyz = np.mean(r_xyz, 0)
    delta_r = r_xyz - np.tile(r_xyz, (4, 1))

    fig = plt.figure(figsize=(9, 9))
    gs0 = fig.add_gridspec(3, 3, hspace=0.3, left=0.1, right=0.9, bottom=0.1, top=0.9)

    gs00 = gs0[0, :].subgridspec(1, 3, wspace=0.35)
    gs10 = gs0[1:, :].subgridspec(1, 1, wspace=0.35)

    axs0 = fig.add_subplot(gs00[0])
    axs1 = fig.add_subplot(gs00[1])
    axs2 = fig.add_subplot(gs00[2])
    axs3 = fig.add_subplot(gs10[0], projection='3d')

    earth = plt.Circle((0, 0), 1, color='k', clip_on=False)

    x_axis_labels = ["$X$ [$r_earth$]", "$Y$ [$r_earth$]", "$X$ [$r_earth$]"]
    y_axis_labels = ["$Z$ [$r_earth$]", "$Z$ [$r_earth$]", "$Y$ [$r_earth$]"]

    for axis, x_idx, y_idx, x_label, y_label in zip([axs0, axs1, axs2], [0, 1, 0], [2, 1, 1],
                                                    x_axis_labels, y_axis_labels):
        for i, marker in enumerate(markers):
            axis.scatter(r_xyz[i, x_idx] / r_earth, r_xyz[i, y_idx] / r_earth, marker=marker)

        axis.add_artist(earth)
        axis.set_xlim([20, -20])
        axis.set_ylim([-20, 20])
        axis.set_aspect("equal")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

    axs3.view_init(elev=13, azim=-20)

    for i, marker in enumerate(markers):
        options = dict(s=50, marker=marker)
        axs3.scatter(delta_r[i, 0], delta_r[i, 1], delta_r[i, 2], **options)

        options = dict(color=colors[i], marker=marker, zdir='z', zs=-30)
        axs3.plot([delta_r[i, 0]] * 2, [delta_r[i, 1]] * 2, **options)
        axs3.plot([delta_r[i, 0]] * 2, [delta_r[i, 2]] * 2, **options)
        axs3.plot([delta_r[i, 1]] * 2, [delta_r[i, 2]] * 2, **options)

        options = dict(color="k", linestyle="--", linewidth=.5)
        axs3.plot([delta_r[i, 0]] * 2, [delta_r[i, 1]] * 2, [-30, delta_r[i, 2]], **options)
        axs3.plot([delta_r[i, 0]] * 2, [-30, delta_r[i, 1]], [delta_r[i, 2]] * 2, **options)
        axs3.plot([-30, delta_r[i, 0]], [delta_r[i, 1]] * 2, [delta_r[i, 2]] * 2, **options)

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
