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

from astropy import constants
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.style.use("seaborn-whitegrid")
date_form = mdates.DateFormatter("%H:%M:%S")
sns.set_context("paper")

plt.rc("lines", linewidth=1)
color = np.array([[0, 0, 0], [213, 94, 0], [0, 158, 115], [86, 180, 233]]) / 255


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

    r_earth = constants.r_earth.value / 1000
    r1, r2, r3, r4 = [np.mean(r_xyz, 0) for r_xyz in r_mms]

    r = np.vstack([r1, r2, r3, r4])
    r = np.mean(r, 0)
    dr = r - np.tile(r, (4, 1))

    fig = plt.figure(figsize=(9, 9))
    gs0 = fig.add_gridspec(3, 3, hspace=0.3, left=0.1, right=0.9, bottom=0.1, top=0.9)

    gs00 = gs0[0, :].subgridspec(1, 3, wspace=0.35)
    gs10 = gs0[1:, :].subgridspec(1, 1, wspace=0.35)

    axs0 = fig.add_subplot(gs00[0])
    axs1 = fig.add_subplot(gs00[1])
    axs2 = fig.add_subplot(gs00[2])
    axs3 = fig.add_subplot(gs10[0], projection='3d')

    axs0.scatter(r[0, 0] / r_earth, r[0, 2] / r_earth, marker="s")
    axs0.scatter(r[1, 0] / r_earth, r[1, 2] / r_earth, marker="d")
    axs0.scatter(r[2, 0] / r_earth, r[2, 2] / r_earth, marker="o")
    axs0.scatter(r[3, 0] / r_earth, r[3, 2] / r_earth, marker="^")
    earth = plt.Circle((0, 0), 1, color='k', clip_on=False)
    axs0.add_artist(earth)
    axs0.set_xlim([20, -20])
    axs0.set_ylim([-20, 20])
    axs0.set_aspect("equal")
    axs0.set_xlabel("$X$ [$r_earth$]")
    axs0.set_ylabel("$Z$ [$r_earth$]")
    axs0.set_title("X = {:2.1f} $r_earth$".format(r[0] / r_earth))

    axs1.scatter(r[0, 1] / r_earth, r[0, 2] / r_earth, marker="s")
    axs1.scatter(r[1, 1] / r_earth, r[1, 2] / r_earth, marker="d")
    axs1.scatter(r[2, 1] / r_earth, r[2, 2] / r_earth, marker="o")
    axs1.scatter(r[3, 1] / r_earth, r[3, 2] / r_earth, marker="^")
    earth = plt.Circle((0, 0), 1, color='k', clip_on=False)
    axs1.add_artist(earth)
    axs1.set_xlim([20, -20])
    axs1.set_ylim([-20, 20])
    axs1.set_aspect("equal")
    axs1.set_xlabel("$Y$ [$r_earth$]")
    axs1.set_ylabel("$Z$ [$r_earth$]")
    axs1.set_title("Y = {:2.1f} $r_earth$".format(r[1] / r_earth))

    axs2.scatter(r[0, 0] / r_earth, r[0, 1] / r_earth, marker="s")
    axs2.scatter(r[1, 0] / r_earth, r[1, 1] / r_earth, marker="d")
    axs2.scatter(r[2, 0] / r_earth, r[2, 1] / r_earth, marker="o")
    axs2.scatter(r[3, 0] / r_earth, r[3, 1] / r_earth, marker="^")
    earth = plt.Circle((0, 0), 1, color='k', clip_on=False)
    axs2.add_artist(earth)
    axs2.set_xlim([20, -20])
    axs2.set_ylim([-20, 20])
    axs2.set_aspect("equal")
    axs2.set_xlabel("$X$ [$r_earth$]")
    axs2.set_ylabel("$Y$ [$r_earth$]")
    axs2.set_title("Z = {:2.1f} $r_earth$".format(r[2] / r_earth))

    axs3.view_init(elev=13, azim=-20)
    axs3.scatter(dr[0, 0], dr[0, 1], dr[0, 2], s=50, marker="s")
    axs3.scatter(dr[1, 0], dr[1, 1], dr[1, 2], s=50, marker="d")
    axs3.scatter(dr[2, 0], dr[2, 1], dr[2, 2], s=50, marker="o")
    axs3.scatter(dr[3, 0], dr[3, 1], dr[3, 2], s=50, marker="^")

    axs3.plot([dr[0, 0]] * 2, [dr[0, 1]] * 2, color=color[0], marker="s", zdir='z', zs=-30)
    axs3.plot([dr[1, 0]] * 2, [dr[1, 1]] * 2, color=color[1], marker="d", zdir='z', zs=-30)
    axs3.plot([dr[2, 0]] * 2, [dr[2, 1]] * 2, color=color[2], marker="o", zdir='z', zs=-30)
    axs3.plot([dr[3, 0]] * 2, [dr[3, 1]] * 2, color=color[3], marker="^", zdir='z', zs=-30)

    axs3.plot([dr[0, 0]] * 2, [dr[0, 2]] * 2, color=color[0], marker="s", zdir='y', zs=-30)
    axs3.plot([dr[1, 0]] * 2, [dr[1, 2]] * 2, color=color[1], marker="d", zdir='y', zs=-30)
    axs3.plot([dr[2, 0]] * 2, [dr[2, 2]] * 2, color=color[2], marker="o", zdir='y', zs=-30)
    axs3.plot([dr[3, 0]] * 2, [dr[3, 2]] * 2, color=color[3], marker="^", zdir='y', zs=-30)

    axs3.plot([dr[0, 1]] * 2, [dr[0, 2]] * 2, color=color[0], marker="s", zdir='x', zs=-30)
    axs3.plot([dr[1, 1]] * 2, [dr[1, 2]] * 2, color=color[1], marker="d", zdir='x', zs=-30)
    axs3.plot([dr[2, 1]] * 2, [dr[2, 2]] * 2, color=color[2], marker="o", zdir='x', zs=-30)
    axs3.plot([dr[3, 1]] * 2, [dr[3, 2]] * 2, color=color[3], marker="^", zdir='x', zs=-30)

    axs3.plot([dr[0, 0]] * 2, [dr[0, 1]] * 2, [-30, dr[0, 2]], 'k--', linewidth=.5)
    axs3.plot([dr[1, 0]] * 2, [dr[1, 1]] * 2, [-30, dr[1, 2]], 'k--', linewidth=.5)
    axs3.plot([dr[2, 0]] * 2, [dr[2, 1]] * 2, [-30, dr[2, 2]], 'k--', linewidth=.5)
    axs3.plot([dr[3, 0]] * 2, [dr[3, 1]] * 2, [-30, dr[3, 2]], 'k--', linewidth=.5)

    axs3.plot([dr[0, 0]] * 2, [-30, dr[0, 1]], [dr[0, 2]] * 2, 'k--', linewidth=.5)
    axs3.plot([dr[1, 0]] * 2, [-30, dr[1, 1]], [dr[1, 2]] * 2, 'k--', linewidth=.5)
    axs3.plot([dr[2, 0]] * 2, [-30, dr[2, 1]], [dr[2, 2]] * 2, 'k--', linewidth=.5)
    axs3.plot([dr[3, 0]] * 2, [-30, dr[3, 1]], [dr[3, 2]] * 2, 'k--', linewidth=.5)

    axs3.plot([-30, dr[0, 0]], [dr[0, 1]] * 2, [dr[0, 2]] * 2, 'k--', linewidth=.5)
    axs3.plot([-30, dr[1, 0]], [dr[1, 1]] * 2, [dr[1, 2]] * 2, 'k--', linewidth=.5)
    axs3.plot([-30, dr[2, 0]], [dr[2, 1]] * 2, [dr[2, 2]] * 2, 'k--', linewidth=.5)
    axs3.plot([-30, dr[3, 0]], [dr[3, 1]] * 2, [dr[3, 2]] * 2, 'k--', linewidth=.5)

    axs3.plot(dr[[0, 1], 0], dr[[0, 1], 1], dr[[0, 1], 2], 'k-')
    axs3.plot(dr[[1, 2], 0], dr[[1, 2], 1], dr[[1, 2], 2], 'k-')
    axs3.plot(dr[[2, 0], 0], dr[[2, 0], 1], dr[[2, 0], 2], 'k-')
    axs3.plot(dr[[0, 3], 0], dr[[0, 3], 1], dr[[0, 3], 2], 'k-')
    axs3.plot(dr[[1, 3], 0], dr[[1, 3], 1], dr[[1, 3], 2], 'k-')
    axs3.plot(dr[[2, 3], 0], dr[[2, 3], 1], dr[[2, 3], 2], 'k-')

    axs3.set_xlim([-30, 30])
    axs3.set_ylim([30, -30])
    axs3.set_zlim([-30, 30])
    axs3.set_xlabel("$\\Delta X$ [km]")
    axs3.set_ylabel("$\\Delta Y$ [km]")
    axs3.set_zlabel("$\\Delta Z$ [km]")

    axs3.legend(["MMS1", "MMS2", "MMS3", "MMS4"], frameon=False)

    axs = [axs0, axs1, axs2, axs3]

    return fig, axs
