#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np
from geopack import geopack
from matplotlib.patches import Wedge

# Local imports
from ..pyrf import magnetosphere, datetime642unix, iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


def _add_earth(ax=None, **kwargs):
    theta1, theta2 = 90., 270.
    nightside_ = Wedge((0., 0.), 1., theta1, theta2, fc="k", ec="k", **kwargs)
    dayside_ = Wedge((0., 0.), 1., theta2, theta1, fc="w", ec="k", **kwargs)
    for wedge in [nightside_, dayside_]:
        ax.add_artist(wedge)
    return [nightside_, dayside_]


def _add_field_lines(ax, tint):
    # Get dipole axis at begin of the time interval
    ut = datetime642unix(iso86012datetime64(np.array(tint)))[0]
    _ = geopack.recalc(ut)

    x_lines_m, y_lines_m, z_lines_m = [[], [], []]
    x_lines_p, y_lines_p, z_lines_p = [[], [], []]

    xx_gsm, zz_gsm = np.meshgrid(np.linspace(-30, 6, 19),
                                 np.linspace(-5, 5, 10))
    xx_gsm = np.reshape(xx_gsm, (xx_gsm.size,))
    zz_gsm = np.reshape(zz_gsm, (zz_gsm.size,))

    for x_gsm, z_gsm in zip(xx_gsm, zz_gsm):
        y_gsm = 0
        _, _, _, xx, yy, zz = geopack.trace(x_gsm, y_gsm, z_gsm, dir=-1,
                                            rlim=100,
                                            r0=.99999, parmod=2, exname='t89',
                                            inname='igrf', maxloop=10000)
        x_lines_m.append(xx)
        z_lines_m.append(zz)

        _, _, _, xx, yy, zz = geopack.trace(x_gsm, y_gsm, z_gsm, dir=1,
                                            rlim=100,
                                            r0=.99999, parmod=2, exname='t89',
                                            inname='igrf', maxloop=10000)
        x_lines_p.append(xx)
        z_lines_p.append(zz)

    for xx, zz in zip(x_lines_m, z_lines_m):
        ax.plot(xx, zz, color="k", linewidth=.2)

    for xx, zz in zip(x_lines_p, z_lines_p):
        ax.plot(xx, zz, color="k", linewidth=.2)

    return


def plot_magnetosphere(ax, tint, colors: list = None,
                       field_lines: bool = True):
    r"""Plot magnetopause, bow shock and earth.

    Parameters
    ----------
    ax : matplotlib.pyplot.subplotsaxes
        Axis to plot.
    tint : list of str
        Time interval.
    colors : list, Optional
        Colors of the magnetopause and the bow show.
        Default use ["tab:blue", "tab:red"]

    Returns
    -------
    ax : matplotlib.pyplot.subplotsaxes
        Axis.

    """

    # Compute Magnetopause
    if colors is None:
        colors = ["tab:blue", "tab:green"]

    x_mp, y_mp = magnetosphere("mp_shue1998", tint)

    # Compute bow show
    x_bs, y_bs = magnetosphere("bs", tint)

    # Plot
    ax.plot(np.hstack([x_mp, np.flip(x_mp)]),
            np.hstack([y_mp, np.flip(-y_mp)]),
            color=colors[0], label="Magnetopause")
    ax.plot(np.hstack([x_bs, np.flip(x_bs)]),
            np.hstack([y_bs, np.flip(-y_bs)]),
            color=colors[1], label="Bow Shock")
    _add_earth(ax)

    if field_lines:
        logging.info("Computing field lines using T89 model...")
        _add_field_lines(ax, tint)

    return ax
