#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np
from matplotlib.patches import Wedge

# Local imports
from ..pyrf import datetime642unix, iso86012datetime64, magnetosphere

logger = logging.getLogger(__name__)


def _add_earth(ax=None, **kwargs):
    # The nightside (black) and the dayside (white) halves
    wedges = [
        Wedge((0.0, 0.0), 1.0, theta1, theta2, fc=color, ec="k", **kwargs)
        for theta1, theta2, color in [(90.0, 270.0, "k"), (270.0, 90.0, "w")]
    ]

    for wedge in wedges:
        ax.add_artist(wedge)

    return wedges


def _add_field_lines(ax, tint):
    # geopack prints and requests the latest IGRF coefficients online when
    # imported, so it is only imported when the field lines are needed
    from geopack import geopack  # pylint: disable=import-outside-toplevel

    # Get dipole axis at begin of the time interval
    ut = datetime642unix(iso86012datetime64(np.array(tint)))[0]
    _ = geopack.recalc(ut)

    xx_gsm, zz_gsm = np.meshgrid(
        np.linspace(-30, 6, 19),
        np.linspace(-5, 5, 10),
    )

    # The field lines traced antiparallel, then parallel, to B from each point
    # of the grid in the y = 0 plane
    for direction in [-1, 1]:
        for x_gsm, z_gsm in zip(xx_gsm.ravel(), zz_gsm.ravel()):
            _, _, _, xx, _, zz = geopack.trace(
                x_gsm,
                0,
                z_gsm,
                dir=direction,
                rlim=100,
                r0=0.99999,
                parmod=2,
                exname="t89",
                inname="igrf",
                maxloop=10000,
            )
            ax.plot(xx, zz, color="k", linewidth=0.2)


def plot_magnetosphere(
    ax,
    tint,
    colors: list = None,
    field_lines: bool = True,
):
    r"""Plot magnetopause, bow shock and earth.

    Parameters
    ----------
    ax : matplotlib.pyplot.subplotsaxes
        Axis to plot.
    tint : list of str
        Time interval.
    colors : list, Optional
        Colors of the magnetopause and the bow shock. Default is
        ["tab:blue", "tab:green"].
    field_lines : bool, Optional
        Plot the T89 magnetic field lines (needs geopack). Default is True.

    Returns
    -------
    ax : matplotlib.pyplot.subplotsaxes
        Axis.

    """

    if colors is None:
        colors = ["tab:blue", "tab:green"]

    # Magnetopause and bow shock, each with its mirror image (y -> -y) as one
    # closed line
    for model, color, label in [
        ("mp_shue1998", colors[0], "Magnetopause"),
        ("bs", colors[1], "Bow Shock"),
    ]:
        x_model, y_model = magnetosphere(model, tint)
        ax.plot(
            np.hstack([x_model, np.flip(x_model)]),
            np.hstack([y_model, np.flip(-y_model)]),
            color=color,
            label=label,
        )

    _add_earth(ax)

    if field_lines:
        logger.info("Computing field lines using T89 model...")
        _add_field_lines(ax, tint)

    return ax
