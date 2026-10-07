#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from matplotlib.axes import Axes
from matplotlib.dates import num2date
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from xarray.core.dataarray import DataArray

# Local imports
from ..constants import R_E

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def add_position(
    ax: Axes,
    r_xyz: DataArray,
    spine: float = 20,
    position: str = "top",
    fontsize: float = 10,
    units: str = "re",
) -> Axes:
    r"""Add extra axes to plot spacecraft position, as X, Y, Z and the distance
    R, with a label on the left. The position is interpolated linearly at the
    ticks of `ax`, and the ticks outside the time series are left without
    label.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axis where to label the spacecraft position.
    r_xyz : xarray.DataArray
        Time series of the spacecraft position.
    spine : float, Optional
        Relative position of the axes. Default is 20.
    position : str, Optional
        Axis position wtr to the reference axis. Default is "top".
    fontsize : float, Optional
        xticks label font size. Default is 10.
    units : {"re", "km"}, Optional
        Units of the labels, Earth radii (:data:`pyrfu.constants.R_E`) or km
        (the units of `r_xyz`). Default is "re".

    Returns
    -------
    axr : matplotlib.axes._axes.Axes
        Twin axis with spacecraft position as x-axis label.

    Raises
    ------
    ValueError
        If `units` is not "re" or "km".

    """

    if units.lower() == "re":
        r_unit, unit_label = [R_E, "$R_E$"]
    elif units.lower() == "km":
        r_unit, unit_label = [1.0, "km"]
    else:
        raise ValueError(f"units must be 're' or 'km', not {units!r}")

    x_lim = ax.get_xlim()

    t_ticks = [t_.replace(tzinfo=None) for t_ in num2date(ax.get_xticks())]
    t_ticks = np.array(t_ticks).astype("<M8[ns]")

    # Position interpolated at the ticks (seconds since the first sample), NaN
    # outside the time series
    t_data = r_xyz.time.data.astype("<M8[ns]")
    s_data = (t_data - t_data[0]) / np.timedelta64(1, "s")
    s_ticks = (t_ticks - t_data[0]) / np.timedelta64(1, "s")
    r_ticks = np.stack(
        [
            np.interp(s_ticks, s_data, r_xyz.data[:, i], left=np.nan, right=np.nan)
            for i in range(3)
        ],
        axis=1,
    )
    r_ticks = np.hstack([r_ticks, np.linalg.norm(r_ticks, axis=1, keepdims=True)])
    r_ticks /= r_unit

    ticks_labels = []
    for ticks_ in r_ticks:
        if np.any(np.isnan(ticks_)):
            ticks_labels.append("")
        else:
            ticks_labels.append("\n".join(f"{r_:3.2f}" for r_ in ticks_))

    axr = ax.twiny()
    axr.spines[position].set_position(("outward", spine))
    axr.xaxis.set_ticks_position(position)
    axr.xaxis.set_label_position(position)
    axr.set_xticks(t_ticks)
    axr.set_xticklabels(ticks_labels, fontsize=fontsize)
    axr.set_xlim(x_lim)

    # Names of the label rows, 4 points left of the axis, at the height of the
    # tick labels
    tick = axr.xaxis.get_major_ticks()[0]
    pad = tick.get_pad() + tick.get_tick_padding()
    if position == "top":
        y_transform, v_align, _ = axr.get_xaxis_text2_transform(pad)
        y_label = 1
    else:
        y_transform, v_align, _ = axr.get_xaxis_text1_transform(pad)
        y_label = 0

    x_shift = ScaledTranslation(-4 / 72, 0, axr.figure.dpi_scale_trans)
    axr.text(
        0,
        y_label,
        "\n".join(f"{c_} [{unit_label}]" for c_ in "XYZR"),
        transform=blended_transform_factory(axr.transAxes + x_shift, y_transform),
        verticalalignment=v_align,
        horizontalalignment="right",
        fontsize=fontsize,
    )

    return axr
