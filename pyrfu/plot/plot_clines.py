#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party import
import matplotlib as mpl
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

# Local imports
from .colorbar import colorbar as _add_colorbar
from .plot_line import plot_line


def plot_clines(axis, inp, yscale="log", cscale="log", cmap="jet", **kwargs):
    r"""Plot lines with color associated to the level.

    Parameters
    ----------
    axis :
        Axes
    inp : xarray.DataArray
        Time series as an energy spectrum to plot.
    yscale : str, Optional
        Scale of the yaxis. Default is "log"
    cscale : str, Optional
        Scale of the colormap. Default is "log".
    cmap : str or matplotlib.colors.Colormap, Optional
        Colormap. Default is "jet"
    kwargs : dict
        Plot options.

    Returns
    -------
    axis :
        Updated axis
    cax :
        Axis of the associated colorbar

    Raises
    ------
    NotImplementedError
        If cscale is not "log".
    ValueError
        If no energy is positive.

    Warns
    -----
    UserWarning
        If lines have a 0 or NaN energy (not drawn).

    Other Parameters
    ----------------
    See pyrfu.plot.plot_line

    """

    pad = 0.01
    c_map = mpl.colormaps.get_cmap(cmap)
    energy = inp.energy.data

    if cscale != "log":
        raise NotImplementedError

    # The energies on the log colour scale: a 0 or NaN energy raised Invalid
    # vmin or vmax
    on_scale = np.isfinite(energy) & (energy > 0)

    if not np.any(on_scale):
        raise ValueError("no positive energy for the log colour scale")

    if not np.all(on_scale):
        warnings.warn(
            f"{np.sum(~on_scale)} line(s) without a positive energy not drawn",
            UserWarning,
        )

    norm = LogNorm(vmin=np.min(energy[on_scale]), vmax=np.max(energy[on_scale]))

    # Color of each line from its energy, as on the colorbar (was from the index)
    for i in np.where(on_scale)[0]:
        plot_line(axis, inp[:, i], color=c_map(norm(energy[i])), **kwargs)

    cax = _add_colorbar(ScalarMappable(norm=norm, cmap=c_map), axis, pad, 0.01)

    if yscale == "log":
        axis.set_yscale("log")

    cax.set_axisbelow(False)

    return axis, cax
