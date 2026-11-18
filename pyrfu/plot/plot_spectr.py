#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from collections.abc import Hashable
from typing import Literal, cast

# 3rd party imports
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Colormap

# Local imports
from .colorbar import colorbar as _add_colorbar


def _edges(centres, axis=0):
    r"""Edges of the cells around `centres` along `axis`: the midpoints, and
    half a step beyond the first and last centres (as the "nearest" shading
    of matplotlib's pcolormesh)."""

    half = np.diff(centres, axis=axis) / 2
    first = np.take(centres, [0], axis=axis) - np.take(half, [0], axis=axis)
    last = np.take(centres, [-1], axis=axis) + np.take(half, [-1], axis=axis)
    inner = np.take(centres, np.arange(centres.shape[axis] - 1), axis=axis) + half
    return np.concatenate([first, inner, last], axis=axis)


def _own_table_mesh(x_data, y_data, inp_data):
    r"""Mesh of a spectrogram whose y values change with time, `y_data` (n_y,
    n_t): each time drawn with the cell edges of its own y values. The times
    get their own left and right edges with a masked column of zero width
    between them (pcolormesh shares the corners of neighbouring columns,
    which would draw every time with the y edges averaged over its
    neighbours, i.e. alternating tables both at their mean)."""

    t_edges = _edges(x_data)
    y_edges = _edges(y_data, axis=0)

    x_mesh = np.broadcast_to(
        np.repeat(t_edges, 2)[1:-1], (len(y_edges), 2 * len(x_data))
    )
    y_mesh = np.repeat(y_edges, 2, axis=1)
    c_mesh = np.ma.masked_all((y_data.shape[0], 2 * len(x_data) - 1))
    c_mesh[:, ::2] = inp_data
    return x_mesh, y_mesh, c_mesh


def plot_spectr(
    axis,
    inp,
    yscale: Literal["linear", "log"] = "linear",
    cscale: Literal["linear", "log"] = "linear",
    clim: list | None = None,
    cmap: str | Colormap | None = None,
    colorbar: str = "right",
    aspect: str = "auto",
    pad: float = 0.01,
    **kwargs,
):
    r"""Plot a spectrogram using pcolormesh.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axis to plot. If None create a new figure.
    inp : xarray.DataArray or xarray.Dataset
        Input 2D data to plot. If a Dataset, it must contain a "data"
        data variable (dims: [time, idx]) and either:

        - a coordinate (e.g. "energy") along idx, with the same shape as
          "data" (dims: [time, idx]), or
        - a second data variable holding the y-axis values.

        The y-axis values that are not finite (e.g., NaN energies) are left
        out, and with a time-dependent table, the times without one. With a
        time-dependent table, each time is drawn with the cell edges of its
        own table (shading "flat"), except with one time or one channel
        (no spacing to define the edges: shading "auto").

    yscale : {"linear", "log"}, Optional
        Y-axis scaling. Default is "linear".
    cscale : {"linear", "log"} or matplotlib.colors.Normalize, Optional
        C-axis scaling, or the norm to use. Default is "linear". With "log"
        and no positive data, the colour limits are [1, 10] (with a warning).
    clim : array_like, Optional
        C-axis bounds [min, max] (list, tuple or array). Default is None (the
        range of the data).
    cmap : str or matplotlib.colors.Colormap, Optional
        Colormap. Default is matplotlib's default (``image.cmap``, set by the
        pyrfu style).
    colorbar : {"right", "top", "none"}, Optional
        Location of the colorbar with respect to the axis, "none" to hide it.
        Default is "right".
    aspect : str or float, Optional
        Aspect ratio of the axis, see matplotlib.axes.Axes.set_aspect.
        Default is "auto".
    pad : float, Optional
        Gap between the axis and the colorbar, in figure coordinates. Default
        is 0.01.
    **kwargs
        Keyword arguments forwarded to matplotlib.axes.Axes.pcolormesh (e.g.,
        alpha, edgecolors), which can also replace rasterized=True and
        shading="auto".

    Returns
    -------
    axis : matplotlib.axes.Axes
        Axis with the spectrogram.
    cax : matplotlib.axes.Axes
        Axis of the colorbar, only if colorbar is not "none" (otherwise only
        `axis` is returned).

    Raises
    ------
    TypeError
        If `inp` is not a DataArray or a Dataset.
    NotImplementedError
        If `colorbar` is not "right", "top" or "none".

    Warns
    -----
    UserWarning
        If `cscale` is "log", without `clim`, and no data is positive.

    """

    if axis is None:
        _, axis = plt.subplots(1)

    if not cmap or isinstance(cmap, str):
        # get_cmap(None) is the image.cmap colormap; the stub omits None
        cmap = mpl.colormaps.get_cmap(cmap)  # type: ignore[arg-type]

    if isinstance(inp, xr.DataArray):
        x_data, y_data = [inp.coords[inp.dims[0]].data, inp.coords[inp.dims[1]].data]
        inp_data = inp.data.T
    elif isinstance(inp, xr.Dataset):
        depend_0_name = inp.data.dims[0]

        # y values from a non-dimension coordinate along idx ("energy" first,
        # e.g., a time-dependent table), else from the other data variable;
        # coordinates not along idx (e.g., scalar ones left by sel) skipped
        non_dim_coords = [
            name
            for name in inp.coords
            if name not in inp.dims and inp.data.dims[1] in inp[name].dims
        ]

        depend_1_name: Hashable

        if "energy" in non_dim_coords:
            depend_1_name = "energy"
        elif non_dim_coords:
            depend_1_name = non_dim_coords[0]
        else:
            depend_1_name = list(
                filter(lambda x: x != "data", list(inp.data_vars)),
            )[0]

        x_data = inp[depend_0_name].data
        y_data = inp[depend_1_name].data.T
        inp_data = inp.data.data.T
    else:
        raise TypeError("inp must be an xarray.DataArray or xarray.Dataset")

    # pcolormesh needs finite coordinates: drop the y bins without a value
    # and, with a time-dependent table, the times without one
    if y_data.ndim == 1:
        keep = np.isfinite(y_data)
        y_data = y_data[keep]
        inp_data = inp_data[keep]
    else:
        keep = np.any(np.isfinite(y_data), axis=1)
        y_data = y_data[keep]
        inp_data = inp_data[keep]
        keep = np.all(np.isfinite(y_data), axis=0)
        x_data = x_data[keep]
        y_data = y_data[:, keep]
        inp_data = inp_data[:, keep]

    if cscale == "log":
        if clim is not None:
            options = {
                "norm": mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1]),
                "cmap": cmap,
            }
        elif np.any(np.isfinite(inp_data) & (inp_data > 0)):
            options = {"norm": mpl.colors.LogNorm(), "cmap": cmap}
        else:
            # Without positive data, LogNorm needs explicit limits
            warnings.warn(
                "no positive data for the log colour scale: limits set to [1, 10]",
                UserWarning,
            )
            options = {"norm": mpl.colors.LogNorm(vmin=1.0, vmax=10.0), "cmap": cmap}
    elif cscale == "linear":
        if clim is not None:
            options = {"cmap": cmap, "vmin": clim[0], "vmax": clim[1]}
        else:
            options = {"cmap": cmap, "vmin": None, "vmax": None}
    else:
        options = {"norm": cscale, "cmap": cmap}

    if y_data.ndim == 2 and min(y_data.shape) > 1:
        # Each time with the cell edges of its own table
        image = axis.pcolormesh(
            *_own_table_mesh(x_data, y_data, inp_data),
            **{"rasterized": True, **options, **kwargs, "shading": "flat"},
        )
    else:
        image = axis.pcolormesh(
            x_data,
            y_data,
            inp_data,
            **{"rasterized": True, "shading": "auto", **options, **kwargs},
        )

    if np.issubdtype(x_data.dtype, np.datetime64):
        locator: mpl.ticker.Locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    if yscale == "log":
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))

    axis.set_axisbelow(False)
    axis.set_ylim([np.min(y_data), np.max(y_data)])
    axis.set_aspect(aspect)

    if colorbar.lower() in ["right", "top"]:
        where = cast(Literal["right", "top"], colorbar.lower())
        cax = _add_colorbar(image, axis, pad, 0.01, where)
        cax.set_axisbelow(False)

        if cscale == "log":
            locator = mpl.ticker.LogLocator(base=10.0, numticks=4)
        else:
            locator = mpl.ticker.MaxNLocator(4)

        (cax.yaxis if colorbar.lower() == "right" else cax.xaxis).set_major_locator(
            locator
        )

        out = (axis, cax)
    elif colorbar.lower() == "none":
        out = axis
    else:
        raise NotImplementedError("colorbar must be: 'right', 'top', or 'none'")

    return out
