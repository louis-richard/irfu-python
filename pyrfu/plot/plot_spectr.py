#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from .colorbar import colorbar as _add_colorbar


def plot_spectr(
    axis,
    inp,
    yscale: str = "linear",
    cscale: str = "linear",
    clim: list | None = None,
    cmap: str | None = None,
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
        out, and with a time-dependent table, the times without one.

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
        shading="auto". They were ignored before.

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
        cmap = mpl.colormaps.get_cmap(cmap)

    if isinstance(inp, xr.DataArray):
        x_data, y_data = [inp.coords[inp.dims[0]].data, inp.coords[inp.dims[1]].data]
        inp_data = inp.data.T
    elif isinstance(inp, xr.Dataset):
        depend_0_name = inp.data.dims[0]

        # The y-axis quantity (e.g. "energy") can be time-dependent, in
        # which case it is stored as a non-dimension coordinate on the
        # Dataset (dims: [time, idx]) rather than as a separate data
        # variable. Prefer a coordinate named "energy" if present, then
        # fall back to any other non-dimension coordinate along idx, and
        # finally to the legacy behaviour of looking for a second data
        # variable. A coordinate not along idx (e.g., the scalar coordinate
        # left by sel) was taken as y (IndexError)
        non_dim_coords = [
            name
            for name in inp.coords
            if name not in inp.dims and inp.data.dims[1] in inp[name].dims
        ]

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

    # pcolormesh takes finite coordinates only (NaN energies raised
    # ValueError): the y bins without a value, and with a time-dependent
    # table, the times without one, are left out
    if y_data.ndim == 1:
        keep = np.isfinite(y_data)
        y_data, inp_data = y_data[keep], inp_data[keep]
    else:
        keep = np.any(np.isfinite(y_data), axis=1)
        y_data, inp_data = y_data[keep], inp_data[keep]
        keep = np.all(np.isfinite(y_data), axis=0)
        x_data, y_data, inp_data = x_data[keep], y_data[:, keep], inp_data[:, keep]

    if cscale == "log":
        if clim is not None:
            options = {
                "norm": mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1]),
                "cmap": cmap,
            }
        elif np.any(np.isfinite(inp_data) & (inp_data > 0)):
            options = {"norm": mpl.colors.LogNorm(), "cmap": cmap}
        else:
            # Without positive data, LogNorm raised Invalid vmin or vmax
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

    image = axis.pcolormesh(
        x_data,
        y_data,
        inp_data,
        **{"rasterized": True, "shading": "auto", **options, **kwargs},
    )

    if np.issubdtype(x_data.dtype, np.datetime64):
        locator = mpl.dates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mpl.dates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    if yscale == "log":
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))

    axis.set_axisbelow(False)
    axis.set_ylim([np.min(y_data), np.max(y_data)])
    axis.set_aspect(aspect)

    if colorbar.lower() in ["right", "top"]:
        cax = _add_colorbar(image, axis, pad, 0.01, colorbar.lower())
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
