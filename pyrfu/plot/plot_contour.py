#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib as mpl
import xarray as xr

__all__ = ["plot_contour"]


def plot_contour(axis, inp, **kwargs):
    r"""Plot a contour plot.

    Parameters
    ----------
    axis : matplotlib.pyplot.subplotsaxes
        Target axis to plot.
    inp : xarray.DataArray
        Input 2D data to plot.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments for matplotlib.pyplot.contour.

    Returns
    -------
    axis : matplotlib.pyplot.subplotsaxes
        Axis with contour plot.
    clines : matplotlib.contour.QuadContourSet
        Contour lines.

    """

    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError("ax must be a matplotlib.pyplot.subplotsaxes.")

    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a xarray.DataArray.")

    # Get dimensions of the input data
    dims = inp.dims
    x, y = [inp[dim].data for dim in dims]

    # Plot contour
    clines = axis.contour(x, y, inp.data.T, **kwargs)

    return axis, clines
