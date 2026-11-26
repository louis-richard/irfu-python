#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from matplotlib import colors

# Local imports
from .colorbar import colorbar as _add_colorbar


def plot_reduced_2d(ax, f2d, clim: list | None = None):
    r"""Plot the 2D reduced distribution `f2d` onto the axis `ax`.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis.
    f2d : xarray.DataArray
        2D reduced distribution at one time, on a cartesian grid (vx, vy)
        or a polar one (phi, v), with the coordinates ``vx`` and ``vy`` of
        the centres of the bins and the edges ``phi_edges`` (deg) and
        ``v_edges`` in the attributes (e.g. a time of `pyrfu.mms.reduce`
        with ``base="pol"``).
    clim : list of floats, Optional
        Coloraxis limits.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis.
    cax : matplotlib.axes._axes.Axes
        Colorbar axis.

    """

    if clim is None:
        clim = [1e-13, 1e-8]

    clim_log = np.log10(clim)
    n_levs = int(np.round(clim_log[1]) - np.round(clim_log[0])) + 1

    if "phi" in f2d.dims:
        # Polar grid: the bins drawn between their edges, the contours on
        # their centres, the first azimuth repeated to close them
        phi_e, v_e = np.meshgrid(
            np.deg2rad(f2d.attrs["phi_edges"]), f2d.attrs["v_edges"]
        )
        mesh = (v_e * np.cos(phi_e), v_e * np.sin(phi_e), np.transpose(f2d.data))
        shading = "flat"
        closed = [np.vstack([x, x[:1]]).T for x in [f2d.vx.data, f2d.vy.data, f2d.data]]
        vx_lim = vy_lim = np.max(np.abs(f2d.attrs["v_edges"]))
    else:
        mesh = (f2d.vx.data, f2d.vy.data, np.transpose(f2d.data))
        shading = "auto"
        closed = list(mesh)
        vx_lim = np.max(abs(f2d.vx.data))
        vy_lim = np.max(abs(f2d.vy.data))

    im = ax.pcolormesh(
        *mesh,
        norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
        cmap="Spectral_r",
        rasterized=True,
        shading=shading,
    )

    c_lines = ax.contour(
        *closed,
        levels=np.logspace(clim_log[0], clim_log[1], n_levs),
        norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
        cmap="viridis",
    )

    ax.set_xlim(np.array([-vx_lim, vx_lim]))
    ax.set_ylim(np.array([-vy_lim, vy_lim]))
    ax.set_aspect("equal")

    cax = _add_colorbar(im, ax, width=0.01, where="top")
    im.colorbar.add_lines(c_lines)

    return ax, cax
