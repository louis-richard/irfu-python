#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from matplotlib import colors

# Local imports
from .colorbar import colorbar as _add_colorbar

# Scale of the velocities (km/s) for each unit of the axes
_V_SCALES = {"km/s": (1.0, "km s$^{-1}$"), "10^3 km/s": (1e3, "10$^3$ km s$^{-1}$")}


def _contour_levels(contour, clim):
    r"""Contour levels: one per decade of `clim` (None or True), `contour` levels
    spaced logarithmically over `clim` (an int), the levels given, or none
    (False)."""

    clim_log = np.log10(clim)

    if contour is None or contour is True:
        n_levs = int(np.round(clim_log[1]) - np.round(clim_log[0])) + 1
        return np.logspace(clim_log[0], clim_log[1], n_levs)

    if contour is False:
        return None

    if isinstance(contour, (int, np.integer)) and not isinstance(contour, bool):
        if contour < 1:
            raise ValueError("contour must be a positive number of levels")

        return np.logspace(clim_log[0], clim_log[1], contour)

    return np.sort(np.atleast_1d(np.asarray(contour, dtype=np.float64)))


# Length (points) of the arrow of a vector in the plane
_ARROW_LENGTH = 60.0


def _plot_vectors(ax, vectors):
    r"""Arrows from the origin along the in-plane parts of the unit vectors,
    of _ARROW_LENGTH points for a vector in the plane (whatever the limits),
    labelled at their tips."""

    for vec, label in vectors:
        vec = np.asarray(vec, dtype=np.float64)
        in_plane = _ARROW_LENGTH * vec[:2] / np.linalg.norm(vec)
        ax.annotate(
            label,
            xy=(0.0, 0.0),
            xytext=tuple(in_plane),
            textcoords="offset points",
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "<-", "color": "k", "linewidth": 1.5},
        )


def plot_reduced_2d(
    ax,
    f2d,
    clim: list | None = None,
    *,
    contour=None,
    vectors: list | None = None,
    v_units: str = "km/s",
):
    r"""Plot the 2D reduced distribution `f2d` onto the axis `ax`.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis.
    f2d : xarray.DataArray
        2D reduced distribution at one time, on a cartesian grid (vx, vy)
        or a polar one (phi, v), with the coordinates ``vx`` and ``vy`` of
        the centres of the bins (km/s) and the edges ``phi_edges`` (deg) and
        ``v_edges`` in the attributes (e.g. a time of `pyrfu.mms.reduce`
        with ``base="pol"``).
    clim : list of floats, Optional
        Coloraxis limits. Default is [1e-13, 1e-8].
    contour : int or array_like or bool, Optional
        Contour lines: one per decade of `clim` (None, default, or True), a number
        of levels spaced logarithmically over `clim` (an int), the levels
        (array_like), or none (False).
    vectors : list of tuple, Optional
        Directions to draw, as (vector, label) pairs, e.g.
        ``[(b_xyz, "B"), (v_xyz, "V")]``, the vectors in the frame of the
        reduction (x and y the axes of the plane, z out of it): an arrow from
        the origin along the part in the plane of the unit vector, 60 points
        long for a vector in the plane (shorter out of it, whatever the
        limits), with its label at the tip.
    v_units : {"km/s", "10^3 km/s"}, Optional
        Units of the velocity axes. Default is "km/s" ("10^3 km/s" suits the
        electrons).

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis.
    cax : matplotlib.axes._axes.Axes
        Colorbar axis.

    Raises
    ------
    ValueError
        If `v_units` is not one of the above or `contour` is a number below 1.

    """

    if clim is None:
        clim = [1e-13, 1e-8]

    if v_units not in _V_SCALES:
        raise ValueError(f'v_units must be "km/s" or "10^3 km/s", not {v_units!r}')

    scale, v_label = _V_SCALES[v_units]
    levels = _contour_levels(contour, clim)

    if "phi" in f2d.dims:
        # Polar grid: the bins drawn between their edges, the contours on
        # their centres, the first azimuth repeated to close them
        phi_e, v_e = np.meshgrid(
            np.deg2rad(f2d.attrs["phi_edges"]), f2d.attrs["v_edges"] / scale
        )
        mesh = (v_e * np.cos(phi_e), v_e * np.sin(phi_e), np.transpose(f2d.data))
        shading = "flat"
        closed = [
            np.vstack([x, x[:1]]).T
            for x in [f2d.vx.data / scale, f2d.vy.data / scale, f2d.data]
        ]
        vx_lim = vy_lim = np.max(np.abs(f2d.attrs["v_edges"])) / scale
    else:
        mesh = (f2d.vx.data / scale, f2d.vy.data / scale, np.transpose(f2d.data))
        shading = "auto"
        closed = list(mesh)
        vx_lim = np.max(abs(f2d.vx.data)) / scale
        vy_lim = np.max(abs(f2d.vy.data)) / scale

    im = ax.pcolormesh(
        *mesh,
        norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
        cmap="Spectral_r",
        rasterized=True,
        shading=shading,
    )

    ax.set_xlim(np.array([-vx_lim, vx_lim]))
    ax.set_ylim(np.array([-vy_lim, vy_lim]))
    ax.set_aspect("equal")
    ax.set_xlabel(f"$v_x$ [{v_label}]")
    ax.set_ylabel(f"$v_y$ [{v_label}]")

    cax = _add_colorbar(im, ax, width=0.01, where="top")

    if levels is not None:
        c_lines = ax.contour(
            *closed,
            levels=levels,
            norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
            cmap="viridis",
        )
        im.colorbar.add_lines(c_lines)

    if vectors:
        _plot_vectors(ax, vectors)

    return ax, cax
