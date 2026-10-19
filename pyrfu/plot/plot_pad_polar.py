#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
from matplotlib import colors

# Local imports
from ..mms._vdf_bins import _pitch_angles, _polar_edges, _sc_pot, _speed_edges
from ..mms._vdf_mc import _bins_d3v, _rng, _sample
from .plot_ang_ang import _check_units

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"

# Largest angular width (degrees) of the cells drawn, so that the pitch-angle
# bins are drawn as arcs rather than chords
_MAX_ARC = 2.0

# Largest mismatch of the edges of neighbouring bins, relative to their width,
# that is joined rather than drawn as a gap (the FPI half widths are rounded:
# the channels overlap or miss each other by up to 0.1 %)
_JOIN = 0.01


def _time_index(pad, tint):
    r"""Indices of the time closest to tint (one element), of the times in
    the interval tint (two elements), or of all the times (None)."""
    time = pad.time.data

    if tint is None:
        return np.arange(len(time))

    if len(tint) == 1:
        return np.array([np.argmin(np.abs(time - np.datetime64(tint[0], "ns")))])

    if len(tint) == 2:
        t_start, t_stop = (np.datetime64(t, "ns") for t in tint)
        index = np.where((time >= t_start) & (time <= t_stop))[0]

        if not index.size:
            raise ValueError("No data in the time interval")

        return index

    raise ValueError("tint must have one or two elements")


def _cells(lower, upper):
    r"""Edges of the cells drawn and the bin of each cell (-1 in the gaps
    between bins). Neighbouring bins whose edges differ by less than _JOIN of
    their width share the midpoint; cells covered by overlapping bins go to
    the bin with the closest centre."""
    lower, upper = (np.array(edge, dtype=np.float64) for edge in [lower, upper])
    order = np.argsort(lower + upper)
    low, high = lower[order], upper[order]
    width = np.minimum((high - low)[:-1], (high - low)[1:])
    join = np.abs(low[1:] - high[:-1]) <= _JOIN * width
    middle = (low[1:] + high[:-1]) / 2
    high[:-1][join] = middle[join]
    low[1:][join] = middle[join]
    lower[order], upper[order] = low, high

    edges = np.unique(np.hstack([lower, upper]))
    # Merge the edges that only differ by round-off
    tol = 1e-9 * np.max(np.abs(edges))
    edges = edges[np.hstack([True, np.diff(edges) > tol])]

    middle = (edges[:-1] + edges[1:]) / 2
    inside = (lower[:, None] <= middle) & (middle <= upper[:, None])
    distance = np.abs((lower + upper)[:, None] / 2 - middle)
    distance[~inside] = np.inf
    index = np.argmin(distance, axis=0)
    index[~np.any(inside, axis=0)] = -1
    return edges, index


def _arcs(edges, index):
    r"""Splits the angular cells into cells of at most _MAX_ARC degrees."""
    n_sub = np.maximum(np.ceil(np.diff(edges) / _MAX_ARC).astype(int), 1)
    sub_edges = [
        np.linspace(e_0, e_1, n, endpoint=False)
        for e_0, e_1, n in zip(edges[:-1], edges[1:], n_sub)
    ]
    return np.hstack([*sub_edges, edges[-1]]), np.repeat(index, n_sub)


def _speed_units(pad):
    r"""Speed units of the axes (10^3 km/s for electrons, else km/s) and
    their value in m/s."""
    if str(pad.attrs["species"]).lower() in ["electrons", "electron", "e"]:
        return "10$^3$ km/s", 1e6

    return "km/s", 1e3


def _binned(pad, sc_pot, scale):
    r"""Polar cells of the bins: speeds (time-averaged edges, axis units)
    and pitch angles (degrees), with the time-averaged distribution."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.nanmean(np.asarray(pad.data.data, dtype=np.float64), axis=0)

    data[data <= 0] = np.nan

    v_lower, v_upper = (
        np.mean(edge, axis=0) / scale for edge in _speed_edges(pad, sc_pot, None)
    )
    a_lower, a_upper = _polar_edges(_pitch_angles(pad), pad.attrs, "pitchangle")

    r_edges, r_index = _cells(v_lower, v_upper)
    a_edges, a_index = _arcs(*_cells(a_lower, a_upper))

    cells = data[r_index[:, None], a_index[None, :]]
    cells[r_index < 0, :] = np.nan
    cells[:, a_index < 0] = np.nan

    r_grid, a_grid = np.meshgrid(r_edges, np.deg2rad(a_edges), indexing="ij")
    mesh = (r_grid * np.cos(a_grid), r_grid * np.sin(a_grid), cells)
    return mesh, data, (v_lower, v_upper), (a_lower, a_upper)


def _monte_carlo(pad, sc_pot, scale, n_mc, n_grid, random_state):
    r"""Distribution on a Cartesian (v_par, v_perp) grid from n_mc particles
    per bin and per time, drawn uniformly in the volume of the bin (each time
    with its own energy edges), and averaged over time as numpy.nanmean."""
    data = np.asarray(pad.data.data, dtype=np.float64)
    valid = np.isfinite(data)
    n_valid = np.sum(valid, axis=0)
    data = np.where(valid & (data > 0), data, 0.0)

    v_lower, v_upper = _speed_edges(pad, sc_pot, None)
    a_lower, a_upper = _polar_edges(_pitch_angles(pad), pad.attrs, "pitchangle")
    # Bins of a skymap with B along z and one gyrophase bin
    n_t = len(pad.time)
    bins = {
        "v_lower": v_lower,
        "v_upper": v_upper,
        "phi_lower": np.zeros((n_t, 1)),
        "phi_upper": np.full((n_t, 1), 360.0),
        "cos_upper": np.cos(np.deg2rad(a_lower)),
        "cos_lower": np.cos(np.deg2rad(a_upper)),
    }
    f_d3v = data * _bins_d3v(bins)[:, :, 0, :]

    i_t, i_e, i_q = (np.repeat(i, n_mc) for i in np.nonzero(f_d3v))
    # _sample gives the velocities opposite to the look directions
    velocity = -_sample(_rng(random_state), bins, i_t, i_e, np.zeros_like(i_t), i_q)
    weights = f_d3v[i_t, i_e, i_q] / (n_mc * n_valid[i_e, i_q])

    v_max = np.max(v_upper)
    par_edges = np.linspace(-v_max, v_max, n_grid + 1)
    perp_edges = np.linspace(0.0, v_max, n_grid // 2 + 1)
    f_d3v, _, _ = np.histogram2d(
        velocity[:, 2],
        np.hypot(velocity[:, 0], velocity[:, 1]),
        [par_edges, perp_edges],
        weights=weights,
    )
    # Volume of the rings (v_par, v_perp) around B
    volume = np.outer(np.diff(par_edges), np.pi * np.diff(perp_edges**2))
    data = f_d3v / volume
    data[data <= 0] = np.nan

    par_grid, perp_grid = np.meshgrid(par_edges, perp_edges, indexing="ij")
    return par_grid / scale, perp_grid / scale, data


def _contours(data, radii, angles):
    r"""Contours of log10(data) at the centres of the bins, extended to the
    pitch angles 0 and 180 degrees when the bins reach them."""
    (r_lower, r_upper), (a_lower, a_upper) = radii, angles
    drawn = r_upper > r_lower
    values = np.log10(data[drawn])
    pitch_angles = (a_lower + a_upper) / 2

    if np.min(a_lower) <= 0.0:
        pitch_angles = np.hstack([0.0, pitch_angles])
        values = np.hstack([values[:, :1], values])

    if np.max(a_upper) >= 180.0:
        pitch_angles = np.hstack([pitch_angles, 180.0])
        values = np.hstack([values, values[:, -1:]])

    r_grid, a_grid = np.meshgrid(
        ((r_lower + r_upper) / 2)[drawn], np.deg2rad(pitch_angles), indexing="ij"
    )
    return r_grid * np.cos(a_grid), r_grid * np.sin(a_grid), values


def _colorbar(ax, image, position):
    r"""Colorbar at the right or on top of ax, or none."""
    fig = ax.get_figure()
    pos = ax.get_position()

    if position == "right":
        cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.01, pos.height])
        fig.colorbar(mappable=image, cax=cax)
    elif position == "top":
        cax = fig.add_axes([pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.01])
        fig.colorbar(mappable=image, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    else:
        cax = None

    return cax


def plot_pad_polar(
    ax,
    pad,
    tint: list = None,
    sc_pot=None,
    clim: list = None,
    cmap: str = None,
    contour=None,
    colorbar: str = "right",
    monte_carlo: bool = False,
    n_mc: int = 200,
    n_grid: int = 100,
    random_state=None,
):
    r"""Plots a pitch-angle distribution in the (v_par, v_perp) half plane, as
    irfu-matlab PDist.plot_pad_polar: the bin of speed v and pitch angle
    alpha is at v_par = v cos(alpha) and v_perp = v sin(alpha) =
    sqrt(v_perp1^2 + v_perp2^2).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot.
    pad : xarray.Dataset
        Pitch-angle distribution (time, energy, pitch angle), see
        `pyrfu.mms.get_pitch_angle_dist`.
    tint : list of str, Optional
        Time interval: the distribution at the closest time (one element) or
        averaged over the interval (two elements). Default averages all the
        times of `pad`.
    sc_pot : xarray.DataArray or float or array_like, Optional
        Spacecraft potential (V): the speeds are those at infinity (electrons
        lose and ions gain q V), and the channels below the potential are not
        drawn. Default is no correction.
    clim : list of float, Optional
        Colour limits, in the units of `pad`. Default is the data range.
    cmap : str, Optional
        Colormap. Default is the matplotlib default.
    contour : int or array_like, Optional
        Number of contour levels (equally spaced in log10) or the levels, in
        the units of `pad`, drawn in black. Default is no contours.
    colorbar : {"right", "top", "none"}, Optional
        Position of the colorbar. Default is "right".
    monte_carlo : bool, Optional
        Resample the distribution on a Cartesian (v_par, v_perp) grid with
        Monte-Carlo particles instead of drawing the bins. Default is False.
    n_mc : int, Optional
        With `monte_carlo`, number of particles per bin and per time. Default
        is 200, as `pyrfu.mms.vdf_rebin`.
    n_grid : int, Optional
        With `monte_carlo`, number of cells of the grid along v_par, half as
        many along v_perp. Default is 100.
    random_state : int or numpy.random.Generator, Optional
        With `monte_carlo`, seed or generator of the particles.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis.
    cax : matplotlib.axes.Axes
        Colorbar axis, None if `colorbar` is "none".

    Raises
    ------
    ValueError
        If `colorbar` is invalid, `tint` does not have one or two elements,
        or there is no data in `tint`.

    See Also
    --------
    pyrfu.mms.get_pitch_angle_dist, pyrfu.plot.plot_reduced_2d

    Notes
    -----
    The colour is the phase-space density averaged over gyrophase in each
    bin (`pyrfu.mms.get_pitch_angle_dist`), not a reduced distribution (the
    integral of f over v_perp dphi, see `pyrfu.mms.reduce`). The speeds are
    in km/s, in 10^3 km/s for electrons.

    By default the bins are drawn from their edges: the energy edges from
    the ``delta_energy_minus`` and ``delta_energy_plus`` attributes
    (midpoints between the channels without them), averaged over time after
    the spacecraft potential correction, as irfu-matlab, and the pitch-angle
    edges from ``delta_pitchangle_minus`` and ``delta_pitchangle_plus``. Gaps
    between the bins are left blank, except mismatches of less than 1 % of
    the bin width, which are joined (the FPI half widths are rounded, so
    that neighbouring channels overlap or miss each other by up to 0.1 %).
    The pitch-angle bins are drawn as arcs. Zeros are not drawn.

    With `monte_carlo`, each bin with data at each time is sampled by `n_mc`
    particles uniform in its volume (v^3 and cos(alpha), as
    `pyrfu.mms.vdf_rebin`), carrying f d3v / `n_mc`; their sum in each cell
    of the grid divided by its volume (the ring pi (v_perp2^2 - v_perp1^2)
    dv_par) is the distribution, averaged over time as the bins. The density
    is conserved, and each time keeps its own energy edges, so that FPI
    burst distributions with alternating energy tables can be averaged. The
    result has Monte-Carlo noise, largest in the small cells near
    v_perp = 0.

    irfu-matlab draws v_par along y, takes the speeds of all species with
    the electron mass and draws the contours at the energies of the channels
    rather than at their speeds.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pyrfu import mms, plot
    >>> tint = ["2017-07-11T22:34:01.000", "2017-07-11T22:34:03.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> b_xyz = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> pad_e = mms.get_pitch_angle_dist(vdf_e, b_xyz, tint, angles=18)

    Electron pitch-angle distribution at 22:34:02, and resampled on a grid

    >>> f, axs = plt.subplots(2)
    >>> axs[0], caxs0 = plot.plot_pad_polar(axs[0], pad_e, [tint[0]])
    >>> axs[1], caxs1 = plot.plot_pad_polar(axs[1], pad_e, monte_carlo=True)

    """
    if colorbar not in ["right", "top", "none"]:
        raise ValueError(f"Invalid colorbar {colorbar}: right, top or none")

    index = _time_index(pad, tint)
    sc_pot = _sc_pot(pad, sc_pot)[index]
    pad = pad.isel(time=index)
    units, scale = _speed_units(pad)

    if monte_carlo:
        mesh = _monte_carlo(pad, sc_pot, scale, n_mc, n_grid, random_state)
        data = mesh[2]
        centre = [(edge[1:, 1:] + edge[:-1, :-1]) / 2 for edge in mesh[:2]]
        lines = (*centre, np.log10(data))
    else:
        mesh, data, radii, angles = _binned(pad, sc_pot, scale)
        lines = _contours(data, radii, angles)

    if clim is None:
        clim = [np.nanmin(data), np.nanmax(data)]

    image = ax.pcolormesh(
        *mesh,
        norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]),
        cmap=cmap,
        shading="flat",
        rasterized=True,
    )

    if contour is not None:
        ax.contour(
            *lines,
            levels=contour if np.ndim(contour) == 0 else np.log10(contour),
            colors="k",
            linewidths=0.8,
            linestyles="solid",
        )

    v_max = np.nanmax(mesh[0])
    ax.set_xlim([-v_max, v_max])
    ax.set_ylim([0.0, v_max])
    ax.set_aspect("equal")
    ax.set_xlabel(f"$v_\\parallel$ [{units}]")
    ax.set_ylabel(f"$v_\\perp$ [{units}]")

    cax = _colorbar(ax, image, colorbar)

    if cax is not None:
        label = _check_units(pad)

        if colorbar == "right":
            cax.set_ylabel(label)
        else:
            cax.set_xlabel(label)

    return ax, cax
