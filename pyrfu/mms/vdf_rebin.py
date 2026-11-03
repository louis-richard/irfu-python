#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ._psd_units import _attrs
from ._vdf_bins import _is_skymap
from ._vdf_mc import _bins_d3v, _rng, _sample, _source_bins
from .vdf_convert import _PSD, vdf_convert
from .vdf_shift import _shift
from .vdf_velocity import _rotation_matrices

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.1"
__status__ = "Prototype"


def _cartesian(vdf, grid, rotation, n_mc, sc_pot, random_state):
    # Monte-Carlo rebinning onto the (vx, vy, vz) boxes of grid (km/s)
    edges = [np.asarray(e, dtype=np.float64) * 1e3 for e in grid]
    shape = tuple(len(e) - 1 for e in edges)
    volume = np.einsum("i,j,k->ijk", *(np.diff(e) for e in edges))

    f_si = np.nan_to_num(vdf_convert(vdf, "s^3/m^6").data.data)
    bins = _source_bins(vdf, sc_pot)
    d3v = _bins_d3v(bins)
    rotations = _rotation_matrices(rotation, len(vdf.time))
    rng = _rng(random_state)

    out = np.zeros((len(vdf.time),) + shape)
    lost = np.zeros(len(vdf.time))

    for i_t in range(len(vdf.time)):
        f_d3v = f_si[i_t] * d3v[i_t]
        i_e, i_p, i_q = np.nonzero(f_d3v)
        i_e, i_p, i_q = (np.repeat(i, n_mc) for i in [i_e, i_p, i_q])
        velocity = _sample(rng, bins, np.full(len(i_e), i_t), i_e, i_p, i_q)
        velocity = velocity @ rotations[i_t].T
        weights = f_d3v[i_e, i_p, i_q] / n_mc

        index = [
            np.searchsorted(e, velocity[:, i], side="right") - 1
            for i, e in enumerate(edges)
        ]
        inside = np.all([(i >= 0) & (i < n) for i, n in zip(index, shape)], axis=0)
        flat = np.ravel_multi_index([i[inside] for i in index], shape)
        summed = np.bincount(flat, weights[inside], minlength=int(np.prod(shape)))
        out[i_t] = summed.reshape(shape) / volume

        total = np.sum(f_d3v)
        lost[i_t] = np.sum(weights[~inside]) / total if total > 0 else 0.0

    return out, edges, lost


def vdf_rebin(
    vdf,
    grid,
    base: str = "sph",
    rotation=None,
    n_mc: int = 200,
    sc_pot=None,
    random_state=None,
):
    r"""Rebins a skymap distribution onto a new spherical or Cartesian
    velocity grid, as irfu-matlab PDist.rebin.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta), in any units of
        `vdf_convert`.
    grid : dict or sequence
        For ``base="sph"``, a dict of edges of the new bins: ``energy`` (eV),
        ``phi`` and ``theta`` (degrees, look directions); a missing or None
        entry keeps the bins of `vdf` (irfu-matlab ``{energy, [], []}``). For
        ``base="cart"``, the edges [vx, vy, vz] of the new boxes (km/s).
    base : {"sph", "cart"}, Optional
        Spherical skymap (default) or Cartesian grid.
    rotation : array_like, Optional
        3x3 matrix, or (time, 3, 3) matrices, whose rows are the new unit
        vectors in the coordinate system of `vdf` (irfu-matlab ``orient``).
        Default is no rotation.
    n_mc : int, Optional
        Number of Monte-Carlo particles per bin with data. Default is 200, as
        irfu-matlab.
    sc_pot : xarray.DataArray or float or array_like, Optional
        Spacecraft potential (V), see `vdf_velocity`: the new grid is in the
        energies at infinity. Default is no correction.
    random_state : int or numpy.random.Generator, Optional
        Seed or generator of the Monte-Carlo particles.

    Returns
    -------
    out : xarray.Dataset
        ``base="sph"``: skymap on the new bins, in the units of `vdf`, as
        `vdf_shift`. ``base="cart"``: phase-space density (time, vx, vy, vz),
        in the units of `vdf` if a phase-space density, else s^3/m^6, with the
        box centres (km/s) as coordinates and the edges in the attributes
        ``vx_edges``, ``vy_edges`` and ``vz_edges``. Both have ``lost`` (time,),
        the fraction of the density outside the new grid.

    Raises
    ------
    ValueError
        If `vdf` is not a skymap, `base` is invalid or `grid` is not three
        arrays of edges for ``base="cart"``.

    See Also
    --------
    pyrfu.mms.vdf_shift, pyrfu.mms.reduce

    Notes
    -----
    As irfu-matlab, each bin with data is sampled by `n_mc` particles carrying
    f d3v / `n_mc`, and their sum in each new bin divided by its volume is the
    new phase-space density; the density is conserved apart from the
    particles outside the new grid. The particles are uniform in the volume of
    their bin (v^3, cos(theta) and phi) as in `vdf_shift`; irfu-matlab draws
    them uniformly in energy and polar angle, which biases them towards the
    low speeds and the poles of each bin.

    The result has Monte-Carlo noise: a new bin that receives N particles
    from an old one has a relative noise of about 1/sqrt(N). For a rebin in
    energy only (e.g. the 475-525 eV of EDI from the FPI channels, about 80
    of 200 particles), this is about 10 % per bin. As the angular bins are
    unchanged, the exact result is the overlap of the bins in v^3: the old
    channel i contributes f_i (v^3 overlap) / (v_2^3 - v_1^3) of the new
    channel, which the Monte-Carlo estimate approaches as `n_mc` increases.

    The spherical grid keeps the azimuths of each time without rotation (FPI
    burst azimuths change with time), and those of the first time otherwise.
    For ``base="cart"``, irfu-matlab takes the azimuth widths of burst skymaps
    from the azimuths of two times and only works with as many vy as vz
    boxes.

    Examples
    --------
    >>> import numpy as np
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)

    FPI electrons in the EDI energy channel

    >>> vdf_e_edi = mms.vdf_rebin(vdf_e, {"energy": [475.0, 525.0]})

    On a Cartesian grid

    >>> v_edges = np.linspace(-2e4, 2e4, 41)
    >>> vdf_e_cart = mms.vdf_rebin(vdf_e, [v_edges] * 3, base="cart")

    """
    if not _is_skymap(vdf):
        raise ValueError("vdf must be a skymap")

    if base not in ["sph", "cart"]:
        raise ValueError(f"Invalid base {base}: sph or cart")

    if base == "sph":
        grid = dict(grid or {})
        phi_per_time = rotation is None and grid.get("phi") is None
        out = _shift(
            vdf,
            [0.0, 0.0, 0.0],
            rotation,
            n_mc,
            grid,
            random_state,
            sc_pot=sc_pot,
            phi_per_time=phi_per_time,
        )
        del out.attrs["frame_velocity"]
        return out

    if len(grid) != 3:
        raise ValueError("grid must be the edges [vx, vy, vz] (km/s)")

    data, edges, lost = _cartesian(vdf, grid, rotation, n_mc, sc_pot, random_state)
    units = vdf.data.attrs["UNITS"]
    units = units if units.lower() in _PSD else "s^3/m^6"
    data = data / _PSD[units.lower()]
    centres = [(e[:-1] + e[1:]) / 2e3 for e in edges]

    out = xr.Dataset(
        {
            "data": (
                ["time", "vx", "vy", "vz"],
                data,
                _attrs(vdf.data.attrs, units),  # the other attributes were lost
            )
        },
        coords={
            "time": vdf.time.data,
            "vx": ("vx", centres[0], {"UNITS": "km/s"}),
            "vy": ("vy", centres[1], {"UNITS": "km/s"}),
            "vz": ("vz", centres[2], {"UNITS": "km/s"}),
        },
    )
    out.attrs = {
        "species": vdf.attrs.get("species"),
        "relativistic": bool(vdf.attrs.get("relativistic", False)),
        "vx_edges": edges[0] / 1e3,
        "vy_edges": edges[1] / 1e3,
        "vz_edges": edges[2] / 1e3,
        "lost": lost,
    }
    return out
