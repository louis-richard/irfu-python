#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ._vdf_bins import _is_skymap
from ._vdf_mc import _rng, _sample, _source_bins
from .vdf_d3v import vdf_d3v
from .vdf_partial_density import vdf_partial_density
from .vdf_velocity import vdf_velocity

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def _particles_per_bin(weights, n_tot, n_bin, skip_zero, division):
    # Number of macroparticles in each bin (time, energy, phi, theta)
    if n_tot is None:
        return np.full(weights.shape, int(n_bin))

    total = np.sum(weights.reshape(len(weights), -1), axis=1)
    total = total.reshape((-1,) + (1,) * (weights.ndim - 1))

    with np.errstate(invalid="ignore", divide="ignore"):
        n_target = np.nan_to_num(weights / total * n_tot)

    rounding = np.floor if division == "counts" else np.ceil
    n_part = rounding(n_target).astype(int)

    if not skip_zero:
        n_part[n_part == 0] = 1

    return n_part


def vdf_macroparticles(
    vdf,
    n_tot: int = None,
    n_bin: int = 1,
    skip_zero: bool = True,
    sc_pot=None,
    positioning: str = "random",
    division: str = "dn",
    random_state=None,
):
    r"""Macroparticles of a skymap distribution, as irfu-matlab
    PDist.macroparticles.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta), in any units of
        `vdf_convert`.
    n_tot : int, Optional
        Total number of macroparticles at each time, divided among the bins in
        proportion to `division` (rounded up, so that every bin with data has
        at least one). Default is None: `n_bin` per bin.
    n_bin : int, Optional
        Number of macroparticles per bin, if `n_tot` is None. Default is 1.
    skip_zero : bool, Optional
        No macroparticle in the bins without data (zero or NaN). Default is
        True.
    sc_pot : xarray.DataArray or float or array_like, Optional
        Spacecraft potential (V), see `vdf_velocity`. Default is no
        correction.
    positioning : {"random", "center"}, Optional
        Macroparticles drawn uniformly in the volume of their bin (v^3,
        cos(theta) and phi uniform; default), or at the velocity of its
        centre (`vdf_velocity`).
    division : {"dn", "f", "counts"}, Optional
        With `n_tot`, macroparticles in proportion to the partial density
        f d3v (default), to the data, or to the data rounded down (one
        macroparticle per count for a distribution of counts).
    random_state : int or numpy.random.Generator, Optional
        Seed or generator of the random positions.

    Returns
    -------
    particles : xarray.Dataset
        Macroparticles along the ``particle`` dimension, with the coordinates
        ``time`` and the bin indices ``idx0``, ``idx1`` and ``idx2`` (energy,
        phi, theta), and the velocity ``vx``, ``vy``, ``vz`` (km/s,
        coordinates of `vdf`), the density ``dn`` (cm^-3), the data of the
        bin ``df`` (units of `vdf`) and the volume ``dv`` (units of
        `vdf_d3v`). The bin's partial density and volume are shared equally by
        its macroparticles, so that ``dn`` summed over the macroparticles of a
        time is the density (of the bins with particles).

    Raises
    ------
    ValueError
        If `vdf` is not a skymap, or `positioning` or `division` is invalid.

    Notes
    -----
    irfu-matlab draws the energy and the polar angle uniformly, which puts
    more macroparticles at the low speeds and near the poles of each bin than
    the volume does. It uses the energy table of the first time for all times
    and drops the particles below the spacecraft potential; here the energies
    of each time are used and the speeds are those of `vdf_d3v` (the energies
    at infinity, at least 0), so that the density is kept.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> particles = mms.vdf_macroparticles(vdf_i, n_tot=5000, random_state=0)
    >>> n_i = particles.dn.groupby("time").sum()

    """
    if not _is_skymap(vdf):
        raise ValueError("vdf must be a skymap")

    if positioning not in ["random", "center"]:
        raise ValueError(f"Invalid positioning {positioning}: random or center")

    if division not in ["dn", "f", "counts"]:
        raise ValueError(f"Invalid division {division}: dn, f or counts")

    f_data = np.nan_to_num(vdf.data.data)
    d3v = vdf_d3v(vdf, sc_pot)
    d_n = np.nan_to_num(vdf_partial_density(vdf, sc_pot=sc_pot).data.data)
    weights = d_n if division == "dn" else f_data
    n_part = _particles_per_bin(weights, n_tot, n_bin, skip_zero, division)

    if skip_zero:
        n_part[d_n == 0] = 0

    i_t, i_e, i_p, i_q = np.nonzero(n_part)
    repeats = n_part[i_t, i_e, i_p, i_q]
    i_t, i_e, i_p, i_q = (np.repeat(i, repeats) for i in [i_t, i_e, i_p, i_q])
    n_bin_part = n_part[i_t, i_e, i_p, i_q]

    if positioning == "random":
        bins = _source_bins(vdf, sc_pot)
        v_xyz = _sample(_rng(random_state), bins, i_t, i_e, i_p, i_q) / 1e3
    else:
        velocity = vdf_velocity(vdf, sc_pot)
        v_xyz = np.stack(
            [velocity[c].data[i_t, i_e, i_p, i_q] for c in ["vx", "vy", "vz"]],
            axis=1,
        )

    particles = xr.Dataset(
        {
            "vx": ("particle", v_xyz[:, 0], {"UNITS": "km/s"}),
            "vy": ("particle", v_xyz[:, 1], {"UNITS": "km/s"}),
            "vz": ("particle", v_xyz[:, 2], {"UNITS": "km/s"}),
            "dn": ("particle", d_n[i_t, i_e, i_p, i_q] / n_bin_part),
            "df": ("particle", f_data[i_t, i_e, i_p, i_q]),
            "dv": ("particle", d3v.data.data[i_t, i_e, i_p, i_q] / n_bin_part),
        },
        coords={
            "time": ("particle", vdf.time.data[i_t]),
            "idx0": ("particle", i_e),
            "idx1": ("particle", i_p),
            "idx2": ("particle", i_q),
        },
    )
    particles["dn"].attrs["UNITS"] = "cm^-3"
    particles["df"].attrs["UNITS"] = vdf.data.attrs.get("UNITS", "")
    particles["dv"].attrs["UNITS"] = d3v.data.attrs["UNITS"]
    particles.attrs = {
        "species": vdf.attrs.get("species"),
        "positioning": positioning,
    }
    return particles
