#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.pyrf.ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def vdf_elim(vdf, e_int):
    r"""Limits the skymap distribution to the selected energy range.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap velocity distribution to clip.
    e_int : list or float
        Energy interval boundaries (list), keeping the channels strictly
        within, or energy to slice (the closest channel).

    Returns
    -------
    vdf_e_clipped : xarray.Dataset
        Skymap of the clipped velocity distribution. The energy tables and the
        delta_energy_plus/minus attributes are clipped to the same channels.

    Raises
    ------
    ValueError
        If no energy channel is within the interval, or if e_int has more than
        two elements.

    """

    energy = vdf.energy
    unique_etables = np.unique(vdf.energy.data, axis=0)

    e_int = sorted(np.atleast_1d(e_int).tolist())

    # energy interval
    if len(e_int) == 2:
        # saves all the unique indices over the 1 (newer data) or 2 (older
        # data) energy tables, i.e. max range
        e_levels = np.unique(
            np.hstack(
                [
                    np.where((e_int[0] < table) & (table < e_int[1]))[0]
                    for table in unique_etables
                ]
            )
        )

        if e_levels.size == 0:
            raise ValueError(f"No energy channel between {e_int[0]} and {e_int[1]}")

        e_levels = list(e_levels.astype(np.int64))
        logger.info(
            "Effective eint = [%(e_min)5.2f, %(e_max)5.2f]",
            {
                "e_min": np.min(energy.data[:, e_levels]),
                "e_max": np.max(energy.data[:, e_levels]),
            },
        )

    elif len(e_int) == 1:
        # pick closest energy level, in the energy table closest to e_int
        e_diff = np.abs(unique_etables - e_int[0])
        i_table = np.argmin(np.min(e_diff, axis=1))
        e_levels = [int(np.argmin(e_diff[i_table]))]
        logger.info(
            "Effective energies alternate in time between %(energies)s",
            {"energies": unique_etables[:, e_levels[0]]},
        )

    else:
        raise ValueError("e_int must be an energy or an interval of two energies")

    energies = energy.data[:, e_levels]
    data = vdf.data.data[:, e_levels, ...]

    # Data attributes
    data_attrs = vdf.data.attrs

    # Coordinates attributes
    coords_attrs = {k: vdf[k].attrs for k in ["time", "energy", "phi", "theta"]}

    # Global attributes, with the energy widths clipped as the energies
    glob_attrs = dict(vdf.attrs)

    for key in ["delta_energy_minus", "delta_energy_plus"]:
        if glob_attrs.get(key) is not None:
            glob_attrs[key] = np.asarray(glob_attrs[key])[..., e_levels]

    # Get energies levels
    energy_0 = np.atleast_1d(glob_attrs.get("energy0", unique_etables[0, :])[e_levels])
    energy_1 = np.atleast_1d(glob_attrs.get("energy1", unique_etables[-1, :])[e_levels])
    esteptable = glob_attrs.get("esteptable", np.zeros(len(vdf.time)))

    vdf_e_clipped = ts_skymap(
        vdf.time.data,
        data,
        energies,
        vdf.phi.data,
        vdf.theta.data,
        energy0=energy_0,
        energy1=energy_1,
        esteptable=esteptable,
        attrs=data_attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )

    return vdf_e_clipped
