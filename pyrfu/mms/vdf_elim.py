#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def vdf_elim(vdf, e_int):
    r"""Limits the skymap distribution to the selected energy range.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap velocity distribution to clip.
    e_int : list or float
        Energy interval boundaries (list) or energy to slice.

    Returns
    -------
    vdf_e_clipped : xarray.Dataset
        Skymap of the clipped velocity distribution.

    """

    energy = vdf.energy
    unique_etables = np.unique(vdf.energy.data, axis=0)
    # n_etables = 2 for older dta and 1 for newer data
    n_etables = unique_etables.shape[0]

    e_int = list(np.atleast_1d(e_int))
    e_int.sort()

    # energy interval
    if len(e_int) == 2:
        e_levels = []

        for i_etable in range(n_etables):
            # loop over 1 or 2 and saves all the unique indices, i.e. max range
            lower_ = e_int[0] < unique_etables[i_etable, :]
            upper_ = unique_etables[i_etable, :] < e_int[1]
            tmp_elevels = np.where(np.logical_and(lower_, upper_))[0]
            e_levels = np.unique(np.hstack([e_levels, tmp_elevels]))

        e_levels = list(e_levels.astype(np.int64))
        e_min = np.min(energy.data[:, e_levels])
        e_max = np.max(energy.data[:, e_levels])
        logging.info(
            "Effective eint = [%(e_min)5.2f, %(e_max)5.2f]",
            {"e_min": e_min, "e_max": e_max},
        )
        energies = energy.data[:, e_levels]
        data = vdf.data.data[:, e_levels, ...]

    else:
        # pick closest energy level
        e_diff0 = np.abs(energy[0, :] - e_int)
        e_diff1 = np.abs(energy[1, :] - e_int)
        if np.min(e_diff0) < np.min(e_diff1):
            e_diff = e_diff0
        else:
            e_diff = e_diff1

        e_levels = int(np.where(e_diff == np.min(e_diff))[0][0])
        logging.info(
            "Effective energies alternate in time between %(e0)5.2f and %(e1)5.2f",
            {"e0": energy.data[0, e_levels], "e1": energy.data[1, e_levels]},
        )
        energies = energy.data[:, e_levels]
        energies = energies[:, np.newaxis]
        data = vdf.data.data[:, e_levels, ...]
        data = data[:, np.newaxis, ...]

    # Data attributes
    data_attrs = vdf.data.attrs

    # Coordinates attributes
    coords_attrs = {k: vdf[k].attrs for k in ["time", "energy", "phi", "theta"]}

    # Global attributes
    glob_attrs = vdf.attrs

    # Get energies levels
    energy_0 = np.atleast_1d(glob_attrs.get("energy0", unique_etables[0, :])[e_levels])
    energy_1 = np.atleast_1d(glob_attrs.get("energy1", unique_etables[0, :])[e_levels])
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
