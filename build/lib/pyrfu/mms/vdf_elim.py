#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf import ts_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.10"
__status__ = "Prototype"


logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


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
    unique_etables = np.unique(vdf.energy, axis=0)
    # n_etables = 2 for older dta and 1 for newer data
    n_etables = unique_etables.shape[0]

    e_int = list(np.atleast_1d(e_int))

    # energy interval
    if len(e_int) == 2:
        e_levels = []

        for i_etable in range(n_etables):
            # loop over 1 or 2 and saves all the unique indices, i.e. max range
            lower_ = e_int[0] < unique_etables[i_etable, :]
            upper_ = unique_etables[i_etable, :] < e_int[1]
            tmp_elevels = np.where(np.logical_and(lower_, upper_))[0]
            e_levels = np.unique(np.hstack([e_levels, tmp_elevels]))

        e_levels = list(e_levels.astype(int))
        e_min = np.min(energy.data[:, e_levels])
        e_max = np.max(energy.data[:, e_levels])
        print(f"Effective eint = [{e_min:5.2f}, {e_max:5.2f}]")

    else:
        # pick closest energy level
        e_diff0 = np.abs(energy[0, :] - e_int)
        e_diff1 = np.abs(energy[1, :] - e_int)
        if np.min(e_diff0) < np.min(e_diff1):
            e_diff = e_diff0
        else:
            e_diff = e_diff1

        e_levels = int(np.where(e_diff == np.min(e_diff))[0][0])
        print(f"Effective energies alternate in time between "
              f"{energy.data[0, e_levels]:5.2f} and "
              f"{energy.data[1, e_levels]:5.2f}")

    vdf_e_clipped = ts_skymap(vdf.time.data, vdf.data.data[:, e_levels, ...],
                              energy=energy.data[:, e_levels],
                              phi=vdf.phi.data, theta=vdf.theta.data)

    energy_0 = vdf.attrs.get("energy0", unique_etables[0, :])[e_levels]
    energy_1 = vdf.attrs.get("energy1", unique_etables[0, :])[e_levels]
    esteptable = vdf.attrs.get("esteptable", np.zeros(len(vdf.time)))
    vdf_e_clipped.attrs["energy0"] = energy_0
    vdf_e_clipped.attrs["energy1"] = energy_1
    vdf_e_clipped.attrs["esteptable"] = esteptable

    return vdf_e_clipped
