#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def ts_skymap(time, data, energy, phi, theta, **kwargs):
    r"""Creates a skymap of the distribution function.

    Parameters
    ----------
    time : ndarray
        List of times.
    data : ndarray
        Values of the distribution function.
    energy : ndarray
        Energy levels.
    phi : ndarray
        Azimuthal angles.
    theta : ndarray
        Elevation angles.

    Other Parameters
    ---------------
    **kwargs
        Hash table of keyword arguments with :
            * energy0 : ndarray
                Energy table 0 (odd time indices).
            * energy1 : ndarray
                Energy table 1 (even time indices).
            * esteptable : ndarray
                Time series of the stepping table between energies (burst).


    Returns
    -------
    out : xarray.Dataset
        Skymap of the distribution function.

    """

    energy0, energy1, esteptable = [None] * 3
    energy0_ok, energy1_ok, esteptable_ok = [False] * 3

    if energy is None:

        if "energy0" in kwargs:
            energy0, energy0_ok = [kwargs["energy0"], True]

        if "energy1" in kwargs:
            energy1, energy1_ok = [kwargs["energy1"], True]

        if "esteptable" in kwargs:
            esteptable, esteptable_ok = [kwargs["esteptable"], True]

        if not energy0_ok and not energy1_ok and not esteptable_ok:
            raise ValueError("Energy input required")

        energy = np.tile(energy0, (len(esteptable), 1))

        energy[esteptable == 1] = np.tile(energy1,
                                          (int(np.sum(esteptable)), 1))
    if phi.ndim == 1:
        phi = np.tile(phi, (len(time), 1))

    out_dict = {"data": (["time", "idx0", "idx1", "idx2"], data),
                "phi": (["time", "idx1"], phi), "theta": (["idx2"], theta),
                "energy": (["time", "idx0"], energy), "time": time,
                "idx0": np.arange(energy.shape[1]),
                "idx1": np.arange(phi.shape[1]), "idx2": np.arange(len(theta))}

    out = xr.Dataset(out_dict)

    if energy0_ok:
        out.attrs["energy0"] = energy0

    if energy1_ok:
        out.attrs["energy1"] = energy1

    if energy0_ok:
        out.attrs["esteptable"] = esteptable

    return out
