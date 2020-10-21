#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ts_skymap.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def ts_skymap(t=None, data=None, energy=None, phi=None, theta=None, **kwargs):
    """Creates a skymap of the distribution function.

    Parameters
    ----------
    t : numpy.ndarray
        List of times.

    data : numpy.ndarray
        Values of the distribution function.

    energy : numpy.ndarray
        Energy levels.

    phi : numpy.ndarray
        Azimuthal angles.

    theta : numpy.ndarray
        Elevation angles.

    **kwargs : dict
        Hash table of keyword arguments with :
            * energy0 : numpy.ndarray
                Energy table 0 (odd time indices).

            * energy1 : numpy.ndarray
                Energy table 1 (even time indices).

            * esteptable : numpy.ndarray
                Time series of the stepping table between energies (burst).


    Returns
    -------
    out : xarray.DataArray
        Skymap of the distribution function.

    """

    assert t is not None and isinstance(t, np.ndarray)
    assert data is not None and isinstance(data, xr.DataArray)
    assert phi is not None and isinstance(phi, np.ndarray)
    assert theta is not None and isinstance(theta, np.ndarray)

    if energy is None:
        if "energy0" in kwargs:
            energy0, energy0_ok = [kwargs["energy0"], True]
        else:
            energy0, energy0_ok = [None, False]

        if "energy1" in kwargs:
            energy1, energy1_ok = [kwargs["energy1"], True]
        else:
            energy1, energy1_ok = [None, False]

        if "esteptable" in kwargs:
            esteptable, esteptable_ok = [kwargs["esteptable"], True]
        else:
            esteptable, esteptable_ok = [None, False]

        if not energy0_ok and not energy1_ok and not esteptable_ok:
            raise ValueError("Energy input required")

        energy = np.tile(energy0, (len(esteptable), 1))

        energy[esteptable == 1] = np.tile(energy1, (int(np.sum(esteptable)), 1))

    else:
        energy0, energy1, esteptable = [None] * 3

        energy0_ok, energy1_ok, esteptable_ok = [False] * 3

    if phi.ndim == 1:
        phi = np.tile(phi, (len(t), 1))

    out_dict = {"data": (["time", "idx0", "idx1", "idx2"], data), "phi": (["time", "idx1"], phi),
                "theta": (["idx2"], theta), "energy": (["time", "idx0"], energy), "time": t,
                "idx0": np.arange(32),
                "idx1": np.arange(32), "idx2": np.arange(16)}

    out = xr.Dataset(out_dict)

    if energy0_ok:
        out.attrs["energy0"] = energy0

    if energy1_ok:
        out.attrs["energy1"] = energy1

    if energy0_ok:
        out.attrs["esteptable"] = esteptable

    return out
