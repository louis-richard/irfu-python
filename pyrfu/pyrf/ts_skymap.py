#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray import Dataset

# Local imports
from pyrfu._core.typing import FloatArray, TimeArray


def ts_skymap(
    time: TimeArray,
    data: FloatArray,
    energy: FloatArray,
    phi: NDArray[np.number[Any]],
    theta: NDArray[np.number[Any]],
    energy0: FloatArray | None = None,
    energy1: FloatArray | None = None,
    esteptable: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
    glob_attrs: dict[str, Any] | None = None,
    coords_attrs: dict[str, dict[str, Any]] | None = None,
) -> Dataset:
    r"""Creates a skymap of the distribution function.

    Parameters
    ----------
    time : np.ndarray
        List of times.
    data : np.ndarray
        Values of the distribution function.
    energy : np.ndarray
        Energy levels.
    phi : np.ndarray
        Azimuthal angles, (time, phi) or (phi,) (then the same at all times).
    theta : np.ndarray
        Elevation angles.
    energy0 : np.ndarray, Optional
        Energy table 0 (odd time indices). Default is the energies of the first
        time.
    energy1 : np.ndarray, Optional
        Energy table 1 (even time indices). Default is the energies of the second
        time, or of the first one for a single time.
    esteptable : np.ndarray, Optional
        Time series of the stepping table between energies (burst).
    attrs : dict, Optional
        Metadata for the VDF.
    glob_attrs : dict, Optional
        Global attributes of the dataset.
    coords_attrs : dict, Optional
        Coordinates attributes of the dataset.

    Returns
    -------
    out : xarray.Dataset
        Skymap of the distribution function.

    Raises
    ------
    TypeError
        If time, data, energy, phi, or theta are not numpy.ndarray.
    TypeError
        If energy0, energy1, or esteptable are not numpy.ndarray.
    TypeError
        If attrs, glob_attrs, or coords_attrs are not dict.

    """
    # Check input type
    if not isinstance(time, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")

    if not isinstance(energy, np.ndarray):
        raise TypeError("energy must be a numpy.ndarray")

    if not isinstance(phi, np.ndarray):
        raise TypeError("phi must be a numpy.ndarray")

    if not isinstance(theta, np.ndarray):
        raise TypeError("theta must be a numpy.ndarray")

    # Azimuths as (time, phi), as get_dist for the fast mode
    if phi.ndim == 1:
        phi = np.tile(phi, (len(time), 1))

    # Check if even (odd) time step energy channels energy1 (energy0), and
    # energy step table are provided.
    if energy0 is None:
        energy0 = energy[0, :]
    else:
        if not isinstance(energy0, np.ndarray):
            raise TypeError("energy0 must be a numpy.ndarray")

    if energy1 is None:
        # A single time (e.g. a distribution selected at one time) has one table
        energy1 = energy[1, :] if energy.shape[0] > 1 else energy[0, :]
    else:
        if not isinstance(energy1, np.ndarray):
            raise TypeError("energy1 must be a numpy.ndarray")

    if esteptable is None:
        esteptable = np.zeros(len(time), dtype=np.uint8)
    else:
        if not isinstance(esteptable, np.ndarray):
            raise TypeError("esteptable must be a numpy.ndarray")

    # Check attributes are dictionaries
    if attrs is None:
        attrs = {}
    else:
        if not isinstance(attrs, dict):
            raise TypeError("attrs must be a dictionary")

    # Check coordinates attributes are dictionaries
    if coords_attrs is None:
        coords_attrs = {}
    else:
        if not isinstance(coords_attrs, dict):
            raise TypeError("coords_attrs must be a dictionary")

    # Check global attributes are dictionaries
    if glob_attrs is None:
        glob_attrs = {}
    else:
        if not isinstance(glob_attrs, dict):
            raise TypeError("glob_attrs must be a dictionary")

    out_dict = {
        "data": (["time", "idx0", "idx1", "idx2"], data),
        "phi": (["time", "idx1"], phi),
        "theta": (["idx2"], theta),
        "energy": (["time", "idx0"], energy),
        "time": time,
        "idx0": np.arange(energy.shape[1]),
        "idx1": np.arange(phi.shape[1]),
        "idx2": np.arange(len(theta)),
    }

    # Construct global attributes and sort them
    # remove energy0, energy1, and esteptable from global attrs to overwrite
    overwrite_keys = ["energy0", "energy1", "esteptable"]
    glob_attrs = {k: glob_attrs[k] for k in glob_attrs if k not in overwrite_keys}
    glob_attrs = {
        "energy0": energy0,
        "energy1": energy1,
        "esteptable": esteptable,
        **glob_attrs,
    }

    glob_attrs = {k: glob_attrs[k] for k in sorted(glob_attrs)}

    # Create Dataset
    out = xr.Dataset(out_dict, attrs=glob_attrs)

    # Sort and fill coordinates attributes
    for k in coords_attrs:
        out[k].attrs = {m: coords_attrs[k][m] for m in sorted(coords_attrs[k])}

    # Sort and fill data attributes
    out.data.attrs = {k: attrs[k] for k in sorted(attrs)}

    return out
