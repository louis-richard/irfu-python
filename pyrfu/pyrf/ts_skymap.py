#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any, Iterable, Mapping, Optional

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def ts_skymap(
    time: NDArray[np.datetime64],
    data: NDArray[np.float64],
    energy: NDArray[np.float64],
    phi: NDArray[np.float64],
    theta: NDArray[np.float64],
    energy0: Optional[NDArray[np.float64]] = None,
    energy1: Optional[NDArray[np.float64]] = None,
    esteptable: Optional[NDArray[np.float64]] = None,
    attrs: Optional[Mapping[str, object]] = None,
    glob_attrs: Optional[Mapping[str, object]] = None,
    coords_attrs: Optional[Mapping[str, Mapping[str, Iterable[Any]]]] = None,
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
        Azimuthal angles.
    theta : np.ndarray
        Elevation angles.
    energy0: np.ndarray, Optional
        Energy table 0 (odd time indices).
    energy1: np.ndarray, Optional
        Energy table 1 (even time indices).
    esteptable: np.ndarray, Optional
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

    # Check if even (odd) time step energy channels energy1 (energy0), and
    # energy step table are provided.
    if energy0 is None:
        energy0 = energy[0, :]
    else:
        if not isinstance(energy0, np.ndarray):
            raise TypeError("energy0 must be a numpy.ndarray")

    if energy1 is None:
        energy1 = energy[1, :]
    else:
        if not isinstance(energy1, np.ndarray):
            raise TypeError("energy0 must be a numpy.ndarray")

    if esteptable is None:
        esteptable = np.zeros(len(time))
    else:
        if not isinstance(esteptable, np.ndarray):
            raise TypeError("esteptable must be a numpy.ndarray")

    # Check that energy0 and energy1
    # assert energy0.ndim == 1, "energy0 must be 1D numpy.ndarray"
    # assert energy0.shape[0] == energy.shape[1], "energy0 is not consistent with time"
    # assert energy1.ndim == 1, "energy1 must be 1D numpy.ndarray"
    # assert energy1.shape[0] == energy.shape[1], "energy1 is not consistent with time"

    # Check esteptable
    # assert esteptable.ndim == 1, "esteptable must be 1D numpy.ndarray"
    # assert esteptable.shape[0] == len(time), "esteptable is not consistent with time"

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
