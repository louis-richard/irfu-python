#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Mapping, Optional, Union

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

NDArrayFloats = NDArray[Union[np.float32, np.float64]]


def ts_spectr(
    time: NDArray[np.datetime64],
    energy: NDArrayFloats,
    data: NDArrayFloats,
    comp_name: Optional[str] = None,
    attrs: Optional[Mapping[str, object]] = None,
) -> DataArray:
    r"""Create a time series containing a spectrum

    Parameters
    ----------
    time : numpy.ndarray
        Array of times.
    energy : numpy.ndarray
        Y value of the spectrum (energies, frequencies, etc.), (m,) or
        (n, m) if it varies with time.
    data : numpy.ndarray
        Data of the spectrum (n, m).
    comp_name : str, Optional
        Name of the second dimension of the DataArray. Default is "energy".
    attrs : dict, Optional
        Attributes of the spectrum (e.g., its units).

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of a spectrum, with attrs and TENSOR_ORDER = 0. If energy
        varies with time, a Dataset with the variable data (the spectrum, with
        attrs and TENSOR_ORDER = 0, as the data of ``ts_skymap``) and the
        coordinate energy (time, idx0); comp_name is not used.

    """

    # Check input type
    if not isinstance(time, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if not isinstance(energy, np.ndarray):
        raise TypeError("time must be a numpy.ndarray")

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")

    # Check input shape must be (n, m, )
    if data.ndim != 2:
        raise ValueError("Input must be a spectrum")

    if energy.ndim > 2:
        raise ValueError("Energy must be 1D or 2D")

    if len(time) != data.shape[0]:
        raise ValueError("Shape mismatch. Time and data must have the same length")

    if energy.shape[-1] != data.shape[1]:
        raise ValueError("Shape mismatch. Energy and data must have the same length")

    if comp_name is None:
        comp_name = "energy"

    if attrs is None or not isinstance(attrs, dict):
        attrs = {}

    if energy.ndim == 2:
        is_all_energy_equal = np.all(energy == energy[0, :], axis=0).all()
        if is_all_energy_equal:
            energy = energy[0, :]
    else:
        is_all_energy_equal = True

    if is_all_energy_equal:
        out: DataArray = xr.DataArray(
            data, coords=[time, energy], dims=["time", comp_name], attrs=attrs
        )
        out.attrs["TENSOR_ORDER"] = 0
    else:
        # The attributes of the spectrum on its data (they were dropped)
        out = xr.Dataset(
            {"data": (["time", "idx0"], data, {**attrs, "TENSOR_ORDER": 0})},
            coords={
                "time": time,
                "idx0": np.arange(energy.shape[1]),
                "energy": (["time", "idx0"], energy),
            },
        )

    return out
