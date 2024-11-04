#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import string
from typing import Dict, Optional

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from .. import pyrf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


__all__ = [
    "generate_timeline",
    "generate_data",
    "generate_ts",
    "generate_spectr",
    "generate_vdf",
    "generate_defatt",
]


def generate_timeline(
    f_s: float,
    n_pts: int = 10000,
    dtype: str = "datetime64[ns]",
    ref_time: str = "2019-01-01T00:00:00.000000000",
) -> NDArray[np.datetime64]:
    r"""Generate timeline for testings

    Parameters
    ----------
    f_s : float
        Sampling frequency.
    n_pts : int
        Number of points.
    dtype : str
        Data type of the time array.
    ref_time : str
        Reference time.

    Returns
    -------
    times : numpy.ndarray
        Timeline.

    """
    ref_time = np.datetime64(ref_time)
    times = [ref_time + np.timedelta64(int(i * 1e9 / f_s), "ns") for i in range(n_pts)]
    times = np.array(times).astype(dtype)
    return times


def generate_data(n_pts: int, tensor_order: int = 0) -> NDArray[np.float32]:
    r"""Generate data (numpy.ndarray) for testings

    Parameters
    ----------
    n_pts : int
        tensor_order
    tensor_order : int
        Tensor order of the data.

    Returns
    -------
    data : numpy.ndarray
        Synthetic data.

    """

    data = np.random.random((n_pts, *([3] * tensor_order)))

    return data.astype(np.float32)


def generate_ts(
    f_s: float,
    n_pts: int,
    tensor_order: int = 0,
    attrs: Optional[Dict[str, object]] = None,
) -> DataArray:
    r"""Generate timeseries for testings

    Parameters
    ----------
    f_s : float
        Sampling frequency.
    n_pts : int
        Number of points.
    tensor_order : int
        Tensor order of the data.
    attrs : dict, Optional
        Attributes of the timeseries.

    Returns
    -------
    DataArray
        Synthetic timeseries.

    """
    if attrs is None:
        attrs = {}

    time = generate_timeline(f_s, n_pts)
    data = generate_data(n_pts, tensor_order=tensor_order)

    if tensor_order == 0:
        out = pyrf.ts_scalar(time, data, attrs=attrs)
    elif tensor_order == 1:
        out = pyrf.ts_vec_xyz(time, data, attrs=attrs)
    elif tensor_order == 2:
        out = pyrf.ts_tensor_xyz(time, data, attrs=attrs)
    else:
        coords = [time, *[np.arange(data.shape[i]) for i in range(1, tensor_order + 1)]]
        dims = ["time", *list(string.ascii_lowercase[8 : 8 + tensor_order])]
        out = xr.DataArray(data, coords=coords, dims=dims, attrs=attrs)

    out.time.attrs = {"UNITS": "I AM GROOT!!"}

    return out


def generate_spectr(
    f_s: float, n_pts: int, shape: tuple, attrs: Optional[Dict[str, object]] = None
) -> DataArray:
    r"""Generates spectrum for testings

    Parameters
    ----------
    f_s : float
        Sampling frequency.
    n_pts : int
        Number of points.
    shape : tuple
        Shape of the spectrum.
    attrs : dict
        Attributes of the spectrum.

    Returns
    -------
    DataArray
        Synthetic spectrum.


    """
    out: DataArray = pyrf.ts_spectr(
        generate_timeline(f_s, n_pts),
        np.arange(shape),
        np.random.random((n_pts, shape)),
        attrs=attrs,
    )
    return out


def generate_vdf(
    f_s: float,
    n_pts: int,
    shape: list,
    energy01: Optional[bool] = None,
    species: Optional[str] = None,
    units: Optional[str] = None,
) -> Dataset:
    r"""Generate skymap for testings

    Parameters
    ----------
    f_s : float
        Sampling frequency.
    n_pts : int
        Number of points.
    shape : list
        Shape of the vdf.
    energy01 : bool, Optional
        If True, energy is set to +0.5 for odd time steps (first part of MMS mission).
    species : str, Optional
        Species of the vdf.
    units : str, Optional
        Units of the vdf.

    Returns
    -------
    Dataset
        Synthetic vdf.

    """

    if energy01 is None:
        energy01 = False

    if species is None:
        species = "ions"

    if units is None:
        units = "s^3/cm^6"

    times = generate_timeline(f_s, n_pts)

    phi = np.arange(shape[1])
    phi = np.tile(phi, (n_pts, 1))
    theta = np.arange(shape[2])
    data = np.random.random((n_pts, *shape))

    if energy01:
        energy0 = np.linspace(0, shape[0], shape[0], endpoint=False)
        energy1 = np.linspace(0, shape[0], shape[0], endpoint=False) + 0.5
        esteptable = np.arange(n_pts, dtype=np.uint8) % 2
        energy = np.empty((n_pts, shape[0]), dtype=np.float32)
        n_energy_1 = int(np.sum(esteptable))
        n_energy_0 = n_pts - n_energy_1
        energy[esteptable == 0, :] = np.tile(energy0, (n_energy_0, 1))
        energy[esteptable == 1, :] = np.tile(energy1, (n_energy_1, 1))
    else:
        energy = np.linspace(0, shape[0], shape[0], endpoint=False, dtype=np.float32)
        energy = np.tile(energy, (n_pts, 1))
        energy0 = energy[0, :]
        energy1 = energy[1, :]
        esteptable = np.zeros(n_pts, dtype=np.uint8)

    attrs = {"UNITS": units}
    glob_attrs = {
        "species": species,
        "delta_energy_plus": np.ones((n_pts, shape[0])),
        "delta_energy_minus": np.ones((n_pts, shape[0])),
    }

    out = pyrf.ts_skymap(
        times,
        data,
        energy,
        phi,
        theta,
        energy0=energy0,
        energy1=energy1,
        esteptable=esteptable,
        attrs=attrs,
        glob_attrs=glob_attrs,
    )
    return out


def generate_defatt(f_s: float, n_pts: int) -> Dataset:
    r"""Generate defatt for testings.

    Parameters
    ----------
    f_s : float
        Sampling frequency.
    n_pts : int
        Number of points.

    Returns
    -------
    defatt : Dataset
        Defatt dataset.

    """
    z_ra = generate_ts(f_s, n_pts, 0)
    z_dec = generate_ts(f_s, n_pts, 0)
    defatt_dict = {"z_ra": z_ra, "z_dec": z_dec}
    return xr.Dataset(defatt_dict)
