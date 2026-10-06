#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Tuple, Union

# 3rd party
import numpy as np
from numpy.typing import NDArray
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def psd_rebin(
    vdf: Dataset,
    phi: NDArray[np.float32],
    energy0: NDArray[np.float32],
    energy1: NDArray[np.float32],
    esteptable: NDArray[np.uint8],
) -> Tuple[
    NDArray[np.datetime64],
    NDArray[Union[np.float32, np.float64]],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    r"""Convert burst mode distribution into 64 energy channel distribution.

    Takes the burst mode distribution sampled in two energy tables and converts to a
    single energy table with 64 energy channels. Time resolution is halved and phi
    angles are averaged over adjacent times.

    Parameters
    ----------
    vdf : Dataset
        Time series of the particle distribution.
    phi : numpy.ndarray
        Time series of the phi angles.
    energy0 : numpy.ndarray
        Energy table 0.
    energy1 : numpy.ndarray
        Energy table 1.
    esteptable : numpy.ndarray
        Time series of the stepping table between energies (burst).

    Returns
    -------
    tuple
        * time_r : numpy.ndarray
            Time array of the recompiled distribution.
        * vdf_r : numpy.ndarray
            Recompiled distribution.
        * energy_r : numpy.ndarray
            Recompiled energy table.
        * phi_r : numpy.ndarray
            Recompiled phi angles.

    Raises
    ------
    TypeError
        If vdf is not a xarray.Dataset or if phi is not a numpy.ndarray or if energy0 is
        not a numpy.ndarray or if energy1 is not a numpy.ndarray or if esteptable is
        not a numpy.ndarray.

    Notes
    -----
    I'm assuming no gaps in the burst data interval. If there is a gap use
    time_clip before running. To be updated later.

    Between two samples, the spin advances the azimuths (2.8 deg for DIS,
    0.56 deg for DES burst), and when the first azimuth would pass one bin
    (11.25 deg) the azimuths are labelled from one bin lower: column j + 1 of
    the second sample then looks in the direction of column j of the first.
    The second sample is shifted by -1 bin to pair the same directions (with
    360 deg added to its last azimuth). This differs from irfu-matlab
    mms.psd_rebin, which shifts by +1 bin and pairs directions 20 deg apart
    (checked on MMS3 DIS and DES burst data, 2015-10-30).

    """
    if not isinstance(vdf, Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    if not isinstance(phi, np.ndarray):
        raise TypeError("phi must be a numpy.ndarray")

    if not isinstance(energy0, np.ndarray):
        raise TypeError("energy0 must be a numpy.ndarray")

    if not isinstance(energy1, np.ndarray):
        raise TypeError("energy1 must be a numpy.ndarray")

    if not isinstance(esteptable, np.ndarray):
        raise TypeError("step_table must be a numpy.ndarray")

    # Get time and data
    vdf_time: NDArray[np.datetime64] = vdf.time.data
    vdf_data: NDArray[np.float64] = vdf.data.data.astype(np.float64)

    # Sort energy levels
    energy_r: NDArray[np.float32] = np.sort(np.hstack([energy0, energy1]))

    # Define new times
    # Sampling period in ns (kept as int64, int16 overflows for ms time steps)
    delta_t: float = np.median(
        np.diff(vdf_time).astype("timedelta64[ns]").astype(np.int64)
    )
    time_r: NDArray[np.datetime64] = vdf_time[:-1:2] + np.timedelta64(
        int(round(delta_t / 2)), "ns"
    )

    # Preallocate output arrays
    vdf_r: NDArray[np.float64] = np.zeros(
        (len(time_r), 64, *vdf.data.shape[2:]), dtype=np.float64
    )
    phi_r: NDArray[np.float32] = np.zeros(
        (len(time_r), vdf.data.shape[2]), dtype=np.float32
    )

    # Azimuths of the next bin, for the pairs where the azimuths are labelled
    # from one bin lower in the second sample (see Notes)
    phi_s: NDArray[np.float64] = np.roll(phi.astype(np.float64), -1, axis=1)
    phi_s[:, -1] = phi_s[:, -1] + 360.0

    # Default integer type (int16 wraps after 32767 samples)
    time_indices: NDArray[np.int_] = np.arange(0, len(vdf.time) - 1, 2)

    for new_el_num, idx in enumerate(time_indices):
        if phi[idx, 0] > phi[idx + 1, 0]:
            phi_r[new_el_num, :] = (phi[idx, :] + phi_s[idx + 1, :]) / 2

            vdf_temp: NDArray[np.float64] = np.roll(vdf_data[idx + 1, ...], -1, axis=1)

            if esteptable[idx]:
                vdf_r[new_el_num, 1:64:2, ...] = vdf_data[idx, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf_temp
            else:
                vdf_r[new_el_num, 0:63:2, ...] = vdf_data[idx, ...]
                vdf_r[new_el_num, 1:64:2, ...] = vdf_temp

        else:
            phi_r[new_el_num, :] = phi[idx, :] + phi[idx + 1, :]
            phi_r[new_el_num, :] /= 2

            if esteptable[idx]:
                vdf_r[new_el_num, 1:64:2, ...] = vdf_data[idx, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf_data[idx + 1, ...]
            else:
                vdf_r[new_el_num, 1:64:2, ...] = vdf_data[idx + 1, ...]
                vdf_r[new_el_num, 0:63:2, ...] = vdf_data[idx, ...]

    out = (time_r, vdf_r.astype(vdf.data.data.dtype), energy_r, phi_r)
    return out
