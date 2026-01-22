#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from pycdfpp import load

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime64 import iso86012datetime64
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_skymap import ts_skymap
from .get_ts import _get_epochs
from .get_variable import _pycdfpp_attributes_to_dict

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Keys of the global attributes to keep from CDF informations
Globkeys = [
    "CDF",
    "Version",
    "Encoding",
    "Checksum",
    "Compressed",
    "LeapSecondUpdated",
]


def get_dist(file_path, cdf_name, tint: list = None):
    r"""Read field named cdf_name in file and convert to velocity distribution
    function.

    Parameters
    ----------
    file_path : str
        Path of the cdf file.
    cdf_name : str
        Name of the target variable in the cdf file.
    tint : list of str, Optional
        Time interval.

    Returns
    -------
    out : xarray.Dataset
        Time series of the velocity distribution function if the target
        specie in the selected time interval.

    """

    tmmode = cdf_name.split("_")[-1]

    if "_dis_" in cdf_name:
        specie = "ions"
    elif "_des_" in cdf_name:
        specie = "electrons"
    else:
        raise AttributeError(
            "Couldn't get the particle species from file name!!",
        )

    # Check time interval type
    # Check time interval
    if tint is None:
        tint = ["1995-10-06T18:50:00.000000000", "2200-10-06T18:50:00.000000000"]
    elif isinstance(tint, (np.ndarray, list)):
        if isinstance(tint[0], np.datetime64):
            tint = datetime642iso8601(np.array(tint))
        elif isinstance(tint[0], str):
            tint = iso86012datetime64(
                np.array(tint),
            )  # to make sure it is ISO8601 ok!!
            tint = datetime642iso8601(np.array(tint))
        else:
            raise TypeError("Values must be in datetime64, or str!!")
    else:
        raise TypeError("tint must be array_like!!")

    # Load CDF file
    file = load(file_path)

    # with CDF(file_path) as file:
    # Get the relevant CDF file information (zVariables)
    z_vars = [z_var[0] for z_var in file.items()]

    # Get the global attributes
    glob_attrs = _pycdfpp_attributes_to_dict(file.attributes)
    glob_attrs = {**glob_attrs, **{"tmmode": tmmode, "species": specie}}

    # Get VDF zVariable attributes
    dist_attrs = _pycdfpp_attributes_to_dict(file[cdf_name].attributes)

    # Get CDF keys to Epoch, energy, azimuthal and elevation angle
    # zVariables
    depends_keys = [dist_attrs[f"DEPEND_{i:d}"] for i in range(4)]

    # Get coordinates attributes
    coords_attrs = {}

    for n, k in zip(["time", "phi", "theta", "energy"], depends_keys):
        coords_attrs[n] = _pycdfpp_attributes_to_dict(file[k].attributes)

    times = _get_epochs(file, cdf_name)

    # If something time is None means that there is nothing interesting
    # in this file so leave!!
    if times["data"] is not None:
        times = times["data"]
    else:
        return None

    dist = np.transpose(file[cdf_name].values, [0, 3, 1, 2])
    phi, theta, energy = [np.squeeze(file[k].values) for k in depends_keys[1:]]

    if tmmode == "brst":
        en0_name = "_".join(
            [
                cdf_name.split("_")[0],
                cdf_name.split("_")[1],
                "energy0",
                cdf_name.split("_")[-1],
            ],
        )
        en1_name = "_".join(
            [
                cdf_name.split("_")[0],
                cdf_name.split("_")[1],
                "energy1",
                cdf_name.split("_")[-1],
            ],
        )

        e_step_table_name = "_".join(
            [
                cdf_name.split("_")[0],
                cdf_name.split("_")[1],
                "steptable_parity",
                cdf_name.split("_")[-1],
            ],
        )

        step_table = file[e_step_table_name].values

        if en0_name not in z_vars:
            if energy.ndim == 1:
                energy0 = energy
                energy1 = energy
            elif energy.shape[0] == 1:
                energy0 = energy[0, :]
                energy1 = energy[0, :]
            else:
                idx_energy0 = np.where(step_table == 0)[0]
                idx_energy1 = np.where(step_table == 1)[0]
                energy0 = energy[idx_energy0[0], :]
                energy1 = energy[idx_energy1[0], :]
        else:
            energy0 = file[en0_name].values
            energy1 = file[en1_name].values

        # Overwrite energy to make sure that energy0 and energy1
        # are used instead
        energy = np.tile(energy0, (len(step_table), 1))
        energy[step_table == 1] = np.tile(energy1, (int(np.sum(step_table)), 1))

    elif tmmode == "fast":
        phi = np.tile(phi, (len(times), 1))

        if energy.ndim == 1:
            energy0 = energy
            energy1 = energy
        elif energy.shape[0] == 1:
            energy0 = energy[0, :]
            energy1 = energy[0, :]
        else:
            energy0 = energy[1, :]
            energy1 = energy[0, :]

        step_table = np.zeros(len(times))

    else:
        raise ValueError("Invalid sampling mode!!")

    d_en_name = "_".join(
        [
            cdf_name.split("_")[0],
            cdf_name.split("_")[1],
            "energy_delta",
            cdf_name.split("_")[-1],
        ],
    )

    if d_en_name in z_vars:
        glob_attrs["delta_energy_plus"] = file[d_en_name].values
        glob_attrs["delta_energy_minus"] = file[d_en_name].values
    else:
        glob_attrs["delta_energy_plus"] = None
        glob_attrs["delta_energy_minus"] = None

    out = ts_skymap(
        times,
        dist,
        energy,
        phi,
        theta,
        energy0=energy0,
        energy1=energy1,
        esteptable=step_table,
        attrs=dist_attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )

    out = time_clip(out, tint)

    return out
