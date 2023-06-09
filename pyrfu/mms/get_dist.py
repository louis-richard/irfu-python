#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in import
import re
import warnings

# 3rd party imports
import numpy as np
from cdflib import CDF, cdfepoch

from ..pyrf.cdfepoch2datetime64 import cdfepoch2datetime64

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.extend_tint import extend_tint
from ..pyrf.iso86012datetime64 import iso86012datetime64
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.3.26"
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


def _shift_epochs(file, epoch):
    r"""Shift times for particles."""

    epoch_shifted = epoch["data"].copy()

    try:
        delta_minus_var = {
            "data": file.varget(epoch["attrs"]["DELTA_MINUS_VAR"]),
            "attrs": file.varget(epoch["attrs"]["DELTA_MINUS_VAR"]),
        }
        delta_plus_var = {
            "data": file.varget(epoch["attrs"]["DELTA_PLUS_VAR"]),
            "attrs": file.varget(epoch["attrs"]["DELTA_PLUS_VAR"]),
        }

        delta_vars = [delta_minus_var, delta_plus_var]
        flags_vars = [1e3, 1e3]  # Time scaling conversion flags

        for i, delta_var in enumerate(delta_vars):
            if isinstance(delta_var["attrs"], dict) and "UNITS" in delta_var["attrs"]:
                if delta_var["attrs"]["UNITS"].lower() == "s":
                    flags_vars[i] = 1e3
                elif delta_var["attrs"]["UNITS"].lower() == "ms":
                    flags_vars[i] = 1e0
                else:
                    message = " units are not clear, assume s"
                    warnings.warn(message)
            else:
                message = "Epoch_plus_var/Epoch_minus_var units are not clear, assume s"
                warnings.warn(message)

        flag_minus, flag_plus = flags_vars
        t_offset = (
            delta_plus_var["data"] * flag_plus - delta_minus_var["data"] * flag_minus
        )
        t_offset = np.timedelta64(int(np.round(t_offset, 1) * 1e6 / 2), "ns")
        t_diff = (
            delta_plus_var["data"] * flag_plus - delta_minus_var["data"] * flag_minus
        )
        t_diff = np.timedelta64(int(np.round(t_diff, 1) * 1e6 / 2), "ns")
        t_diff_data = np.median(np.diff(epoch["data"])) / 2

        if t_diff_data != np.mean(t_diff):
            t_offset = t_diff_data

        epoch_shifted += t_offset

        return {"data": epoch_shifted, "attrs": epoch["attrs"]}

    except KeyError:
        return {"data": epoch_shifted, "attrs": epoch["attrs"]}


def _get_epochs(file, cdf_name, tint):
    r"""Get epochs form cdf and shift if needed."""

    depend0_key = file.varattsget(cdf_name)["DEPEND_0"]

    out = {
        "data": file.varget(depend0_key, starttime=tint[0], endtime=tint[1]),
    }

    if file.varinq(depend0_key).Data_Type_Description == "CDF_TIME_TT2000":
        try:
            out["data"] = cdfepoch2datetime64(out["data"])

            # Get epoch attributes
            out["attrs"] = file.varattsget(depend0_key)

            # Shift times if particle data
            is_part = re.search("^mms[1-4]_d[ei]s_", cdf_name)
            is_part = is_part or re.search("^mms[1-4]_hpca_", cdf_name)

            if is_part:
                out = _shift_epochs(file, out)

        except TypeError:
            pass

    return out


def get_dist(file_path, cdf_name, tint):
    r"""Read field named cdf_name in file and convert to velocity distribution
    function.

    Parameters
    ----------
    file_path : str
        Path of the cdf file.
    cdf_name : str
        Name of the target variable in the cdf file.
    tint : list of str
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

    tint_org = tint
    tint = extend_tint(tint, [-1, 1])
    tint = list(datetime642iso8601(iso86012datetime64(np.array(tint))))
    tint = np.stack(list(map(cdfepoch.parse, tint)))

    with CDF(file_path) as file:
        # Get the relevant CDF file information (zVariables)
        cdf_infos = file.cdf_info()
        z_vars = cdf_infos.zVariables

        # Get the global attributes
        glob_attrs = file.globalattsget()
        glob_attrs = {**glob_attrs, **{"tmmode": tmmode, "species": specie}}

        # Get VDF zVariable attributes
        dist_attrs = file.varattsget(cdf_name)

        # Get CDF keys to Epoch, energy, azimuthal and elevation angle
        # zVariables
        dpnd_keys = [dist_attrs[f"DEPEND_{i:d}"] for i in range(4)]
        _, depend1_key, depend2_key, depend3_key = dpnd_keys

        # Get coordinates attributes
        coords_names = ["time", "phi", "theta", "energy"]
        coords_attrs = {n: file.varattsget(k) for n, k in zip(coords_names, dpnd_keys)}

        times = _get_epochs(file, cdf_name, tint)

        # If something time is None means that there is nothing interesting
        # in this file so leave!!
        if times["data"] is not None:
            times = times["data"]
        else:
            return None

        dist = file.varget(cdf_name, starttime=tint[0], endtime=tint[1])
        dist = np.transpose(dist, [0, 3, 1, 2])
        phi = file.varget(depend1_key, starttime=tint[0], endtime=tint[1])
        theta = file.varget(depend2_key)
        energy = file.varget(depend3_key, starttime=tint[0], endtime=tint[1])

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

            step_table = file.varget(
                e_step_table_name,
                starttime=tint[0],
                endtime=tint[1],
            )

            if en0_name not in z_vars:
                if energy.ndim == 1:
                    energy0 = energy
                    energy1 = energy
                elif energy.shape[0] == 1:
                    energy0 = energy[0, :]
                    energy1 = energy[0, :]
                else:
                    energy0 = energy[1, :]
                    energy1 = energy[0, :]
            else:
                energy0 = file.varget(en0_name)
                energy1 = file.varget(en1_name)

            # Overwrite energy to make sure that energy0 and energy1
            # are used instead
            energy = None

        elif tmmode == "fast":
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
            glob_attrs["delta_energy_plus"] = file.varget(
                d_en_name,
                starttime=tint[0],
                endtime=tint[1],
            )
            glob_attrs["delta_energy_minus"] = file.varget(
                d_en_name,
                starttime=tint[0],
                endtime=tint[1],
            )
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

    out = time_clip(out, tint_org)

    return out
