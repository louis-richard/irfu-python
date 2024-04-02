#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in import
import re
import warnings
from typing import Optional, Union

# 3rd party imports
import numpy as np
import xarray as xr
from pycdfpp import DataType, load, to_datetime64
from xarray.core.dataarray import DataArray

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime64 import iso86012datetime64
from ..pyrf.time_clip import time_clip
from .get_variable import _pycdfpp_attributes_to_dict

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _shift_epochs(file, epoch):
    r"""Shift times for particles."""

    epoch_shifted = epoch["data"].copy()

    try:
        delta_minus_key = epoch["attrs"]["DELTA_MINUS_VAR"]
        delta_minus_var = {
            "data": file[delta_minus_key].values,
            "attrs": _pycdfpp_attributes_to_dict(file[delta_minus_key].attributes),
        }
        delta_plus_key = epoch["attrs"]["DELTA_PLUS_VAR"]
        delta_plus_var = {
            "data": file[delta_plus_key].values,
            "attrs": _pycdfpp_attributes_to_dict(file[delta_plus_key].attributes),
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

        t_offset = (np.round(t_offset, 1) * 1e6 / 2).astype("timedelta64[ns]")
        t_diff = (
            delta_plus_var["data"] * flag_plus - delta_minus_var["data"] * flag_minus
        )
        t_diff = (np.round(t_diff, 1) * 1e6 / 2).astype("timedelta64[ns]")
        t_diff_data = np.median(np.diff(epoch["data"])) / 2

        if t_diff_data != np.mean(t_diff):
            t_offset = t_diff_data

        epoch_shifted += t_offset

        return {"data": epoch_shifted, "attrs": epoch["attrs"]}

    except KeyError:
        return {"data": epoch_shifted, "attrs": epoch["attrs"]}


def _get_epochs(file, cdf_name):
    r"""Get epochs form cdf and shift if needed."""

    depend0_key = file[cdf_name].attributes["DEPEND_0"][0]

    out = {
        "data": file[depend0_key].values,
    }

    if file[depend0_key].type == DataType.CDF_TIME_TT2000:
        try:
            out["data"] = to_datetime64(out["data"])

            # Get epoch attributes
            out["attrs"] = _pycdfpp_attributes_to_dict(file[depend0_key].attributes)

            # Shift times if particle data
            is_part = re.search(
                "^mms[1-4]_d[ei]s_",
                cdf_name,
            )  # Is it FPI data?
            is_part = is_part or re.search(
                "^mms[1-4]_hpca_",
                cdf_name,
            )  # Is it HPCA data?

            if is_part:
                out = _shift_epochs(file, out)

        except TypeError:
            pass

    return out


def _get_depend_attributes(file, depend_key):
    attributes = _pycdfpp_attributes_to_dict(file[depend_key].attributes)

    # Remove spaces in label
    try:
        attributes["LABLAXIS"] = attributes["LABLAXIS"].replace(" ", "_")

        if attributes["LABLAXIS"] == "Diffential_energy_channels":
            attributes["LABLAXIS"] = "Differential_energy_channels"

    except (KeyError, AttributeError):
        attributes["LABLAXIS"] = "comp"

    return attributes


def _get_depend(file, cdf_name, dep_num=1):
    out = {}

    if f"DEPEND_{dep_num:d}" in file[cdf_name].attributes:
        depend_key = file[cdf_name].attributes[f"DEPEND_{dep_num:d}"][0]
    elif f"REPRESENTATION_{dep_num:d}" in file[cdf_name].attributes:
        depend_key = file[cdf_name].attributes[f"REPRESENTATION_{dep_num:d}"][0]
    else:
        raise KeyError(f"no DEPEND_{dep_num:d}/REPRESENTATION_{dep_num:d} attributes")

    if depend_key == "x,y,z":
        out["data"] = np.array(depend_key.split(","))

        out["attrs"] = {"LABLAXIS": "comp"}
    else:
        out["data"] = file[depend_key].values

        if len(out["data"]) == 1:
            out["data"] = out["data"][0]

        if len(out["data"]) == 4 and all(
            out["data"].astype(str) == ["x", "y", "z", "r"]
        ):
            out["data"] = out["data"].astype(str)[:-1]

        elif out["data"].ndim == 2:
            if len(out["data"].flatten()) == 3:
                out["data"] = out["data"].flatten()
            else:
                try:
                    out["data"] = out["data"][0, :]
                except IndexError:
                    pass

        out["attrs"] = _get_depend_attributes(file, depend_key)

    return out


def get_ts(
    file_path: Union[str, bytes], cdf_name: str, tint: Optional[list] = None
) -> DataArray:
    r"""Reads field named cdf_name in file and convert to time series.

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
    out : xarray.DataArray
        Time series of the target variable in the selected time interval.

    """

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

    out_dict = {}
    time, depend_1, depend_2, depend_3 = [{}, {}, {}, {}]

    # Load CDF file
    file = load(file_path)

    var_attrs = _pycdfpp_attributes_to_dict(file[cdf_name].attributes)
    glb_attrs = _pycdfpp_attributes_to_dict(file.attributes)
    out_dict["attrs"] = {"GLOBAL": glb_attrs, **var_attrs}
    out_dict["attrs"] = {k: out_dict["attrs"][k] for k in sorted(out_dict["attrs"])}

    assert "DEPEND_0" in var_attrs and "epoch" in var_attrs["DEPEND_0"].lower()

    time = _get_epochs(file, cdf_name)

    if time["data"] is None:
        return None

    if "DEPEND_1" in var_attrs or "REPRESENTATION_1" in var_attrs:
        depend_1 = _get_depend(file, cdf_name, 1)

    elif "afg" in cdf_name or "dfg" in cdf_name:
        depend_1 = {
            "data": ["x", "y", "z"],
            "attrs": {"LABLAXIS": "comp"},
        }

    if "DEPEND_2" in var_attrs or "REPRESENTATION_2" in var_attrs:
        depend_2 = _get_depend(file, cdf_name, 2)

        if depend_2["attrs"]["LABLAXIS"] == depend_1["attrs"]["LABLAXIS"]:
            depend_1["attrs"]["LABLAXIS"] = "rcomp"
            depend_2["attrs"]["LABLAXIS"] = "ccomp"

    if "DEPEND_3" in var_attrs or "REPRESENTATION_3" in var_attrs:
        if "REPRESENTATION_3" in var_attrs:
            assert out_dict["attrs"]["REPRESENTATION_3"] != "x,y,z"

        depend_3 = _get_depend(file, cdf_name, 3)

        if depend_3["attrs"]["LABLAXIS"] == depend_2["attrs"]["LABLAXIS"]:
            depend_2["attrs"]["LABLAXIS"] = "rcomp"
            depend_3["attrs"]["LABLAXIS"] = "ccomp"

    if "sector_mask" in cdf_name:
        cdf_name_mask = cdf_name.replace("sector_mask", "intensity")
        depend_1_key = file.varattsget(cdf_name_mask)["DEPEND_1"]

        depend_1["data"] = file[depend_1_key].values
        depend_1["attrs"] = _pycdfpp_attributes_to_dict(file[depend_1_key].attributes)

        depend_1["attrs"]["LABLAXIS"] = depend_1["attrs"]["LABLAXIS"].replace(" ", "_")

    if "edp_dce_sensor" in cdf_name:
        depend_1["data"] = ["x", "y", "z"]
        depend_1["attrs"] = {"LABLAXIS": "comp"}

    out_dict["data"] = file[cdf_name].values

    if out_dict["data"].ndim == 2 and out_dict["data"].shape[1] == 4:
        out_dict["data"] = out_dict["data"][:, :-1]
        # depend_1["data"] = depend_1["data"][:-1]

    if out_dict["data"].ndim == 2 and not depend_1:
        depend_1["data"] = np.arange(out_dict["data"].shape[1])
        depend_1["attrs"] = {"LABLAXIS": "idx"}

    if time and not depend_1 and not depend_2 and not depend_3:
        dims = ["time"]
        coords_data = [time["data"]]
        coords_attrs = [time["attrs"]]

    elif time and depend_1 and not depend_2 and not depend_3:
        dims = ["time", depend_1["attrs"]["LABLAXIS"]]
        coords_data = [time["data"], depend_1["data"]]
        coords_attrs = [time["attrs"], depend_1["attrs"]]

    elif time and depend_1 and depend_2 and not depend_3:
        dims = [
            "time",
            depend_1["attrs"]["LABLAXIS"],
            depend_2["attrs"]["LABLAXIS"],
        ]
        coords_data = [time["data"], depend_1["data"], depend_2["data"]]
        coords_attrs = [time["attrs"], depend_1["attrs"], depend_2["attrs"]]

    elif time and depend_1 and depend_2 and depend_3:
        dims = [
            "time",
            depend_1["attrs"]["LABLAXIS"],
            depend_2["attrs"]["LABLAXIS"],
            depend_3["attrs"]["LABLAXIS"],
        ]
        coords_data = [
            time["data"],
            depend_1["data"],
            depend_2["data"],
            depend_3["data"],
        ]
        coords_attrs = [
            time["attrs"],
            depend_1["attrs"],
            depend_2["attrs"],
            depend_3["attrs"],
        ]

    else:
        raise NotImplementedError

    out = xr.DataArray(
        out_dict["data"],
        coords=coords_data,
        dims=dims,
        attrs=out_dict["attrs"],
    )

    for dim, coord_attrs in zip(dims, coords_attrs):
        # Sort attributes and set to coordinates attribute
        out[dim].attrs = {k: coord_attrs[k] for k in sorted(coord_attrs)}

    # Time clip to original time interval
    out = time_clip(out, tint)

    return out
