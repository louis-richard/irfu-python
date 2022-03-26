#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from cdflib import CDF, cdfepoch

# Local imports
from ..pyrf import datetime642iso8601, iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _get_epochs(file, cdf_name, tint):
    depend0_key = file.varattsget(cdf_name)["DEPEND_0"]

    out = {"data": file.varget(depend0_key,
                               starttime=tint[0], endtime=tint[1])}

    if file.varinq(depend0_key)["Data_Type_Description"] == "CDF_TIME_TT2000":
        try:
            out["data"] = cdfepoch.to_datetime(out["data"], to_np=True)
        except TypeError:
            pass

    out["atts"] = file.varattsget(depend0_key)

    return out


def _get_depend_attributes(file, depend_key):
    attributes = file.varattsget(depend_key)

    # Remove spaces in label
    try:
        attributes["LABLAXIS"] = attributes["LABLAXIS"].replace(" ", "_")

        if attributes["LABLAXIS"] == "Diffential_energy_channels":
            attributes["LABLAXIS"] = "Differential_energy_channels"

    except (KeyError, AttributeError):
        attributes["LABLAXIS"] = "comp"

    return attributes


def _get_depend(file, cdf_name, tint, dep_num=1):
    out = {}

    try:
        depend_key = file.varattsget(cdf_name)[f"DEPEND_{dep_num:d}"]
    except KeyError:
        depend_key = file.varattsget(cdf_name)[f"REPRESENTATION_{dep_num:d}"]

    if depend_key == "x,y,z":
        out["data"] = np.array(depend_key.split(","))

        out["atts"] = {"LABLAXIS": "comp"}
    else:
        try:
            out["data"] = file.varget(depend_key, starttime=tint[0],
                                      endtime=tint[1])
        except IndexError:
            out["data"] = file.varget(depend_key)

        out["data"] = file.varget(depend_key)

        if len(out["data"]) == 1:
            out["data"] = out["data"][0]

        if len(out["data"]) == 4 and all(out["data"] == ["x", "y", "z", "r"]):
            out["data"] = out["data"][:-1]

        elif out["data"].ndim == 2:
            if len(out["data"].flatten()) == 3:
                out["data"] = out["data"].flatten()
            else:
                try:
                    out["data"] = out["data"][0, :]
                except IndexError:
                    pass

        out["atts"] = _get_depend_attributes(file, depend_key)

    return out


def get_ts(file_path, cdf_name, tint):
    r"""Reads field named cdf_name in file and convert to time series.

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
    out : xarray.DataArray
        Time series of the target variable in the selected time interval.

    """

    # Convert time interval to epochs
    tint = list(datetime642iso8601(iso86012datetime64(np.array(tint))))
    tint = list(map(cdfepoch.parse, tint))

    out_dict = {}
    time, depend_1, depend_2, depend_3 = [{}, {}, {}, {}]

    with CDF(file_path) as file:
        attrs_ = file.varattsget(cdf_name)
        out_dict["atts"] = attrs_

        assert "DEPEND_0" in attrs_ and "epoch" in attrs_["DEPEND_0"].lower()

        time = _get_epochs(file, cdf_name, tint)

        if time["data"] is None:
            return None

        if "DEPEND_1" in attrs_ or "REPRESENTATION_1" in attrs_:
            depend_1 = _get_depend(file, cdf_name, tint, 1)

        elif "afg" in cdf_name or "dfg" in cdf_name:
            depend_1 = {"data": ["x", "y", "z"], "atts": {"LABLAXIS": "comp"}}

        if "DEPEND_2" in attrs_ or "REPRESENTATION_2" in attrs_:
            depend_2 = _get_depend(file, cdf_name, tint, 2)

            if depend_2["atts"]["LABLAXIS"] == depend_1["atts"]["LABLAXIS"]:
                depend_1["atts"]["LABLAXIS"] = "rcomp"
                depend_2["atts"]["LABLAXIS"] = "ccomp"

        if "DEPEND_3" in attrs_ or "REPRESENTATION_3" in attrs_:
            if "REPRESENTATION_3" in attrs_:
                assert out_dict["atts"]["REPRESENTATION_3"] != "x,y,z"

            depend_3 = _get_depend(file, cdf_name, tint, 3)

            if depend_3["atts"]["LABLAXIS"] == depend_2["atts"]["LABLAXIS"]:
                depend_2["atts"]["LABLAXIS"] = "rcomp"
                depend_3["atts"]["LABLAXIS"] = "ccomp"

        if "sector_mask" in cdf_name:
            cdf_name_mask = cdf_name.replace("sector_mask", "intensity")
            depend_1_key = file.varattsget(cdf_name_mask)["DEPEND_1"]

            depend_1["data"] = file.varget(depend_1_key)
            depend_1["atts"] = file.varattsget(depend_1_key)

            depend_1["atts"]["LABLAXIS"] = depend_1["atts"][
                "LABLAXIS"].replace(" ", "_")

        if "edp_dce_sensor" in cdf_name:
            depend_1["data"] = ["x", "y", "z"]
            depend_1["atts"] = {"LABLAXIS": "comp"}

        out_dict["data"] = file.varget(cdf_name, starttime=tint[0],
                                       endtime=tint[1])

        if out_dict["data"].ndim == 2 and out_dict["data"].shape[1] == 4:
            out_dict["data"] = out_dict["data"][:, :-1]

    if out_dict["data"].ndim == 2 and not depend_1:
        depend_1["data"] = np.arange(out_dict["data"].shape[1])
        depend_1["atts"] = {"LABLAXIS": "idx"}

    if time and not depend_1 and not depend_2 and not depend_3:
        dims = ["time"]
        coords_data = [time["data"]]
        coords_atts = [time["atts"]]

    elif time and depend_1 and not depend_2 and not depend_3:
        dims = ["time", depend_1["atts"]["LABLAXIS"]]
        coords_data = [time["data"], depend_1["data"]]
        coords_atts = [time["atts"], depend_1["atts"]]

    elif time and depend_1 and depend_2 and not depend_3:
        if depend_1["atts"]["LABLAXIS"] == depend_2["atts"]["LABLAXIS"]:
            depend_1["atts"]["LABLAXIS"] = "rcomp"
            depend_2["atts"]["LABLAXIS"] = "ccomp"

        dims = ["time", depend_1["atts"]["LABLAXIS"],
                depend_2["atts"]["LABLAXIS"]]
        coords_data = [time["data"], depend_1["data"], depend_2["data"]]
        coords_atts = [time["atts"], depend_1["atts"], depend_2["atts"]]

    elif time and depend_1 and depend_2 and depend_3:
        if depend_2["atts"]["LABLAXIS"] == depend_3["atts"]["LABLAXIS"]:
            depend_2["atts"]["LABLAXIS"] = "rcomp"
            depend_3["atts"]["LABLAXIS"] = "ccomp"

        dims = ["time", depend_1["atts"]["LABLAXIS"],
                depend_2["atts"]["LABLAXIS"], depend_3["atts"]["LABLAXIS"]]
        coords_data = [time["data"], depend_1["data"], depend_2["data"],
                       depend_3["data"]]
        coords_atts = [time["atts"], depend_1["atts"], depend_2["atts"],
                       depend_3["atts"]]

    else:
        raise NotImplementedError

    out = xr.DataArray(out_dict["data"], coords=coords_data, dims=dims,
                       attrs=out_dict["atts"])

    for dim, coord_atts in zip(dims, coords_atts):
        out[dim].attrs = coord_atts

    return out
