#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.


import numpy as np
import xarray as xr

from cdflib import CDF, cdfepoch
from dateutil import parser as date_parser

from ..pyrf import datetime_to_tt2000


def get_epochs(file, cdf_name, tint):
    """Read time data and its attributes associated to the variable named `cdf_name` in `file`.

    Parameters
    ----------
    file : cdflib.cdfread.CDF
        cdf file

    cdf_name : str
        Name of the target variable in the cdf file

    tint : list of str
        Time interval

    Returns
    -------
    out : dict
        Hash table with the data of the time and its meta-data.

    """

    depend0_key = file.varattsget(cdf_name)["DEPEND_0"]

    out = {"data": file.varget(depend0_key, starttime=tint[0], endtime=tint[1])}

    if file.varinq(depend0_key)["Data_Type_Description"] == "CDF_TIME_TT2000":
        out["data"] = cdfepoch.to_datetime(out["data"], to_np=True)

    out["atts"] = {}

    for k in file.varattsget(depend0_key):
        out["atts"][k] = file.varattsget(depend0_key)[k]
        if isinstance(out["atts"][k], str) and out["atts"][k] in file.cdf_info()["zVariables"] \
                and not k == "LABLAXIS":
            if out["atts"][k] not in ["Epoch_MINUS", "Epoch_PLUS"]:

                try:
                    # If array
                    out["atts"][k] = file.varget(out["atts"][k], starttime=tint[0], endtime=tint[1])
                except IndexError:
                    # If scalar
                    out["atts"][k] = file.varget(out["atts"][k])

    return out


def get_depend_attributes(file, depend_key, tint):
    """Get dependy attributes

    Parameters
    ----------
    file : cdflib.cdfread.CDF
        cdf file

    depend_key : str
        Name of the variable depency in the cdf file

    tint : list of str
        Time interval

    Returns
    -------
    attributes : dict
        Hash table with the attributes of the dependency.

    """

    attributes = {}

    for k in file.varattsget(depend_key):
        attributes[k] = file.varattsget(depend_key)[k]

        if isinstance(attributes[k], str) and attributes[k] in file.cdf_info()["zVariables"] \
                and not k == "DEPEND_0":
            try:
                attributes[k] = file.varget(attributes[k], starttime=tint[0], endtime=tint[1])
            except IndexError:
                attributes[k] = file.varget(attributes[k])
            # If atts is 2D get only first row
            if attributes[k].ndim == 2:
                try:
                    attributes[k] = attributes[k][0, :]
                except IndexError:
                    pass

    # Remove spaces in label
    try:
        attributes["LABLAXIS"] = attributes["LABLAXIS"].replace(" ", "_")

        if attributes["LABLAXIS"] == "Diffential_energy_channels":
            attributes["LABLAXIS"] = "Differential_energy_channels"

    except (KeyError, AttributeError):
        attributes["LABLAXIS"] = "comp"

    return attributes


def get_depend(file, cdf_name, tint, depend_num=1):
    """
    Read the `depend_num`th dependency data and its attributes associated to the variable named
    `cdf_name` in `file`.

    Parameters
    ----------
    file : cdflib.cdfread.CDF
        cdf file

    cdf_name : str
        Name of the target variable in the cdf file

    tint : list of str
        Time interval

    depend_num : int
        Index of the dependency

    Returns
    -------
    out : dict
        Hash table with the data of the dependency and its meta-data.

    """

    out = {}

    try:
        depend_key = file.varattsget(cdf_name)["DEPEND_{:d}".format(depend_num)]
    except KeyError:
        depend_key = file.varattsget(cdf_name)["REPRESENTATION_{:d}".format(depend_num)]

    if depend_key == "x,y,z":
        out["data"] = np.array(depend_key.split(","))

        out["atts"] = {"LABLAXIS": "comp"}
    else:
        try:
            out["data"] = file.varget(depend_key, starttime=tint[0], endtime=tint[1])
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

        out["atts"] = get_depend_attributes(file, depend_key, tint)

    return out


def get_ts(file_path, cdf_name, tint):
    """
    Read field named cdf_name in file and convert to time series.

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
    tint = list(map(date_parser.parse, tint))
    tint = list(map(datetime_to_tt2000, tint))
    tint = list(map(cdfepoch.parse, tint))

    out_dict = {}
    time, depend_1, depend_2, depend_3 = [{}, {}, {}, {}]

    with CDF(file_path) as file:
        out_dict["atts"] = file.varattsget(cdf_name)

        assert "DEPEND_0" in out_dict["atts"] and "epoch" in out_dict["atts"]["DEPEND_0"].lower()

        time = get_epochs(file, cdf_name, tint)

        if "DEPEND_1" in out_dict["atts"] or "REPRESENTATION_1" in out_dict["atts"]:
            depend_1 = get_depend(file, cdf_name, tint, 1)

        elif "afg" in cdf_name or "dfg" in cdf_name:
            depend_1 = {"data": ["x", "y", "z"], "atts": {"LABLAXIS": "comp"}}

        if "DEPEND_2" in out_dict["atts"] or "REPRESENTATION_2" in out_dict["atts"]:
            depend_2 = get_depend(file, cdf_name, tint, 2)

            if depend_2["atts"]["LABLAXIS"] == depend_1["atts"]["LABLAXIS"]:
                depend_1["atts"]["LABLAXIS"] = "rcomp"
                depend_2["atts"]["LABLAXIS"] = "ccomp"

        if "DEPEND_3" in out_dict["atts"] or "REPRESENTATION_3" in out_dict["atts"]:
            assert out_dict["atts"]["REPRESENTATION_3"] != "x,y,z"

            depend_3 = get_depend(file, cdf_name, tint, 3)

            if depend_3["atts"]["LABLAXIS"] == depend_2["atts"]["LABLAXIS"]:
                depend_2["atts"]["LABLAXIS"] = "rcomp"
                depend_3["atts"]["LABLAXIS"] = "ccomp"

        if "sector_mask" in cdf_name:
            depend_1_key = file.varattsget(cdf_name.replace("sector_mask", "intensity"))["DEPEND_1"]

            depend_1["data"] = file.varget(depend_1_key)
            depend_1["atts"] = file.varattsget(depend_1_key)

            depend_1["atts"]["LABLAXIS"] = depend_1["atts"]["LABLAXIS"].replace(" ", "_")

        if "edp_dce_sensor" in cdf_name:
            depend_1["data"] = ["x", "y", "z"]
            depend_1["atts"] = {"LABLAXIS": "comp"}

        out_dict["data"] = file.varget(cdf_name, starttime=tint[0], endtime=tint[1])

        if out_dict["data"].ndim == 2 and out_dict["data"].shape[1] == 4:
            out_dict["data"] = out_dict["data"][:, :-1]

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

        dims = ["time", depend_1["atts"]["LABLAXIS"], depend_2["atts"]["LABLAXIS"]]
        coords_data = [time["data"], depend_1["data"], depend_2["data"]]
        coords_atts = [time["atts"], depend_1["atts"], depend_2["atts"]]

    elif time and depend_1 and depend_2 and depend_3:
        if depend_2["atts"]["LABLAXIS"] == depend_3["atts"]["LABLAXIS"]:
            depend_2["atts"]["LABLAXIS"] = "rcomp"
            depend_3["atts"]["LABLAXIS"] = "ccomp"

        dims = ["time", depend_1["atts"]["LABLAXIS"], depend_2["atts"]["LABLAXIS"],
                depend_3["atts"]["LABLAXIS"]]
        coords_data = [time["data"], depend_1["data"], depend_2["data"], depend_3["data"]]
        coords_atts = [time["atts"], depend_1["atts"], depend_2["atts"], depend_3["atts"]]

    else:
        raise NotImplementedError

    out = xr.DataArray(out_dict["data"], coords=coords_data, dims=dims, attrs=out_dict["atts"])

    for dim, coord_atts in zip(dims, coords_atts):
        out[dim].attrs = coord_atts

    return out
