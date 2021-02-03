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

    depend0, depend1, depend2, depend3 = [{}, {}, {}, {}]
    out_dict = {}

    with CDF(file_path) as file:
        depend0_key = file.varattsget(cdf_name)["DEPEND_0"]

        depend0["data"] = file.varget(depend0_key, starttime=tint[0], endtime=tint[1])

        if file.varinq(depend0_key)["Data_Type_Description"] == "CDF_TIME_TT2000":
            depend0["data"] = cdfepoch.to_datetime(depend0["data"], to_np=True)

        depend0["atts"] = {}

        for k in file.varattsget(depend0_key):
            depend0["atts"][k] = file.varattsget(depend0_key)[k]
            if isinstance(depend0["atts"][k], str) \
                    and depend0["atts"][k] in file.cdf_info()["zVariables"] and not k == "LABLAXIS":
                if depend0["atts"][k] not in ["Epoch_MINUS", "Epoch_PLUS"]:
                    try:
                        # If array
                        depend0["atts"][k] = file.varget(depend0["atts"][k],
                                                         starttime=tint[0], endtime=tint[1])
                    except IndexError:
                        # If scalar
                        depend0["atts"][k] = file.varget(depend0["atts"][k])

        if "DEPEND_1" in file.varattsget(cdf_name) \
                or "REPRESENTATION_1" in file.varattsget(cdf_name):
            try:
                depend1_key = file.varattsget(cdf_name)["DEPEND_1"]
            except KeyError:
                depend1_key = file.varattsget(cdf_name)["REPRESENTATION_1"]

            if depend1_key == "x,y,z":
                depend1["data"] = np.array(depend1_key.split(","))
                depend1["atts"] = {"LABLAXIS": "comp"}
            else:
                try:
                    depend1["data"] = file.varget(depend1_key, starttime=tint[0], endtime=tint[1])
                except IndexError:
                    depend1["data"] = file.varget(depend1_key)

                if len(depend1["data"]) == 1:
                    depend1["data"] = depend1["data"][0]

                # If vector components remove magnitude index

                if len(depend1["data"]) == 4 and all(depend1["data"] == ["x", "y", "z", "r"]):
                    depend1["data"] = depend1["data"][:-1]
                # if y is 2d get only first row assuming that the bins are the same
                elif depend1["data"].ndim == 2:
                    if len(depend1["data"].flatten()) == 3:
                        depend1["data"] = depend1["data"].flatten()
                    else:
                        try:
                            depend1["data"] = depend1["data"][0, :]
                        except IndexError:
                            pass

                depend1["atts"] = {}

                # Get attributes
                for k in file.varattsget(depend1_key):
                    depend1["atts"][k] = file.varattsget(depend1_key)[k]

                    if isinstance(depend1["atts"][k], str) \
                            and depend1["atts"][k] in file.cdf_info()["zVariables"]:
                        if k not in ["DEPEND_0", "LABLAXIS"]:
                            try:
                                depend1["atts"][k] = file.varget(depend1["atts"][k],
                                                                 starttime=tint[0], endtime=tint[1])
                            except IndexError:
                                depend1["atts"][k] = file.varget(depend1["atts"][k])
                            # If atts is 2D get only first row
                            if depend1["atts"][k].ndim == 2:
                                try:
                                    depend1["atts"][k] = depend1["atts"][k][0, :]
                                except IndexError:
                                    pass

                # Remove spaces in label
                try:
                    depend1["atts"]["LABLAXIS"] = depend1["atts"]["LABLAXIS"].replace(" ", "_")

                    if depend1["atts"]["LABLAXIS"] == "Diffential_energy_channels":
                        depend1["atts"]["LABLAXIS"] = "Differential_energy_channels"

                except KeyError:
                    depend1["atts"]["LABLAXIS"] = "comp"

        elif "afg" in cdf_name or "dfg" in cdf_name:
            depend1["data"] = ["x", "y", "z"]
            depend1["atts"] = {"LABLAXIS": "comp"}

        if "DEPEND_2" in file.varattsget(cdf_name) \
                or "REPRESENTATION_2" in file.varattsget(cdf_name):
            try:
                depend2_key = file.varattsget(cdf_name)["DEPEND_2"]
            except KeyError:
                depend2_key = file.varattsget(cdf_name)["REPRESENTATION_2"]

            if depend2_key == "x,y,z":
                depend2["data"] = np.array(depend2_key.split(","))

                depend2["atts"] = {"LABLAXIS": "comp"}
            else:
                depend2["data"] = file.varget(depend2_key)

                if len(depend2["data"]) == 1:
                    depend2["data"] = depend2["data"][0]

                depend2["atts"] = {}

                for k in file.varattsget(depend2_key):
                    depend2["atts"][k] = file.varattsget(depend2_key)[k]

                    if isinstance(depend2["atts"][k], str) \
                            and depend2["atts"][k] in file.cdf_info()["zVariables"] \
                            and not k == "DEPEND_0":
                        depend2["atts"][k] = file.varget(depend2["atts"][k],
                                                         starttime=tint[0], endtime=tint[1])

                if "LABLAXIS" not in depend2["atts"].keys():
                    depend2["atts"]["LABLAXIS"] = "comp"

        if "DEPEND_3" in file.varattsget(cdf_name) \
                or "REPRESENTATION_3" in file.varattsget(cdf_name) \
                and file.varattsget(cdf_name)["REPRESENTATION_3"] != "x,y,z":

            try:
                depend3_key = file.varattsget(cdf_name)["DEPEND_3"]
            except KeyError:
                depend3_key = file.varattsget(cdf_name)["REPRESENTATION_3"]

            depend3["data"] = file.varget(depend3_key)

            if len(depend3["data"]) == 1:
                depend3["data"] = depend3["data"][0]

            if depend3["data"].ndim == 2:
                try:
                    depend3["data"] = depend3["data"][0, :]
                except IndexError:
                    pass

            depend3["atts"] = {}
            for k in file.varattsget(depend3_key):
                depend3["atts"][k] = file.varattsget(depend3_key)[k]

                if isinstance(depend3["atts"][k], str) \
                        and depend3["atts"][k] in file.cdf_info()["zVariables"] \
                        and not k == "DEPEND_0":
                    depend3["atts"][k] = file.varget(depend3["atts"][k],
                                                     starttime=tint[0], endtime=tint[1])

            if "LABLAXIS" not in depend3["atts"]:
                depend3["atts"]["LABLAXIS"] = "comp"

        if "sector_mask" in cdf_name:
            depend1_key = file.varattsget(cdf_name.replace("sector_mask", "intensity"))["DEPEND_1"]

            depend1["data"] = file.varget(depend1_key)
            depend1["atts"] = file.varattsget(depend1_key)

            depend1["atts"]["LABLAXIS"] = depend1["atts"]["LABLAXIS"].replace(" ", "_")

        if "edp_dce_sensor" in cdf_name:
            depend1["data"] = ["x", "y", "z"]
            depend1["atts"] = {"LABLAXIS": "comp"}

        out_dict["data"] = file.varget(cdf_name, starttime=tint[0], endtime=tint[1])

        if out_dict["data"].ndim == 2 and out_dict["data"].shape[1] == 4:
            out_dict["data"] = out_dict["data"][:, :-1]

        out_dict["atts"] = file.varattsget(cdf_name)

    if depend0 and not depend1 and not depend2 and not depend3:
        dims = ["time"]
        coords_data = [depend0["data"]]
        coords_atts = [depend0["atts"]]

    elif depend0 and depend1 and not depend2 and not depend3:
        dims = ["time", depend1["atts"]["LABLAXIS"]]
        coords_data = [depend0["data"], depend1["data"]]
        coords_atts = [depend0["atts"], depend1["atts"]]

    elif depend0 and depend1 and depend2 and not depend3:
        if depend1["atts"]["LABLAXIS"] == depend2["atts"]["LABLAXIS"]:
            depend1["atts"]["LABLAXIS"] = "rcomp"
            depend2["atts"]["LABLAXIS"] = "ccomp"

        dims = ["time", depend1["atts"]["LABLAXIS"], depend2["atts"]["LABLAXIS"]]
        coords_data = [depend0["data"], depend1["data"], depend2["data"]]
        coords_atts = [depend0["atts"], depend1["atts"], depend2["atts"]]

    elif depend0 and depend1 and depend2 and depend3:
        if depend2["atts"]["LABLAXIS"] == depend3["atts"]["LABLAXIS"]:
            depend2["atts"]["LABLAXIS"] = "rcomp"
            depend3["atts"]["LABLAXIS"] = "ccomp"

        dims = ["time", depend1["atts"]["LABLAXIS"],
                depend2["atts"]["LABLAXIS"], depend3["atts"]["LABLAXIS"]]
        coords_data = [depend0["data"], depend1["data"], depend2["data"], depend3["data"]]
        coords_atts = [depend0["atts"], depend1["atts"], depend2["atts"], depend3["atts"]]

    else:
        raise NotImplementedError

    out = xr.DataArray(out_dict["data"], coords=coords_data, dims=dims, attrs=out_dict["atts"])

    for dim, coord_atts in zip(dims, coords_atts):
        out[dim].attrs = coord_atts

    return out
