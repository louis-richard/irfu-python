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

    x, y, z, w = [{}, {}, {}, {}]
    out_dict = {}

    with CDF(file_path) as f:
        depend0_key = f.varattsget(cdf_name)["DEPEND_0"]

        x["data"] = f.varget(depend0_key, starttime=tint[0], endtime=tint[1])

        if f.varinq(depend0_key)["Data_Type_Description"] == "CDF_TIME_TT2000":
            x["data"] = cdfepoch.to_datetime(x["data"], to_np=True)

        x["atts"] = {}

        for k in f.varattsget(depend0_key):
            x["atts"][k] = f.varattsget(depend0_key)[k]
            if isinstance(x["atts"][k], str) and x["atts"][k] in f.cdf_info()["zVariables"] \
                    and not k == "LABLAXIS":
                try:
                    # If array
                    x["atts"][k] = f.varget(x["atts"][k], starttime=tint[0], endtime=tint[1])
                except IndexError:
                    # If scalar
                    x["atts"][k] = f.varget(x["atts"][k])

        if "DEPEND_1" in f.varattsget(cdf_name) or "REPRESENTATION_1" in f.varattsget(cdf_name):
            try:
                depend1_key = f.varattsget(cdf_name)["DEPEND_1"]
            except KeyError:
                depend1_key = f.varattsget(cdf_name)["REPRESENTATION_1"]

            if depend1_key == "x,y,z":
                y["data"], y["atts"] = [np.array(depend1_key.split(",")), {"LABLAXIS": "comp"}]
            else:
                try:
                    y["data"] = f.varget(depend1_key, starttime=tint[0], endtime=tint[1])
                except IndexError:
                    y["data"] = f.varget(depend1_key)

                if len(y["data"]) == 1:
                    y["data"] = y["data"][0]

                # If vector components remove magnitude index

                if len(y["data"]) == 4 and all(y["data"] == ["x", "y", "z", "r"]):
                    y["data"] = y["data"][:-1]
                # if y is 2d get only first row assuming that the bins are the same
                elif y["data"].ndim == 2:
                    try:
                        y["data"] = y["data"][0, :]
                    except IndexError:
                        pass

                y["atts"] = {}

                # Get attributes
                for k in f.varattsget(depend1_key):
                    y["atts"][k] = f.varattsget(depend1_key)[k]

                    if isinstance(y["atts"][k], str) and y["atts"][k] in f.cdf_info()["zVariables"]:
                        if k not in ["DEPEND_0", "LABLAXIS"]:
                            try:
                                y["atts"][k] = f.varget(y["atts"][k], starttime=tint[0],
                                                        endtime=tint[1])
                            except IndexError:
                                y["atts"][k] = f.varget(y["atts"][k])
                            # If atts is 2D get only first row
                            if y["atts"][k].ndim == 2:
                                try:
                                    y["atts"][k] = y["atts"][k][0, :]
                                except IndexError:
                                    pass

                # Remove spaces in label
                try:
                    y["atts"]["LABLAXIS"] = y["atts"]["LABLAXIS"].replace(" ", "_")

                    if y["atts"]["LABLAXIS"] == "Diffential_energy_channels":
                        y["atts"]["LABLAXIS"] = "Differential_energy_channels"

                except KeyError:
                    y["atts"]["LABLAXIS"] = "comp"

        elif "afg" in cdf_name or "dfg" in cdf_name:
            y["data"] = ["x", "y", "z"]
            y["atts"] = {"LABLAXIS": "comp"}

        if "DEPEND_2" in f.varattsget(cdf_name) or "REPRESENTATION_2" in f.varattsget(cdf_name):
            try:
                depend2_key = f.varattsget(cdf_name)["DEPEND_2"]
            except KeyError:
                depend2_key = f.varattsget(cdf_name)["REPRESENTATION_2"]

            if depend2_key == "x,y,z":
                z["data"] = np.array(depend2_key.split(","))

                z["atts"] = {"LABLAXIS": "comp"}
            else:
                z["data"] = f.varget(depend2_key)

                if len(z["data"]) == 1:
                    z["data"] = z["data"][0]

                z["atts"] = {}

                for k in f.varattsget(depend2_key):
                    z["atts"][k] = f.varattsget(depend2_key)[k]

                    if isinstance(z["atts"][k], str) and z["atts"][k] in f.cdf_info()[
                        "zVariables"] and \
                            not k == "DEPEND_0":
                        z["atts"][k] = f.varget(z["atts"][k], starttime=tint[0], endtime=tint[1])

                if "LABLAXIS" not in z["atts"].keys():
                    z["atts"]["LABLAXIS"] = "comp"

        if "DEPEND_3" in f.varattsget(cdf_name) or "REPRESENTATION_3" in f.varattsget(cdf_name) \
                and f.varattsget(cdf_name)["REPRESENTATION_3"] != "x,y,z":

            try:
                depend3_key = f.varattsget(cdf_name)["DEPEND_3"]
            except KeyError:
                depend3_key = f.varattsget(cdf_name)["REPRESENTATION_3"]

            w["data"] = f.varget(depend3_key)

            if len(w["data"]) == 1:
                w["data"] = w["data"][0]

            if w["data"].ndim == 2:
                try:
                    w["data"] = w["data"][0, :]
                except IndexError:
                    pass

            w["atts"] = {}
            for k in f.varattsget(depend3_key):
                w["atts"][k] = f.varattsget(depend3_key)[k]

                if isinstance(w["atts"][k], str) and w["atts"][k] in f.cdf_info()["zVariables"] \
                        and not k == "DEPEND_0":
                    w["atts"][k] = f.varget(w["atts"][k], starttime=tint[0], endtime=tint[1])

            if "LABLAXIS" not in w["atts"]:
                w["atts"]["LABLAXIS"] = "comp"

        if "sector_mask" in cdf_name:
            depend1_key = f.varattsget(cdf_name.replace("sector_mask", "intensity"))["DEPEND_1"]

            y["data"] = f.varget(depend1_key)
            y["atts"] = f.varattsget(depend1_key)

            y["atts"]["LABLAXIS"] = y["atts"]["LABLAXIS"].replace(" ", "_")

        if "edp_dce_sensor" in cdf_name:
            y["data"] = ["x", "y", "z"]
            y["atts"] = {"LABLAXIS": "comp"}

        out_dict["data"] = f.varget(cdf_name, starttime=tint[0], endtime=tint[1])

        if out_dict["data"].ndim == 2 and out_dict["data"].shape[1] == 4:
            out_dict["data"] = out_dict["data"][:, :-1]

        out_dict["atts"] = f.varattsget(cdf_name)

    if x and not y and not z and not w:
        dims = ["time"]
        coords_data = [x["data"]]
        coords_atts = [x["atts"]]

    elif x and y and not z and not w:
        dims = ["time", y["atts"]["LABLAXIS"]]
        coords_data = [x["data"], y["data"]]
        coords_atts = [x["atts"], y["atts"]]

    elif x and y and z and not w:
        if y["atts"]["LABLAXIS"] == z["atts"]["LABLAXIS"]:
            y["atts"]["LABLAXIS"] = "rcomp"
            z["atts"]["LABLAXIS"] = "ccomp"

        dims = ["time", y["atts"]["LABLAXIS"], z["atts"]["LABLAXIS"]]
        coords_data = [x["data"], y["data"], z["data"]]
        coords_atts = [x["atts"], y["atts"], z["atts"]]

    elif x and y and z and w:
        if z["atts"]["LABLAXIS"] == w["atts"]["LABLAXIS"]:
            z["atts"]["LABLAXIS"] = "rcomp"
            w["atts"]["LABLAXIS"] = "ccomp"

        dims = ["time", y["atts"]["LABLAXIS"], z["atts"]["LABLAXIS"], w["atts"]["LABLAXIS"]]
        coords_data = [x["data"], y["data"], z["data"], w["data"]]
        coords_atts = [x["atts"], y["atts"], z["atts"], w["atts"]]

    else:
        raise NotImplementedError

    out = xr.DataArray(out_dict["data"], coords=coords_data, dims=dims, attrs=out_dict["atts"])

    for dim, coord_atts in zip(dims, coords_atts):
        out[dim].attrs = coord_atts

    return out
