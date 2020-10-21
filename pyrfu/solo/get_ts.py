#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_ts.py

@author : Louis RICHARD
"""

import bisect
import numpy as np
import xarray as xr

from spacepy import pycdf
from dateutil import parser


def get_ts(file_path="", cdf_name="", trange=None):
    """
    Read field named cdf_name in file and convert to time series

    Parameters :
        file_path : str
            Path of the cdf file

        cdf_name : str
            Name of the target variable in the cdf file

        trange : list
            Time interval

    Returns :
        out : DataArray
            Time series of the target variable in the selected time interval

    """

    if not file_path or not cdf_name or trange is None:
        raise ValueError("get_ts requires at least 3 arguments")

    assert isinstance(file_path, str)
    assert isinstance(cdf_name, str)

    x, y, z, w = [{}, {}, {}, {}]

    out_dict = {}

    with pycdf.CDF(file_path) as f:
        depend0_key = f[cdf_name].attrs["DEPEND_0"]
        start_ind, stop_ind = [bisect.bisect_left(f[depend0_key], parser.parse(trange[0])),
                               bisect.bisect_left(f[depend0_key], parser.parse(trange[1]))]

        x["data"], x["attrs"] = [f[depend0_key][start_ind:stop_ind], {}]

        for k in f[depend0_key].attrs.keys():
            x["attrs"][k] = f[depend0_key].attrs[k]
            if isinstance(x["attrs"][k], str) and x["attrs"][k] in f.keys() and not k == "LABLAXIS":
                try:
                    # If array
                    x["attrs"][k] = f[x["attrs"][k]][start_ind:stop_ind, ...]
                except IndexError:
                    # If scalar
                    x["attrs"][k] = f[x["attrs"][k]][...]

        if "DEPEND_1" in f[cdf_name].attrs or "REPRESENTATION_1" in f[cdf_name].attrs:
            try:
                depend1_key = f[cdf_name].attrs["DEPEND_1"]
            except KeyError:
                depend1_key = f[cdf_name].attrs["REPRESENTATION_1"]

            if depend1_key == "x,y,z":
                y["data"], y["attrs"] = [np.array(depend1_key.split(",")), {"LABLAXIS": "comp"}]
            else:
                try:
                    y["data"] = f[depend1_key][start_ind:stop_ind, :]
                except IndexError:
                    y["data"] = f[depend1_key][...]

                # If vector componenents remove magnitude index

                if len(y["data"]) == 4 and all(y["data"] == ["x", "y", "z", "r"]):
                    y["data"] = y["data"][:-1]
                # if y is 2d get only first row assuming that the bins are the same
                elif y["data"].ndim == 2:
                    try:
                        y["data"] = y["data"][0, :]
                    except IndexError:
                        pass

                y["attrs"] = {}

                # Get attributes
                for k in f[depend1_key].attrs.keys():
                    y["attrs"][k] = f[depend1_key].attrs[k]

                    if isinstance(y["attrs"][k], str) and y["attrs"][k] in f.keys():
                        if k not in ["DEPEND_0", "LABLAXIS"]:
                            try:
                                y["attrs"][k] = f[y["attrs"][k]][start_ind:stop_ind, ...]
                            except:
                                y["attrs"][k] = f[y["attrs"][k]][...]
                            # If attrs is 2D get only first row
                            if y["attrs"][k].ndim == 2:
                                try:
                                    y["attrs"][k] = y["attrs"][k][0, :]
                                except IndexError:
                                    pass

                # Remove spaces in label
                try:
                    y["attrs"]["LABLAXIS"] = y["attrs"]["LABLAXIS"].replace(" ", "_")

                    if y["attrs"]["LABLAXIS"] == "Diffential_energy_channels":
                        y["attrs"]["LABLAXIS"] = "Differential_energy_channels"

                except KeyError:
                    y["attrs"]["LABLAXIS"] = "comp"

        elif "afg" in cdf_name or "dfg" in cdf_name:
            y["data"] = ["x", "y", "z"]
            y["attrs"] = {"LABLAXIS": "comp"}

        if "DEPEND_2" in f[cdf_name].attrs or "REPRESENTATION_2" in f[cdf_name].attrs:
            try:
                depend2_key = f[cdf_name].attrs["DEPEND_2"]
            except KeyError:
                depend2_key = f[cdf_name].attrs["REPRESENTATION_2"]

            if depend2_key == "x,y,z":
                z["data"] = np.array(depend2_key.split(","))

                z["attrs"] = {"LABLAXIS": "comp"}
            else:
                z["data"] = f[depend2_key][...]

                z["attrs"] = {}

                for k in f[depend2_key].attrs.keys():
                    z["attrs"][k] = f[depend2_key].attrs[k]

                    if isinstance(z["attrs"][k], str) and z["attrs"][k] in f.keys() and \
                            not k == "DEPEND_0":
                        z["attrs"][k] = f[z["attrs"][k]][start_ind:stop_ind, ...]

                if "LABLAXIS" not in z["attrs"].keys():
                    z["attrs"]["LABLAXIS"] = "comp"

        if "DEPEND_3" in f[cdf_name].attrs or "REPRESENTATION_3" in f[cdf_name].attrs and \
                f[cdf_name].attrs["REPRESENTATION_3"] != "x,y,z":

            try:
                depend3_key = f[cdf_name].attrs["DEPEND_3"]
            except KeyError:
                depend3_key = f[cdf_name].attrs["REPRESENTATION_3"]

            w["data"] = f[depend3_key][...]

            if w["data"].ndim == 2:
                try:
                    w["data"] = w["data"][0, :]
                except IndexError:
                    pass

            w["attrs"] = {}
            for k in f[depend3_key].attrs.keys():
                w["attrs"][k] = f[depend3_key].attrs[k]

                if isinstance(w["attrs"][k], str) and w["attrs"][k] in f.keys() and \
                        not k == "DEPEND_0":
                    w["attrs"][k] = f[w["attrs"][k]][start_ind:stop_ind, ...]

            if "LABLAXIS" not in w["attrs"].keys():
                w["attrs"]["LABLAXIS"] = "comp"

        if "sector_mask" in cdf_name:
            y["data"] = f[f[cdf_name.replace("sector_mask", "intensity")].attrs["DEPEND_1"]][...]

            y["attrs"] = {}

            y_attrs_keys = f[
                f[cdf_name.replace("sector_mask", "intensity")].attrs["DEPEND_1"]].attrs.keys()

            for k in y_attrs_keys:
                y["attrs"][k] = \
                f[f[cdf_name.replace("sector_mask", "intensity")].attrs["DEPEND_1"]].attrs[k]

            y["attrs"]["LABLAXIS"] = y["attrs"]["LABLAXIS"].replace(" ", "_")

        if "edp_dce_sensor" in cdf_name:
            y["data"] = ["x", "y", "z"]

            y["attrs"] = {"LABLAXIS": "comp"}

        out_dict["data"] = f[cdf_name][start_ind:stop_ind, ...]

        if out_dict["data"].ndim == 2 and out_dict["data"].shape[1] == 4:
            out_dict["data"] = out_dict["data"][:, :-1]

        out_dict["attrs"] = {}

        for k in f[cdf_name].attrs:
            out_dict["attrs"][k] = f[cdf_name].attrs[k]

    if x and not y and not z and not w:
        dims, coords = [["time"], [x["data"]]]

        out = xr.DataArray(out_dict["data"], coords=coords, dims=dims, attrs=out_dict["attrs"])
        exec("out." + dims[0] + ".attrs = x['attrs']")

    elif x and y and not z and not w:
        dims, coords = [["time", y["attrs"]["LABLAXIS"]], [x["data"], y["data"]]]

        out = xr.DataArray(out_dict["data"], coords=coords, dims=dims, attrs=out_dict["attrs"])
        exec("out." + dims[0] + ".attrs = x['attrs']")
        exec("out." + dims[1] + ".attrs = y['attrs']")

    elif x and y and z and not w:

        if y["attrs"]["LABLAXIS"] == z["attrs"]["LABLAXIS"]:
            y["attrs"]["LABLAXIS"] = "rcomp"
            z["attrs"]["LABLAXIS"] = "ccomp"

        dims = ["time", y["attrs"]["LABLAXIS"], z["attrs"]["LABLAXIS"]]
        coords = [x["data"], y["data"], z["data"]]

        out = xr.DataArray(out_dict["data"], coords=coords, dims=dims, attrs=out_dict["attrs"])
        exec("out." + dims[0] + ".attrs = x['attrs']")
        exec("out." + dims[1] + ".attrs = y['attrs']")
        exec("out." + dims[2] + ".attrs = z['attrs']")
    elif x and y and z and w:
        if z["attrs"]["LABLAXIS"] == w["attrs"]["LABLAXIS"]:
            z["attrs"]["LABLAXIS"] = "rcomp"
            w["attrs"]["LABLAXIS"] = "ccomp"

        dims = ["time", y["attrs"]["LABLAXIS"], z["attrs"]["LABLAXIS"], w["attrs"]["LABLAXIS"]]
        coords = [x["data"], y["data"], z["data"], w["data"]]

        out = xr.DataArray(out_dict["data"], coords=coords, dims=dims, attrs=out_dict["attrs"])
        exec("out." + dims[0] + ".attrs = x['attrs']")
        exec("out." + dims[1] + ".attrs = y['attrs']")
        exec("out." + dims[2] + ".attrs = z['attrs']")
        exec("out." + dims[3] + ".attrs = w['attrs']")

    else:
        raise NotImplementedError

    return out
