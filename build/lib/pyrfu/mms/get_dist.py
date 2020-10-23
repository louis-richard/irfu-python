#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_dist.py

@author : Louis RICHARD
"""

import bisect
import numpy as np

from spacepy import pycdf
from dateutil import parser

from ..pyrf import ts_skymap


def get_dist(file_path="", cdf_name="", tint=None):
    """Read field named cdf_name in file and convert to velocity distribution function.

    Parameters
    ----------
    file_path : str
        Path of the cdf file.

    cdf_name : str
        Name of the target variable in the cdf file.

    tint : list
        Time interval.

    Returns
    -------
    out : xarray.Dataset
        Time series of the velocity distribution function if the target specie in the selected time
        interval.

    """

    tmmode = cdf_name.split("_")[-1]
    with pycdf.CDF(file_path) as f:
        if tmmode == "brst":

            DEPEND_0 = f[cdf_name].attrs["DEPEND_0"]
            DEPEND_1 = f[cdf_name].attrs["DEPEND_1"]
            DEPEND_2 = f[cdf_name].attrs["DEPEND_2"]
            DEPEND_3 = f[cdf_name].attrs["DEPEND_3"]

            t = f[DEPEND_0][...]
            idx_left = bisect.bisect_left(t, parser.parse(tint[0]))
            idx_right = bisect.bisect_left(t, parser.parse(tint[1]))
            t = t[idx_left:idx_right]
            if not t.size:
                return None

            dist = f[cdf_name][idx_left:idx_right, ...]
            dist = np.transpose(dist, [0, 3, 1, 2])
            ph = f[DEPEND_1][idx_left:idx_right, ...]
            th = f[DEPEND_2][...]
            en = f[DEPEND_3][idx_left:idx_right, ...]

            en0_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1], "energy0",
                                 cdf_name.split("_")[-1]])
            en1_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1], "energy1",
                                 cdf_name.split("_")[-1]])
            d_en_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1], "energy_delta",
                                  cdf_name.split("_")[-1]])
            e_step_table_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1],
                                         "steptable_parity", cdf_name.split("_")[-1]])

            step_table = f[e_step_table_name][idx_left:idx_right, ...]
            if d_en_name in f.keys():
                delta_plus_var = f[d_en_name][idx_left:idx_right, ...]
                delta_minus_var = f[d_en_name][idx_left:idx_right, ...]

            if en0_name not in f.keys():
                energy0 = en[1, :]
                energy1 = en[0, :]
            else:
                energy0 = f[en0_name][...]
                energy1 = f[en1_name][...]

            res = ts_skymap(t, dist, None, ph, th, energy0=energy0, energy1=energy1,
                            esteptable=step_table)

            if "delta_plus_var" in locals() and "delta_minus_var" in locals():
                res.attrs["delta_energy_minus"] = delta_minus_var
                res.attrs["delta_energy_plus"] = delta_plus_var

            for k in f[cdf_name].attrs:
                res.attrs[k] = f[cdf_name].attrs[k]

            for k in f.attrs:
                res.attrs[k] = f.attrs[k]

            res.attrs["tmmode"] = tmmode
            if "_dis_" in cdf_name:
                res.attrs["species"] = "ions"
            else:
                res.attrs["species"] = "electrons"

        elif tmmode == "fast":
            DEPEND_0 = f[cdf_name].attrs["DEPEND_0"]
            DEPEND_1 = f[cdf_name].attrs["DEPEND_1"]
            DEPEND_2 = f[cdf_name].attrs["DEPEND_2"]
            DEPEND_3 = f[cdf_name].attrs["DEPEND_3"]
            t = f[DEPEND_0][...]
            idx_left = bisect.bisect_left(t, parser.parse(tint[0]))
            idx_right = bisect.bisect_left(t, parser.parse(tint[1]))
            t = t[idx_left:idx_right]
            dist = f[cdf_name][idx_left:idx_right, ...]
            dist = np.transpose(dist, [0, 3, 1, 2])
            ph = f[DEPEND_1][...]
            th = f[DEPEND_2][...]
            en = f[DEPEND_3][idx_left:idx_right, ...]
            res = ts_skymap(t, dist, en, ph, th)

            for k in f[cdf_name].attrs:
                res.attrs[k] = f[cdf_name].attrs[k]

            for k in f.attrs:
                res.attrs[k] = f.attrs[k]

            res.attrs["tmmode"] = tmmode
            if "_dis_" in cdf_name:
                res.attrs["species"] = "ions"
            else:
                res.attrs["species"] = "electrons"
    return res
