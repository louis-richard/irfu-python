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

from cdflib import CDF, cdfepoch
from dateutil import parser as date_parser

from ..pyrf import ts_skymap, datetime_to_tt2000


def get_dist(file_path, cdf_name, tint):
    """Read field named cdf_name in file and convert to velocity distribution function.

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
        Time series of the velocity distribution function if the target specie in the selected time
        interval.

    """

    tmmode = cdf_name.split("_")[-1]

    tint = list(map(date_parser.parse, tint))
    tint = list(map(datetime_to_tt2000, tint))
    tint = list(map(cdfepoch.parse, tint))

    with CDF(file_path) as f:
        if tmmode == "brst":
            depend0_key = f.varattsget(cdf_name)["DEPEND_0"]
            depend1_key = f.varattsget(cdf_name)["DEPEND_1"]
            depend2_key = f.varattsget(cdf_name)["DEPEND_2"]
            depend3_key = f.varattsget(cdf_name)["DEPEND_3"]

            t = f.varget(depend0_key, starttime=tint[0], endtime=tint[1])
            t = cdfepoch.to_datetime(t, to_np=True)

            if not t.size:
                return None

            dist = f.varget(cdf_name, starttime=tint[0], endtime=tint[1])
            dist = np.transpose(dist, [0, 3, 1, 2])
            ph = f.varget(depend1_key, starttime=tint[0], endtime=tint[1])
            th = f.varget(depend2_key)
            en = f.varget(depend3_key, starttime=tint[0], endtime=tint[1])

            en0_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1], "energy0",
                                 cdf_name.split("_")[-1]])
            en1_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1], "energy1",
                                 cdf_name.split("_")[-1]])
            d_en_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1], "energy_delta",
                                  cdf_name.split("_")[-1]])
            e_step_table_name = "_".join([cdf_name.split("_")[0], cdf_name.split("_")[1],
                                          "steptable_parity", cdf_name.split("_")[-1]])

            step_table = f.varget(e_step_table_name, starttime=tint[0], endtime=tint[1])
            if d_en_name in f.cdf_info()["zVariables"]:
                delta_plus_var = f.varget(d_en_name, starttime=tint[0], endtime=tint[1])
                delta_minus_var = f.varget(d_en_name, starttime=tint[0], endtime=tint[1])

            if en0_name not in f.cdf_info()["zVariables"]:
                energy0 = en[1, :]
                energy1 = en[0, :]
            else:
                energy0 = f.varget(en0_name)
                energy1 = f.varget(en1_name)

            res = ts_skymap(t, dist, None, ph, th, energy0=energy0, energy1=energy1,
                            esteptable=step_table)

            if "delta_plus_var" in locals() and "delta_minus_var" in locals():
                res.attrs["delta_energy_minus"] = delta_minus_var
                res.attrs["delta_energy_plus"] = delta_plus_var

            res.attrs = {**res.attrs, **f.varattsget(cdf_name)}

            for k in f.cdf_info():
                res.attrs[k] = f.cdf_info()[k]

            res.attrs["tmmode"] = tmmode
            if "_dis_" in cdf_name:
                res.attrs["species"] = "ions"
            else:
                res.attrs["species"] = "electrons"

        elif tmmode == "fast":
            depend0_key = f.varattsget(cdf_name)["DEPEND_0"]
            depend1_key = f.varattsget(cdf_name)["DEPEND_1"]
            depend2_key = f.varattsget(cdf_name)["DEPEND_2"]
            depend3_key = f.varattsget(cdf_name)["DEPEND_3"]

            t = f.varget(depend0_key, starttime=tint[0], endtime=tint[1])

            dist = f.varget(cdf_name, starttime=tint[0], endtime=tint[1])
            dist = np.transpose(dist, [0, 3, 1, 2])
            ph = f.varget(depend1_key)
            th = f.varget(depend2_key)
            en = f.varget(depend3_key, starttime=tint[0], endtime=tint[1])
            res = ts_skymap(t, dist, en, ph, th)

            for k in f.varattsget(cdf_name):
                res.attrs[k] = f.varattsget(cdf_name)[k]

            for k in f.cdf_info():
                res.attrs[k] = f.cdf_info()[k]

            res.attrs["tmmode"] = tmmode
            if "_dis_" in cdf_name:
                res.attrs["species"] = "ions"
            else:
                res.attrs["species"] = "electrons"
    return res
