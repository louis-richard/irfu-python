#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from cdflib import CDF, cdfepoch

# Local imports
from ..pyrf import ts_skymap, iso86012datetime64, datetime642ttns

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


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

    tint = list(datetime642ttns(iso86012datetime64(np.array(tint))))

    with CDF(file_path) as file:
        if tmmode == "brst":
            depend0_key = file.varattsget(cdf_name)["DEPEND_0"]
            depend1_key = file.varattsget(cdf_name)["DEPEND_1"]
            depend2_key = file.varattsget(cdf_name)["DEPEND_2"]
            depend3_key = file.varattsget(cdf_name)["DEPEND_3"]

            times = file.varget(depend0_key, starttime=tint[0], endtime=tint[1])
            times = cdfepoch.to_datetime(times, to_np=True)

            if not times.size:
                return None

            dist = file.varget(cdf_name, starttime=tint[0], endtime=tint[1])
            dist = np.transpose(dist, [0, 3, 1, 2])
            phi = file.varget(depend1_key, starttime=tint[0], endtime=tint[1])
            theta = file.varget(depend2_key)
            energy = file.varget(depend3_key, starttime=tint[0], endtime=tint[1])

            en0_name = "_".join(
                [
                    cdf_name.split("_")[0],
                    cdf_name.split("_")[1],
                    "energy0",
                    cdf_name.split("_")[-1],
                ]
            )
            en1_name = "_".join(
                [
                    cdf_name.split("_")[0],
                    cdf_name.split("_")[1],
                    "energy1",
                    cdf_name.split("_")[-1],
                ]
            )
            d_en_name = "_".join(
                [
                    cdf_name.split("_")[0],
                    cdf_name.split("_")[1],
                    "energy_delta",
                    cdf_name.split("_")[-1],
                ]
            )
            e_step_table_name = "_".join(
                [
                    cdf_name.split("_")[0],
                    cdf_name.split("_")[1],
                    "steptable_parity",
                    cdf_name.split("_")[-1],
                ]
            )

            step_table = file.varget(
                e_step_table_name, starttime=tint[0], endtime=tint[1]
            )
            if d_en_name in file.cdf_info()["zVariables"]:
                delta_plus_var = file.varget(
                    d_en_name, starttime=tint[0], endtime=tint[1]
                )
                delta_minus_var = file.varget(
                    d_en_name, starttime=tint[0], endtime=tint[1]
                )

            if en0_name not in file.cdf_info()["zVariables"]:
                energy0 = energy[1, :]
                energy1 = energy[0, :]
            else:
                energy0 = file.varget(en0_name)
                energy1 = file.varget(en1_name)

            res = ts_skymap(
                times,
                dist,
                None,
                phi,
                theta,
                energy0=energy0,
                energy1=energy1,
                esteptable=step_table,
            )

            if "delta_plus_var" in locals() and "delta_minus_var" in locals():
                res.attrs["delta_energy_minus"] = delta_minus_var
                res.attrs["delta_energy_plus"] = delta_plus_var

            res.attrs = {**res.attrs, **file.varattsget(cdf_name)}

        elif tmmode == "fast":
            depend0_key = file.varattsget(cdf_name)["DEPEND_0"]
            depend1_key = file.varattsget(cdf_name)["DEPEND_1"]
            depend2_key = file.varattsget(cdf_name)["DEPEND_2"]
            depend3_key = file.varattsget(cdf_name)["DEPEND_3"]

            times = file.varget(depend0_key, starttime=tint[0], endtime=tint[1])

            dist = file.varget(cdf_name, starttime=tint[0], endtime=tint[1])
            dist = np.transpose(dist, [0, 3, 1, 2])
            phi = file.varget(depend1_key)
            theta = file.varget(depend2_key)
            energy = file.varget(depend3_key, starttime=tint[0], endtime=tint[1])
            res = ts_skymap(times, dist, energy, phi, theta)

            for k in file.varattsget(cdf_name):
                res.attrs[k] = file.varattsget(cdf_name)[k]

        for k in file.cdf_info():
            res.attrs[k] = file.cdf_info()[k]

        res.attrs["tmmode"] = tmmode
        if "_dis_" in cdf_name:
            res.attrs["species"] = "ions"
        else:
            res.attrs["species"] = "electrons"
    return res
