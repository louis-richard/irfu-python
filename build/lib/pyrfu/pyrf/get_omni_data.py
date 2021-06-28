#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import urllib
import datetime

# 3rd party imports
import numpy as np
import pandas as pd

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


var_omni_1 = {"b": 13, "avgb": -1, "blat": -1, "blong": -1, "bx": 14,
              "bxgse": 14, "bxgsm": 14, "by": 15, "bygse": 15, "bz": 16,
              "bzgse": 16, "bygsm": 17, "bzgsm": 18, "t": 26, "n": 25,
              "nanp": -1, "v": 21, "vx": 22, "vy": 23, "vz": 24, "vlon": -1,
              "vlat": -1, "p": 27, "e": 28, "beta": 29, "ma": 30, "bsnx": 34,
              "bsny": 35, "bsnz": 36, "ms": 45, "ssn": -1, "dst": -1,
              "ae": 37, "al": 38, "au": 39, "kp": -1, "pc": 44, "f10.7": -1,
              "imfid": 4, "swid": 5, "ts": 9, "rmsts": 10}

var_omni_2 = {"b": 8, "avgb": 9, "blat": 10, "blong": 11, "bx": 12,
              "bxgse": 12, "bxgsm": 12, "by": 13, "bygse": 13, "bz": 14,
              "bzgse": 14, "bygsm": 15, "bzgsm": 16, "t": 22, "n": 23,
              "nanp": 27, "v": 24, "vx": -1, "vy": -1, "vz": -1, "vlon": 25,
              "vlat": 26, "p": 28, "e": 35, "beta": 36, "ma": 37, "bsnx": -1,
              "bsny": -1, "bsnz": -1, "ms": 56, "ssn": 39, "dst": 40,
              "ae": 41, "al": 52, "au": 53, "kp": 38, "pc": 51, "f10.7": 50,
              "imfid": -1, "swid": -1, "ts": -1, "rmsts": -1}


def _omni_url(tint, omni_database):
    if omni_database == "omni_hour":
        data_source = "omni2"
        date_format = "%Y%m%d"
        delta_t_min = 24 * 3600
    elif omni_database == "omni_min":
        data_source = "omni_min"
        date_format = "%Y%m%d%H"
        delta_t_min = 3600
    else:
        raise ValueError("Invalid database")

    url_ = "omniweb.gsfc.nasa.gov/cgi/nx1.cgi?activity=retrieve&spacecraft="
    url_ = f"https://{url_}{data_source}"

    tint[0] += np.timedelta64(0, "[s]")
    tint[1] += np.timedelta64(delta_t_min, "[s]")
    tint = [t_.astype(datetime.datetime) for t_ in tint]
    start_date, end_date = [t_.strftime(date_format) for t_ in tint]

    url_ = f"{url_}&start_date={start_date}&end_date={end_date}"

    return url_


def get_omni_data(variables, tint, database: str = "omni_hour"):
    r"""Downloads OMNI data.

    Parameters
    ----------
    variables : list
        Keys of the variables to download.
    tint : list
        Time interval.
    database : {"omni_hour", "omni_min"}, Optional
        OMNI data resolution. Default is database = "omni_hour".

    Returns
    -------
    data : xarray.Dataset
        OMNI data.

    """

    tint = iso86012datetime64(np.array(tint)).astype("<M8[s]")

    url_ = _omni_url(tint, database)

    vars_ = ""
    for variable in variables:
        vars_ = f"{vars_}&vars={var_omni_2[variable]:d}"

    with urllib.request.urlopen(f"{url_}{vars_}") as file:
        out = str(file.read())

    idx_start, idx_end = [out.find("YEAR"), out.find("</pre>")]

    lines = out[idx_start:idx_end].split("\\n")[:-1]
    lines = [list(filter(lambda x: x != "", l_.split(" "))) for l_ in lines]
    lines = [[f"{l_[0]}-{l_[1]}/{l_[2]}", *l_[3:]] for l_ in lines[1:]]
    data = pd.DataFrame(lines, columns=["time", *variables])
    data["time"] = pd.to_datetime(data["time"], format="%Y-%j/%H")

    data = data.set_index("time").astype(float)
    fmt_ = f"<M8[{database[5].lower()}]"
    data = data.loc[data.index.isin(tint.astype(fmt_).astype("<M8[ns]"))]
    data = data.to_xarray()

    return data
