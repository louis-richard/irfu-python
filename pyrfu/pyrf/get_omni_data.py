#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import urllib.request

# 3rd party imports
import numpy as np
import pandas as pd

# Local imports
from .iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


var_omni_1 = {
    "b": 13,
    "avgb": -1,
    "blat": -1,
    "blong": -1,
    "bx": 14,
    "bxgse": 14,
    "bxgsm": 14,
    "by": 15,
    "bygse": 15,
    "bz": 16,
    "bzgse": 16,
    "bygsm": 17,
    "bzgsm": 18,
    "t": 26,
    "n": 25,
    "nanp": -1,
    "v": 21,
    "vx": 22,
    "vy": 23,
    "vz": 24,
    "vlon": -1,
    "vlat": -1,
    "p": 27,
    "e": 28,
    "beta": 29,
    "ma": 30,
    "bsnx": 34,
    "bsny": 35,
    "bsnz": 36,
    "ms": 45,
    "ssn": -1,
    "dst": -1,
    "ae": 37,
    "al": 38,
    "au": 39,
    "kp": -1,
    "pc": 44,
    "f10.7": -1,
    "imfid": 4,
    "swid": 5,
    "ts": 9,
    "rmsts": 10,
}

var_omni_2 = {
    "b": 8,
    "avgb": 9,
    "blat": 10,
    "blong": 11,
    "bx": 12,
    "bxgse": 12,
    "bxgsm": 12,
    "by": 13,
    "bygse": 13,
    "bz": 14,
    "bzgse": 14,
    "bygsm": 15,
    "bzgsm": 16,
    "t": 22,
    "n": 23,
    "nanp": 27,
    "v": 24,
    "vx": -1,
    "vy": -1,
    "vz": -1,
    "vlon": 25,
    "vlat": 26,
    "p": 28,
    "e": 35,
    "beta": 36,
    "ma": 37,
    "bsnx": -1,
    "bsny": -1,
    "bsnz": -1,
    "ms": 54,
    "ssn": 39,
    "dst": 40,
    "ae": 41,
    "al": 52,
    "au": 53,
    "kp": 38,
    "pc": 51,
    "f10.7": 50,
    "imfid": 4,
    "swid": 5,
    "ts": -1,
    "rmsts": -1,
}


# Fill values of the variables (by code) for the 1-minute (HRO) and hourly (OMNI2)
# data, from https://omniweb.gsfc.nasa.gov/html/omni_min_data.html and
# https://omniweb.gsfc.nasa.gov/html/ow_data.html (word number = code + 1)
fill_omni_1 = {
    **dict.fromkeys([4, 5], 99.0),
    **dict.fromkeys([9, 10], 999999.0),
    **dict.fromkeys([13, 14, 15, 16, 17, 18, 34, 35, 36], 9999.99),
    **dict.fromkeys([21, 22, 23, 24], 99999.9),
    25: 999.99,
    26: 9999999.0,
    27: 99.99,
    **dict.fromkeys([28, 29, 44], 999.99),
    30: 999.9,
    **dict.fromkeys([37, 38, 39], 99999.0),
    45: 99.9,
}

fill_omni_2 = {
    **dict.fromkeys([4, 5, 38], 99.0),
    **dict.fromkeys([8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 25, 26, 37], 999.9),
    22: 9999999.0,
    24: 9999.0,
    27: 9.999,
    28: 99.99,
    **dict.fromkeys([35, 36], 999.99),
    39: 999.0,
    **dict.fromkeys([40, 52, 53], 99999.0),
    41: 9999.0,
    **dict.fromkeys([50, 51], 999.9),
    54: 99.9,
}

# Variable codes, fill values and time resolution of each database
_DATABASES = {
    "omni_hour": ("omni2", var_omni_2, fill_omni_2, "%Y%m%d", np.timedelta64(1, "h")),
    "omni_min": (
        "omni_min",
        var_omni_1,
        fill_omni_1,
        "%Y%m%d%H",
        np.timedelta64(1, "m"),
    ),
}


def _omni_url(tint, omni_database, codes):
    # OMNIWeb returns whole days (hourly data) or hours (1-minute data), up to and
    # including the end date
    data_source, _, _, date_format, _ = _DATABASES[omni_database]

    url_ = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi?activity=retrieve"
    start_date, end_date = [
        t_.astype("datetime64[s]").astype(datetime.datetime).strftime(date_format)
        for t_ in tint
    ]
    url_ = f"{url_}&spacecraft={data_source}&start_date={start_date}"
    url_ = f"{url_}&end_date={end_date}"
    url_ += "".join(f"&vars={code:d}" for code in codes)

    return url_


def _parse_omni(text, n_codes):
    # Table between <pre> and </pre>: header "YEAR DOY HR" (hourly) or
    # "YYYY DOY HR MN" (1-minute), then one line per time
    table = text.split("<pre>", 1)[-1].split("</pre>", 1)[0]
    lines = table.splitlines()
    i_header = next(
        (i for i, line in enumerate(lines) if line.split()[:1] in (["YEAR"], ["YYYY"])),
        None,
    )

    if i_header is None:
        raise ValueError(f"OMNIWeb returned no data: {' '.join(table.split())[:200]}")

    n_time = 4 if lines[i_header].split()[0] == "YYYY" else 3
    rows = [line.split() for line in lines[i_header + 1 :] if line.strip()]
    rows = np.array([row for row in rows if len(row) == n_time + n_codes], dtype=float)
    rows = rows.reshape(-1, n_time + n_codes)

    times = pd.to_datetime(
        [f"{int(y):04d}{int(d):03d}" for y, d in rows[:, :2]], format="%Y%j"
    )
    times += pd.to_timedelta(rows[:, 2], unit="h")

    if n_time == 4:
        times += pd.to_timedelta(rows[:, 3], unit="m")

    return times, rows[:, n_time:]


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
        OMNI data, at the times whose averaging interval (hour or minute,
        starting at the time) overlaps `tint`. Fill values are replaced by NaN.

    Raises
    ------
    ValueError
        If the database or a variable is not supported, or if OMNIWeb returns no
        data.

    """

    if database not in _DATABASES:
        raise ValueError(f"Invalid database {database}. Use 'omni_hour' or 'omni_min'")

    _, var_codes, fill_values, _, resolution = _DATABASES[database]

    codes = []

    for variable in variables:
        if var_codes.get(variable, -1) < 0:
            available = [k for k, v in var_codes.items() if v >= 0]
            raise ValueError(
                f"{variable} is not available in {database}. Available: {available}"
            )

        codes.append(var_codes[variable])

    # Each code is requested once (e.g., "bx" and "bxgse" are the same variable)
    codes_unique = list(dict.fromkeys(codes))

    tint = iso86012datetime64(np.array(tint)).astype("datetime64[ns]")
    url_ = _omni_url(tint, database, codes_unique)

    with urllib.request.urlopen(url_, timeout=60) as file:
        text = file.read().decode("utf-8", errors="replace")

    times, values = _parse_omni(text, len(codes_unique))

    # Fill values to NaN
    for i, code in enumerate(codes_unique):
        if code in fill_values:
            values[np.isclose(values[:, i], fill_values[code]), i] = np.nan

    columns = {
        var: values[:, codes_unique.index(code)] for var, code in zip(variables, codes)
    }
    data = pd.DataFrame(columns, index=pd.DatetimeIndex(times, name="time"))

    # Times whose averaging interval [t, t + resolution) overlaps tint
    t_start, t_end = [pd.Timestamp(t_) for t_ in tint]
    in_tint = (data.index > t_start - resolution) & (data.index <= t_end)
    data = data.loc[in_tint].to_xarray()

    return data
