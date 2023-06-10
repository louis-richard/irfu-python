#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os

# Third party imports
import numpy as np
import requests
from scipy.io import readsav

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.extend_tint import extend_tint
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_time import ts_time

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

URL = "http://www.spedas.org/mms/mms_brst_intervals.sav"


def load_brst_segments(tint, data_path: str = None, download: bool = True):
    r"""Load burst segment time intervals associated with the input time
    interval `tint`.

    Parameters
    ----------
    tint : list
        Time interval to look for burst segments.
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu/mms/config.json`

    Returns
    -------
    brst_segments : list
        Segments of burst mode data.

    """

    # Check path
    if not data_path:
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    # Define path of the brst segment file to save
    file_path = os.path.join(data_path, URL.split("/", maxsplit=100)[-1])

    if download:
        # Get url content
        response = requests.get(URL, timeout=10)

        # Write content of the brst segment file to the local file
        with open(file_path, "wb", encoding="utf-8") as fs:
            fs.write(response.content)

    # Read brst segment content
    intervals = readsav(file_path)

    unix_start = ts_time(intervals["brst_intervals"].start_times[0])
    unix_end = ts_time(intervals["brst_intervals"].end_times[0])

    unix_start = time_clip(unix_start, extend_tint(tint, [-300, 300]))
    unix_end = time_clip(unix_end, extend_tint(tint, [-300, 300]))

    # +10 second offset added; there appears to be an extra 10
    # seconds of data, consistently, not included in the range here
    offset = np.timedelta64(10, "s")
    unix_end = unix_end.assign_coords(time=unix_end.time.data + offset)
    unix_end.data += np.timedelta64(10, "s")

    brst_segments = []
    l_bound, r_bound = [np.datetime64(t_) for t_ in tint]

    for start_time, end_time in zip(unix_start, unix_end):
        if end_time >= l_bound and start_time <= r_bound:
            segment = np.array([start_time.data, end_time.data])
            segment = list(datetime642iso8601(segment))
            brst_segments.append(segment)

    return brst_segments
