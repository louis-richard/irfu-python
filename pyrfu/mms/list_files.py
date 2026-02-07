#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import datetime
import json
import os
import re
from typing import Mapping, Optional, Union

# 3rd party imports
import numpy as np
from dateutil import parser
from dateutil.rrule import DAILY, rrule

from pyrfu.mms.db_init import MMS_CFG_PATH

# Local imports
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601
from pyrfu.pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def list_files(
    tint: list[str],
    mms_id: Union[str, int],
    var: Mapping[str, str],
    data_path: Optional[str] = "",
) -> list[str]:
    r"""Find available files in the data directories for `var`.

    Parameters
    ----------
    tint : list
        Time interval
    mms_id : str or int
        Index of the spacecraft
    var : dict
        Dictionary containing at least 4 keys
            * var["inst"] : name of the instrument
            * var["tmmode"] : data rate
            * var["lev"] : data level
            * var["dtype"] : data type
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`

    Returns
    -------
    file_names : list
        List of files corresponding to the parameters in the selected time
        interval

    """
    # Check path
    if not data_path:
        # Read the current version of the MMS configuration file
        with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
            config = json.load(fs)

        root_path = os.path.normpath(config["local"])
    else:
        root_path = os.path.normpath(data_path)

    # Make sure that the data path exists
    assert os.path.exists(root_path), f"{root_path} doesn't exist!!"

    # Check time interval
    if isinstance(tint, list):
        tint_array = np.array(tint)
    else:
        raise TypeError("tint must be a list!!")

    # Convert time interval to ISO 8601
    if isinstance(tint_array[0], str):
        tint_iso8601 = datetime642iso8601(iso86012datetime64(tint_array))
    else:
        raise TypeError("Values must be in str!!")

    files_out = []

    if not isinstance(mms_id, str):
        mms_id = str(mms_id)

    # directory and file name search patterns:
    # - assume directories are of the form:
    # (srvy, SITL): spacecraft/instrument/rate/level[/datatype]/year/month/
    # (brst): spacecraft/instrument/rate/level[/datatype]/year/month/day/
    # - assume file names are of the form:
    # spacecraft_instrument_rate_level[_datatype]_YYYYMMDD[hhmmss]_version.cdf

    file_name = (
        f"mms{mms_id}_{var['inst']}_{var['tmmode']}_{var['lev']}"
        + r"(_)?.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf"
    )

    d_start = parser.parse(parser.parse(tint_iso8601[0]).strftime("%Y-%m-%d"))
    until_ = parser.parse(tint_iso8601[1]) - datetime.timedelta(seconds=1)
    days = rrule(DAILY, dtstart=d_start, until=until_)

    if var["dtype"] == "" or var["dtype"] is None:
        level_and_dtype = var["lev"]
    else:
        level_and_dtype = os.sep.join([var["lev"], var["dtype"]])

    for date in days:
        if var["tmmode"] == "brst":
            local_dir = os.sep.join(
                [
                    root_path,
                    f"mms{mms_id}",
                    var["inst"],
                    var["tmmode"],
                    level_and_dtype,
                    date.strftime("%Y"),
                    date.strftime("%m"),
                    date.strftime("%d"),
                ],
            )
        else:
            local_dir = os.sep.join(
                [
                    root_path,
                    f"mms{mms_id}",
                    var["inst"],
                    var["tmmode"],
                    level_and_dtype,
                    date.strftime("%Y"),
                    date.strftime("%m"),
                ],
            )

        if os.name == "nt":
            full_path = os.sep.join([re.escape(local_dir) + os.sep, file_name])
        else:
            full_path = os.sep.join([re.escape(local_dir), file_name])

        regex = re.compile(full_path)

        for root, _, files in os.walk(local_dir):
            for file in files:
                file_path = os.sep.join([root, file])

                matches = regex.match(file_path)
                if matches:
                    this_time = parser.parse(matches.groups()[1])
                    if d_start <= this_time <= until_:
                        this_file = {
                            "file_name": file,
                            "timetag": "",
                            "full_name": file_path,
                            "file_size": "",
                        }

                        if this_file not in files_out:
                            files_out.append(this_file)

    in_files = files_out

    file_name = r"mms.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf"

    file_times = []

    regex = re.compile(file_name)

    for in_file in in_files:
        matches = regex.match(in_file["file_name"])
        if matches:
            file_times.append(
                (
                    in_file["file_name"],
                    parser.parse(matches.groups()[0]).timestamp(),
                    in_file["timetag"],
                    in_file["file_size"],
                ),
            )

    # sort in time
    sorted_files = sorted(file_times, key=lambda x: x[1])

    times = [t[1] for t in sorted_files]

    idx_min = bisect.bisect_left(times, parser.parse(tint_iso8601[0]).timestamp())

    # note: purposefully liberal here; include one extra file so that we
    # always get the burst mode
    # data
    if idx_min == 0:
        files_in_interval = []
        for sorted_file in sorted_files[idx_min:]:
            files_in_interval.append(
                {
                    "file_name": sorted_file[0],
                    "timetag": sorted_file[2],
                    "file_size": sorted_file[3],
                },
            )
    else:
        files_in_interval = []
        for sorted_file in sorted_files[idx_min - 1 :]:
            files_in_interval.append(
                {
                    "file_name": sorted_file[0],
                    "timetag": sorted_file[2],
                    "file_size": sorted_file[3],
                },
            )

    local_files = []

    file_names = [f["file_name"] for f in files_in_interval]

    for file_out in files_out:
        if file_out["file_name"] in file_names:
            local_files.append(file_out["full_name"])

    file_names = sorted(local_files)

    return file_names
