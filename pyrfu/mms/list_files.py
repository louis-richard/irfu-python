#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
list_files.py

@author : Louis RICHARD
"""

import os
import re
import bisect
import datetime

from dateutil import parser
from dateutil.rrule import rrule, DAILY

from .mms_config import CONFIG


def list_files(tint=None, mms_id="1", var=None):
    """Find files in the data directories of the target instrument, data type, data rate, mms_id and
    level during the target time interval

    Parameters
    ----------
    tint : list
        Time interval

    mms_id : str or int
        Index of the spacecraft

    var : dict
        Dictionary containing 4 keys
            * var["inst"] : name of the instrument
            * var["tmmode"] : data rate
            * var["lev"] : data level
            * var["dtype"] : data type

    Returns
    -------
    files : list
        List of files corresponding to the parameters in the selected time interval

    """

    data_path = CONFIG["local_data_dir"]

    if var is None:
        raise ValueError("var is empty")

    files_out = []

    if not isinstance(mms_id, str):
        mms_id = str(mms_id)
    # directory and file name search patterns
    #   -assume directories are of the form:
    #      (srvy, SITL): spacecraft/instrument/rate/level[/datatype]/year/month/
    #      (brst): spacecraft/instrument/rate/level[/datatype]/year/month/day/
    #   -assume file names are of the form:
    #      spacecraft_instrument_rate_level[_datatype]_YYYYMMDD[hhmmss]_version.cdf

    file_name = "mms" + mms_id + "_" + var["inst"] + "_" + var["tmmode"] + "_" + var[
        "lev"] + "(_)?.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf"

    days = rrule(DAILY, dtstart=parser.parse(parser.parse(tint[0]).strftime("%Y-%m-%d")),
                 until=parser.parse(tint[1]) - datetime.timedelta(seconds=1))

    if var["dtype"] == "" or var["dtype"] is None:
        level_and_dtype = var["lev"]
    else:
        level_and_dtype = os.sep.join([var["lev"], var["dtype"]])

    for date in days:
        if var["tmmode"] == "brst":
            local_dir = os.sep.join(
                [data_path, f"mms{mms_id}", var["inst"], var["tmmode"], level_and_dtype,
                 date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")])
        else:
            local_dir = os.sep.join(
                [data_path, f"mms{mms_id}", var["inst"], var["tmmode"], level_and_dtype,
                 date.strftime("%Y"), date.strftime("%m")])

        if os.name == "nt":
            full_path = os.sep.join([re.escape(local_dir)+os.sep, file_name])
        else:
            full_path = os.sep.join([re.escape(local_dir), file_name])

        regex = re.compile(full_path)

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                this_file = os.sep.join([root, file])

                matches = regex.match(this_file)
                if matches:
                    this_time = parser.parse(matches.groups()[1])
                    if (parser.parse(parser.parse(tint[0]).strftime("%Y-%m-%d")) <= this_time
                            <= parser.parse(tint[1]) - datetime.timedelta(seconds=1)):
                        if this_file not in files_out:
                            files_out.append(
                                {"file_name": file, "timetag": "", "full_name": this_file,
                                 "file_size": ""})

    in_files = files_out

    file_name = "mms.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf"

    file_times = []

    regex = re.compile(file_name)

    for file in in_files:
        matches = regex.match(file["file_name"])
        if matches:
            file_times.append((file["file_name"], parser.parse(matches.groups()[0]).timestamp(),
                               file["timetag"], file["file_size"]))

    # sort in time
    sorted_files = sorted(file_times, key=lambda x: x[1])

    times = [t[1] for t in sorted_files]

    idx_min = bisect.bisect_left(times, parser.parse(tint[0]).timestamp())

    # note: purposefully liberal here; include one extra file so that we always get the burst mode
    # data
    if idx_min == 0:
        files_in_interval = []
        for f in sorted_files[idx_min:]:
            files_in_interval.append({"file_name": f[0], "timetag": f[2], "file_size": f[3]})
    else:
        files_in_interval = []
        for f in sorted_files[idx_min - 1:]:
            files_in_interval.append({"file_name": f[0], "timetag": f[2], "file_size": f[3]})

    local_files = []

    file_names = [f["file_name"] for f in files_in_interval]

    for file in files_out:
        if file["file_name"] in file_names:
            local_files.append(file["full_name"])

    return sorted(local_files)
