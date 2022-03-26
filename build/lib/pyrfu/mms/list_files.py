#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import re
import json
import bisect
import datetime

# 3rd party imports
from dateutil import parser
from dateutil.rrule import rrule, DAILY

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.11"
__status__ = "Prototype"


def list_files(tint, mms_id, var, data_path=""):
    """Find files in the data directories of the target instrument, data type,
    data rate, mms_id and level during the target time interval.

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
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`

    Returns
    -------
    files : list
        List of files corresponding to the parameters in the selected time
        interval

    """

    # Check path
    if not data_path:
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r") as f:
            config = json.load(f)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    # Make sure that the data path exists
    assert os.path.exists(data_path), f"{data_path} doesn't exist!!"

    files_out = []

    if not isinstance(mms_id, str):
        mms_id = str(mms_id)
    # directory and file name search patterns:
    # - assume directories are of the form:
    # (srvy, SITL): spacecraft/instrument/rate/level[/datatype]/year/month/
    # (brst): spacecraft/instrument/rate/level[/datatype]/year/month/day/
    # - assume file names are of the form:
    # spacecraft_instrument_rate_level[_datatype]_YYYYMMDD[hhmmss]_version.cdf

    file_name = f"mms{mms_id}_{var['inst']}_{var['tmmode']}_{var['lev']}" \
                + r"(_)?.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf"

    d_start = parser.parse(parser.parse(tint[0]).strftime("%Y-%m-%d"))
    until_ = parser.parse(tint[1]) - datetime.timedelta(seconds=1)
    days = rrule(DAILY, dtstart=d_start, until=until_)

    if var["dtype"] == "" or var["dtype"] is None:
        level_and_dtype = var["lev"]
    else:
        level_and_dtype = os.sep.join([var["lev"], var["dtype"]])

    for date in days:
        if var["tmmode"] == "brst":
            local_dir = os.sep.join([data_path, f"mms{mms_id}", var["inst"],
                                     var["tmmode"], level_and_dtype,
                                     date.strftime("%Y"), date.strftime("%m"),
                                     date.strftime("%d")])
        else:
            local_dir = os.sep.join([data_path, f"mms{mms_id}", var["inst"],
                                     var["tmmode"], level_and_dtype,
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
                    if d_start <= this_time <= until_:
                        if this_file not in files_out:
                            files_out.append({"file_name": file,
                                              "timetag": "",
                                              "full_name": this_file,
                                              "file_size": ""})

    in_files = files_out

    file_name = r"mms.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf"

    file_times = []

    regex = re.compile(file_name)

    for file in in_files:
        matches = regex.match(file["file_name"])
        if matches:
            file_times.append((file["file_name"],
                               parser.parse(matches.groups()[0]).timestamp(),
                               file["timetag"], file["file_size"]))

    # sort in time
    sorted_files = sorted(file_times, key=lambda x: x[1])

    times = [t[1] for t in sorted_files]

    idx_min = bisect.bisect_left(times, parser.parse(tint[0]).timestamp())

    # note: purposefully liberal here; include one extra file so that we
    # always get the burst mode
    # data
    if idx_min == 0:
        files_in_interval = []
        for file in sorted_files[idx_min:]:
            files_in_interval.append({"file_name": file[0],
                                      "timetag": file[2],
                                      "file_size": file[3]})
    else:
        files_in_interval = []
        for file in sorted_files[idx_min - 1:]:
            files_in_interval.append({"file_name": file[0],
                                      "timetag": file[2],
                                      "file_size": file[3]})

    local_files = []

    file_names = [f["file_name"] for f in files_in_interval]

    for file in files_out:
        if file["file_name"] in file_names:
            local_files.append(file["full_name"])

    return sorted(local_files)
