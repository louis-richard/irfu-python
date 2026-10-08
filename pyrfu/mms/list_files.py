#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import json
import os
import re
from typing import Mapping, Optional, Union

# 3rd party imports
import numpy as np

from pyrfu.mms.db_init import MMS_CFG_PATH

# Local imports
from pyrfu.mms.list_files_aws import _file_time
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601
from pyrfu.pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
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

    if not isinstance(mms_id, str):
        mms_id = str(mms_id)

    t_start, t_end = [_file_time(re.sub(r"\D", "", t)[:14]) for t in tint_iso8601]

    # directory and file name search patterns:
    # - assume directories are of the form:
    # (srvy, SITL): spacecraft/instrument/rate/level[/datatype]/year/month/
    # (brst): spacecraft/instrument/rate/level[/datatype]/year/month/day/
    # - assume file names are of the form:
    # spacecraft_instrument_rate_level[_datatype]_YYYYMMDD[hhmmss]_version.cdf
    file_regex = re.compile(
        rf"^mms{mms_id}_{var['inst']}_{var['tmmode']}_{var['lev']}"
        + r"(?:_.*)?_([0-9]{8,14})_v(\d+)\.(\d+)\.(\d+)\.cdf$"
    )

    if var["dtype"] == "" or var["dtype"] is None:
        level_and_dtype = [var["lev"]]
    else:
        level_and_dtype = [var["lev"], var["dtype"]]

    # Latest version of each file, by time tag
    files = {}
    day = datetime.datetime.combine(t_start.date(), datetime.time())

    while day < t_end:
        local_dir = os.path.join(
            root_path,
            f"mms{mms_id}",
            var["inst"],
            var["tmmode"],
            *level_and_dtype,
            day.strftime("%Y"),
            day.strftime("%m"),
        )

        if var["tmmode"] == "brst":
            local_dir = os.path.join(local_dir, day.strftime("%d"))

        for root, _, file_names in os.walk(local_dir):
            for file_name in file_names:
                matches = file_regex.match(file_name)

                if not matches:
                    continue

                time_tag, *version = matches.groups()
                version = tuple(map(int, version))

                if time_tag not in files or version > files[time_tag][0]:
                    files[time_tag] = (version, os.path.join(root, file_name))

        day += datetime.timedelta(days=1)

    # Files starting within the time interval, and the last one starting before
    # (or at) its start, which covers it (as list_files_aws)
    times = sorted(files, key=_file_time)
    starts = [_file_time(time_tag) for time_tag in times]
    i_first = max([i for i, start in enumerate(starts) if start <= t_start] or [0])

    file_paths = [
        files[time_tag][1]
        for time_tag, start in zip(times[i_first:], starts[i_first:])
        if start < t_end
    ]

    return file_paths
