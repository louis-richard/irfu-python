#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import fnmatch
import glob
import json

# Built-in imports
import os
import re

# 3rd party imports
import numpy as np
import pandas as pd

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime import iso86012datetime
from ..pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.3.26"
__status__ = "Prototype"


def list_files_ancillary(tint, mms_id, product, data_path: str = ""):
    r"""Loads ancillary data.

    Parameters
    ----------
    tint : list of str
        Time interval
    mms_id : str or int
        Spacecraft index
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu.mms.mms_config.py`

    Returns
    -------
    files_names : list
        Ancillary files in interval.

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

    # Make sure that the data path exists
    assert os.path.exists(data_path), f"{data_path} doesn't exist!!"

    if isinstance(mms_id, int):
        mms_id = str(mms_id)

    # Check time interval type
    if isinstance(tint, (np.ndarray, list)):
        if isinstance(tint[0], np.datetime64):
            tint = datetime642iso8601(tint)
            tint = iso86012datetime(tint)
        elif isinstance(tint[0], str):
            tint = iso86012datetime64(
                np.array(tint),
            )  # to make sure it is ISO8601 ok!
            tint = datetime642iso8601(tint)
            tint = iso86012datetime(tint)
        elif isinstance(tint[0], datetime.datetime):
            pass
        else:
            raise TypeError("Values must be in datetime, datetime64, or str!!")
    else:
        raise TypeError("tint must be a DataArray or array_like!!")

    # directory and file name search patterns
    # For now
    # -all ancillary data is in one directory:
    #       mms\ancillary
    # -assume file names are of the form:
    #   SPACECRAFT_FILETYPE_startDate_endDate.version
    #   where SPACECRAFT is [MMS1, MMS2, MMS3, MMS4] in uppercase
    #   and FILETYPE is either DEFATT, PREDATT, DEFEPH, PREDEPH in uppercase
    #   and start/endDate is YYYYDOY
    #   and version is Vnn (.V00, .V01, etc..)
    dir_pattern = os.sep.join([data_path, "ancillary", f"mms{mms_id}", product])
    file_pattern = "_".join([f"MMS{mms_id}", product.upper(), "???????_???????.V??"])

    files_in_tint = []
    out_files = []

    files = glob.glob(os.sep.join([dir_pattern, file_pattern]))

    # find the files within the time interval
    fname_fmt = f"MMS{mms_id}_{product.upper()}_([0-9]{{7}})_([0-9]{{7}}).V[0-9]{{2}}"

    if os.name == "nt":
        full_path = os.sep.join([re.escape(dir_pattern) + os.sep, fname_fmt])
    else:
        full_path = os.sep.join([re.escape(dir_pattern), fname_fmt])

    file_regex = re.compile(full_path)

    for file in files:
        time_match = file_regex.match(file)
        if time_match is not None:
            start_time = pd.to_datetime(time_match.group(1), format="%Y%j")
            end_time = pd.to_datetime(time_match.group(2), format="%Y%j")
            if start_time < tint[1] and end_time >= tint[0]:
                files_in_tint.append(file)

    # ensure only the latest version of each file is loaded
    for file in files_in_tint:
        this_file = file[0:-3] + "V??"
        versions = fnmatch.filter(files_in_tint, this_file)
        if len(versions) > 1:
            # only grab the latest version
            out_files.append(sorted(versions)[-1])
        else:
            out_files.append(versions[0])

    files_names = list(set(out_files))
    files_names.sort()

    return files_names
