#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import fnmatch
import glob
import json
import os
import re
from typing import Optional, Union

# 3rd party imports
import numpy as np
import pandas as pd

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime import iso86012datetime
from ..pyrf.iso86012datetime64 import iso86012datetime64
from .db_init import MMS_CFG_PATH

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def list_files_ancillary(
    tint: list[str],
    mms_id: Union[str, int],
    product: str,
    data_path: Optional[str] = "",
) -> list[str]:
    r"""Find available ancillary files in the data directories for the target product
    type.

    Parameters
    ----------
    tint : list of str
        Time interval
    mms_id : str or int
        Spacecraft index
    product : str
        Ancillary type: {"predatt", "predeph", "defatt", "defeph"}.
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu.mms.mms_config.py`

    Returns
    -------
    file_names : list
        Ancillary files in interval.

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

    if isinstance(mms_id, int):
        mms_id = str(mms_id)

    # Check time interval
    if isinstance(tint, list):
        tint_array = np.array(tint)
    else:
        raise TypeError("tint must be a list!!")

    # Convert time interval to ISO 8601
    if isinstance(tint_array[0], str):
        tint_iso8601 = datetime642iso8601(iso86012datetime64(tint_array))
        tint_datetime = iso86012datetime(tint_iso8601)
    else:
        raise TypeError("Values must be in str!!")

    # PAD time interval to handle ancillary file start after midnight
    tint_datetime[0] = tint_datetime[0] - datetime.timedelta(days=1)
    # tint_datetime[1] = tint_datetime[1] + datetime.timedelta(days=1)

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
    dir_pattern = os.sep.join([root_path, "ancillary", f"mms{mms_id}", product])
    file_pattern = "_".join([f"MMS{mms_id}", product.upper(), "???????_???????.V??"])

    # find all files in the directory
    files = glob.glob(os.sep.join([dir_pattern, file_pattern]))

    # find the files within the time interval
    fname_fmt = f"MMS{mms_id}_{product.upper()}_([0-9]{{7}})_([0-9]{{7}}).V[0-9]{{2}}"

    if os.name == "nt":
        full_path = os.sep.join([re.escape(dir_pattern) + os.sep, fname_fmt])
    else:
        full_path = os.sep.join([re.escape(dir_pattern), fname_fmt])

    file_regex = re.compile(full_path)

    files_in_tint = []

    for file in files:
        time_match = file_regex.match(file)
        if time_match is not None:
            start_time = pd.to_datetime(time_match.group(1), format="%Y%j")
            end_time = pd.to_datetime(time_match.group(2), format="%Y%j")
            if start_time < tint_datetime[1] and end_time >= tint_datetime[0]:
                files_in_tint.append(file)

    # ensure only the latest version of each file is loaded
    out_files = []

    for file_in_tint in files_in_tint:
        this_file = file_in_tint[0:-3] + "V??"
        versions = fnmatch.filter(files_in_tint, this_file)
        if len(versions) > 1:
            # only grab the latest version
            out_files.append(sorted(versions)[-1])
        else:
            out_files.append(versions[0])

    file_names = list(set(out_files))
    file_names.sort()

    return file_names
