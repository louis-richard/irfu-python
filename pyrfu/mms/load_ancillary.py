#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import re
import json
import glob
import bisect
import fnmatch
import logging

# 3rd party imports
import pandas as pd

# Local imports
from ..pyrf import iso86012datetime

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.11"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


def load_ancillary(level_and_dtype, tint, mms_id, verbose: bool = True,
                   data_path: str = ""):
    r"""Loads ancillary data.

    Parameters
    ----------
    level_and_dtype : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    tint : list of str
        Time interval
    mms_id : str or int
        Spacecraft index
    verbose : bool, Optional
        Set to True to follow the loading. Default is True
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu.mms.mms_config.py`

    Returns
    -------
    out : xarray.Dataset
        Time series of the ancillary data

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

    if isinstance(mms_id, int):
        mms_id = str(mms_id)

    tint = iso86012datetime(tint)

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
    dir_pattern = os.sep.join([data_path, "ancillary", f"mms{mms_id}",
                               level_and_dtype])
    file_pattern = "_".join(["MMS{}".format(mms_id), level_and_dtype.upper(),
                             "???????_???????.V??"])

    files_in_tint = []
    out_files = []

    files = glob.glob(os.sep.join([dir_pattern, file_pattern]))

    # find the files within the time interval
    fname_fmt = f"MMS{mms_id}_{level_and_dtype.upper()}" \
                f"_([0-9]{{7}})_([0-9]{{7}}).V[0-9]{{2}}"
    file_regex = re.compile(os.sep.join([dir_pattern, fname_fmt]))
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

    # Read length of header and columns names from .json file
    # Root path
    pkg_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.sep.join([pkg_path, "ancillary.json"])) as file:
        anc_dict = json.load(file)

    if verbose:
        logging.info(f"Loading ancillary {level_and_dtype} files...")

    data_frame_dict = {}

    for i, file in enumerate(files_names):
        rows = pd.read_csv(file, delim_whitespace=True, header=None,
                           skiprows=anc_dict[level_and_dtype]["header"])

        # Remove footer
        rows = rows[:][:-1]

        # Convert time
        fmt = anc_dict[level_and_dtype]["time_format"]
        rows[0] = pd.to_datetime(rows[0], format=fmt)

        start_idx = bisect.bisect_left(rows[0][:], tint[0])
        end_idx = bisect.bisect_left(rows[0][:], tint[1])
        rows.columns = anc_dict[level_and_dtype]["columns_names"]

        data_frame_dict[i] = rows[:][start_idx:end_idx]

    data_frame = data_frame_dict[0]

    for k in list(data_frame_dict.keys())[1:]:
        data_frame = data_frame.append(data_frame_dict[k])

    data_frame = data_frame.sort_values(by="time").set_index(["time"])

    return data_frame.to_xarray()
