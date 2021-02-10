#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import os
import re
import json
import glob
import bisect
import fnmatch
import pandas as pd

from dateutil import parser as date_parser

from .mms_config import CONFIG


def load_ancillary(level_and_dtype, tint, probe, verbose=True, data_path=""):
    """Load ancillary data

    Parameters
    ----------
    level_and_dtype : str
        Ancillary type :
            * predatt
            * predeph
            * defatt
            * defeph

    tint : list of str
        Time interval

    probe : str or int
        Spacecraft index

    verbose : bool, optional
        Set to True to follow the loading. Default is True

    data_path : str, optional
        Path of MMS data. If None use `pyrfu.mms.mms_config.py`

    Returns
    -------
    out : xarray.Dataset
        Time series of the ancillary data

    """

    if not data_path:
        data_path = CONFIG["local_data_dir"]

    if isinstance(probe, int):
        probe = str(probe)

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
    dir_pattern = os.sep.join([data_path, "ancillary", "mms{}".format(probe), level_and_dtype])
    file_pattern = "_".join(["MMS{}".format(probe), level_and_dtype.upper(), "???????_???????.V??"])

    files_in_tint = []
    out_files = []

    files = glob.glob(os.sep.join([dir_pattern, file_pattern]))

    # find the files within the time interval
    file_regex = re.compile(os.sep.join([dir_pattern,
                                         'MMS' + probe + '_' + level_and_dtype.upper()
                                         + '_([0-9]{7})_([0-9]{7}).V[0-9]{2}']))
    for file in files:
        time_match = file_regex.match(file)
        if time_match is not None:
            start_time = pd.to_datetime(time_match.group(1), format="%Y%j")
            end_time = pd.to_datetime(time_match.group(2), format="%Y%j")
            if start_time < date_parser.parse(tint[1]) and end_time >= date_parser.parse(tint[0]):
                files_in_tint.append(file)

    # ensure only the latest version of each file is loaded
    for file in files_in_tint:
        this_file = file[0:-3] + "V??"
        versions = fnmatch.filter(files_in_tint, this_file)
        if len(versions) > 1:
            out_files.append(sorted(versions)[-1])  # only grab the latest version
        else:
            out_files.append(versions[0])

    files_names = list(set(out_files))
    files_names.sort()

    # Read length of header and columns names from .json file
    with open("./ancillary.json") as file:
        anc_dict = json.load(file)

    if verbose:
        print("Loading ancillary {} files...".format(level_and_dtype))

    data_frame_dict = {}

    for i, file in enumerate(files_names):
        rows = pd.read_csv(file, delim_whitespace=True, header=None,
                           skiprows=anc_dict[level_and_dtype]["header"])

        # Remove footer
        rows = rows[:][:-1]

        # Convert time
        rows[0] = pd.to_datetime(rows[0], format=anc_dict[level_and_dtype]["time_format"])

        start_idx = bisect.bisect_left(rows[0][:], date_parser.parse(tint[0]))
        end_idx = bisect.bisect_left(rows[0][:], date_parser.parse(tint[1]))
        rows.columns = anc_dict[level_and_dtype]["columns_names"]

        data_frame_dict[i] = rows[:][start_idx:end_idx]

    data_frame = data_frame_dict[0]

    for k in list(data_frame_dict.keys())[1:]:
        data_frame = data_frame.append(data_frame_dict[k])

    data_frame = data_frame.sort_values(by="time").set_index(["time"])

    return data_frame.to_xarray()
