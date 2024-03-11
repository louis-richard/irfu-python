#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json

# Built-in imports
import os
import re

# 3rd party imports
import boto3
import numpy as np
from dateutil import parser
from dateutil.rrule import DAILY, rrule

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime64 import iso86012datetime64
from .db_init import MMS_CFG_PATH

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def list_files_aws(tint, mms_id, var, bucket_prefix: str = ""):
    r"""Find available files from the Amazon Wed Services (AWS) for the target
    instrument, data type, data rate, mms_id and level during the target time interval.

    Parameters
    ----------
    tint : array_like
        Time interval
    mms_id : str or int
        Index of the spacecraft
    var : dict
        Dictionary containing 4 keys
            * var["inst"] : name of the instrument
            * var["tmmode"] : data rate
            * var["lev"] : data level
            * var["dtype"] : data type
    bucket_prefix : str, Optional
        Name of AWS S3 bucket.

    Returns
    -------
    file_names : list
        List of files corresponding to the parameters in the selected time
        interval

    """
    # Start S3 session
    s3 = boto3.resource("s3")

    # Check path
    if not bucket_prefix:
        # Read the current version of the MMS configuration file
        with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
            config = json.load(fs)

        bucket_name, prefix = config["aws"].split("/")
    else:
        bucket_name, prefix = bucket_prefix.split("/")

    # Make sure that the data path exists
    bucket = s3.Bucket(bucket_name)
    assert bucket, f"{bucket_name} doesn't exist!!"
    assert bucket.objects.filter(
        Prefix=prefix
    ), f"{prefix} doesn't exist in {bucket_name}"

    # Check time interval
    if isinstance(tint, (np.ndarray, list)):
        if isinstance(tint[0], np.datetime64):
            tint = datetime642iso8601(np.array(tint))
        elif isinstance(tint[0], str):
            tint = iso86012datetime64(
                np.array(tint),
            )  # to make sure it is ISO8601 ok!!
            tint = datetime642iso8601(np.array(tint))
        else:
            raise TypeError("Values must be in datetime64, or str!!")
    else:
        raise TypeError("tint must be array_like!!")

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

    d_start = parser.parse(parser.parse(tint[0]).strftime("%Y-%m-%d"))
    until_ = parser.parse(tint[1]) - datetime.timedelta(seconds=1)
    days = rrule(DAILY, dtstart=d_start, until=until_)

    if var["dtype"] == "" or var["dtype"] is None:
        level_and_dtype = var["lev"]
    else:
        level_and_dtype = os.sep.join([var["lev"], var["dtype"]])

    files_out = []

    for date in days:
        if var["tmmode"] == "brst":
            bucket_prefix = os.sep.join(
                [
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
            bucket_prefix = os.sep.join(
                [
                    f"mms{mms_id}",
                    var["inst"],
                    var["tmmode"],
                    level_and_dtype,
                    date.strftime("%Y"),
                    date.strftime("%m"),
                ],
            )

        full_path = os.sep.join([re.escape(bucket_prefix), file_name])

        regex = re.compile(full_path)

        files = bucket.objects.filter(Prefix=bucket_prefix)

        for file in files:
            this_file = file.key
            matches = regex.match(this_file)
            if matches:
                this_time = parser.parse(matches.groups()[1])
                if d_start <= this_time <= until_:
                    if this_file not in files_out:
                        files_out.append(
                            {
                                "s3_obj": file,
                                "timetag": "",
                                "full_name": this_file,
                                "file_size": "",
                            },
                        )
    return files_out
