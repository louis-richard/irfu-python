#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import json
import os
import re
from typing import Any, Mapping, Optional, Union

# 3rd party imports
import boto3
import numpy as np
from dateutil import parser
from dateutil.rrule import DAILY, rrule

from pyrfu.mms.db_init import MMS_CFG_PATH

# Local imports
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601
from pyrfu.pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def list_files_aws(
    tint: list[str],
    mms_id: Union[str, int],
    var: Mapping[str, str],
    bucket_prefix: Optional[str] = "",
) -> list[dict[str, Any]]:
    r"""List files from Amazon Web Services (AWS).

    Find available files from the Amazon Wed Services (AWS) for the target instrument,
    data type, data rate, mms_id and level during the target time interval.

    Parameters
    ----------
    tint : list of str
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
    file_names : list of str
        List of files corresponding to the parameters in the selected time
        interval

    Raises
    ------
    FileNotFoundError
        If the path doesn't exist in the AWS S3 bucket or if the bucket doesn't exist.
    TypeError
        If the time interval is not array_like or if tint values are not in datetime64
        or str.

    """
    # Start S3 session
    s3 = boto3.resource("s3")

    # Check path
    if not bucket_prefix:
        # Read the current version of the MMS configuration file
        with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
            config = json.load(fs)

        aws_path_split = config["aws"].split("/")
    else:
        aws_path_split = bucket_prefix.split("/")

    bucket_name, prefix = aws_path_split[0], "/".join(aws_path_split[1:])

    # Make sure that the data path exists
    bucket = s3.Bucket(bucket_name)

    if not bucket:
        raise FileNotFoundError(f"{bucket_name} doesn't exist!!")

    if bucket.objects.filter(Prefix=prefix):
        raise FileNotFoundError(f"{prefix} doesn't exist in {bucket_name}")

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

    files_out = []

    for date in days:
        if var["tmmode"] == "brst":
            bucket_prefix = os.sep.join(
                [
                    prefix,
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
                    prefix,
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
