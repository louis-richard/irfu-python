#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import json
import posixpath
import re
from typing import Any, Mapping, Optional, Union

# 3rd party imports
import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# Local imports
from pyrfu.mms.db_init import MMS_CFG_PATH
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601
from pyrfu.pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

# Public copy of the MMS archive (CDAWeb) on NASA HelioCloud, used if no bucket
# is set in the configuration file
HELIOCLOUD_MMS = "gov-nasa-hdrl-data1/spdf/cdaweb/data/mms"


def _s3_resource():
    r"""S3 resource, with unsigned (anonymous) requests if no AWS credentials
    are available, which is enough for public buckets such as HelioCloud's."""
    session = boto3.session.Session()

    if session.get_credentials() is None:
        return session.resource("s3", config=Config(signature_version=UNSIGNED))

    return session.resource("s3")


def _bucket_and_prefix(bucket_prefix: Optional[str] = "") -> tuple[str, str]:
    r"""Bucket name and key prefix from "bucket/prefix" (or "s3://bucket/prefix"),
    by default from the configuration file, else HelioCloud."""
    if not bucket_prefix:
        # Read the current version of the MMS configuration file
        with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
            config = json.load(fs)

        bucket_prefix = config.get("aws") or HELIOCLOUD_MMS

    bucket_prefix = bucket_prefix.removeprefix("s3://").strip("/")
    bucket_name, _, prefix = bucket_prefix.partition("/")

    return bucket_name, prefix


def _file_time(time_tag: str) -> datetime.datetime:
    r"""Start time of a file from its time tag (YYYYMMDD[hh[mm[ss]]])."""
    return datetime.datetime.strptime(time_tag.ljust(14, "0"), "%Y%m%d%H%M%S")


def list_files_aws(
    tint: list[str],
    mms_id: Union[str, int],
    var: Mapping[str, str],
    bucket_prefix: Optional[str] = "",
) -> list[dict[str, Any]]:
    r"""List files from Amazon Web Services (AWS).

    Find available files in an AWS S3 bucket for the target instrument, data type,
    data rate, mms_id and level during the target time interval. By default, the
    files are read from the public copy of the MMS archive on NASA HelioCloud,
    which doesn't need AWS credentials (requests are then anonymous).

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
        Bucket and key prefix of the MMS data, as "bucket/prefix" (or
        "s3://bucket/prefix"). Default uses the "aws" entry of
        `pyrfu/mms/config.json`, or HelioCloud
        ("gov-nasa-hdrl-data1/spdf/cdaweb/data/mms") if it is empty.

    Returns
    -------
    files_out : list of dict
        Files starting within the time interval, and the last one starting before
        it (which covers its start), in time order, with keys "s3_obj" (S3 object
        summary), "timetag" (start time, ISO 8601), "full_name" (key) and
        "file_size" (bytes). Only the latest version of each file is kept.

    Raises
    ------
    FileNotFoundError
        If the bucket can't be listed (e.g., it doesn't exist or access is denied).
    TypeError
        If the time interval is not a list or if tint values are not str.

    Notes
    -----
    The directories follow the SDC convention
    (spacecraft/instrument/rate/level[/datatype]/year/month/), with burst files in
    the month directory, as on HelioCloud, or in a day directory below, as at the
    SDC.

    """
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

    t_start, t_end = [_file_time(re.sub(r"\D", "", t)[:14]) for t in tint_iso8601]

    if not isinstance(mms_id, str):
        mms_id = str(mms_id)

    bucket_name, prefix = _bucket_and_prefix(bucket_prefix)
    bucket = _s3_resource().Bucket(bucket_name)

    # Directories and file names (SDC convention):
    # - spacecraft/instrument/rate/level[/datatype]/year/month/ (HelioCloud keeps
    #   burst files there too; the SDC adds a day/ directory for burst files)
    # - spacecraft_instrument_rate_level[_datatype]_YYYYMMDD[hhmmss]_vX.Y.Z.cdf
    dtype = var.get("dtype") or ""
    directory = [prefix, f"mms{mms_id}", var["inst"], var["tmmode"], var["lev"]]
    directory += [dtype] if dtype else []
    stem = "_".join([f"mms{mms_id}", var["inst"], var["tmmode"], var["lev"]])
    stem += f"_{dtype}" if dtype else ""
    file_regex = re.compile(
        rf"^{re.escape(stem)}_([0-9]{{8,14}})_v(\d+)\.(\d+)\.(\d+)\.cdf$"
    )

    # Files of each day, from the day before the start of the time interval
    # (a file starting then can cover it) to its end. The file name prefix
    # restricts the listing to that day on the server side.
    day = datetime.datetime(t_start.year, t_start.month, t_start.day)
    day -= datetime.timedelta(days=1)
    files = {}

    while day < t_end:
        month_dir = posixpath.join(*directory, day.strftime("%Y"), day.strftime("%m"))
        file_prefix = f"{stem}_{day.strftime('%Y%m%d')}"
        key_prefixes = [posixpath.join(month_dir, file_prefix)]

        if var["tmmode"] == "brst":
            key_prefixes.append(
                posixpath.join(month_dir, day.strftime("%d"), file_prefix)
            )

        for key_prefix in key_prefixes:
            try:
                objects = list(bucket.objects.filter(Prefix=key_prefix))
            except ClientError as err:
                code = err.response.get("Error", {}).get("Code", "")
                raise FileNotFoundError(
                    f"Cannot list s3://{bucket_name}/{key_prefix} ({code})"
                ) from err

            for obj in objects:
                matches = file_regex.match(posixpath.basename(obj.key))

                if not matches:
                    continue

                time_tag, *version = matches.groups()
                version = tuple(map(int, version))

                # Keep the latest version of each file
                if time_tag not in files or version > files[time_tag][0]:
                    files[time_tag] = (version, obj)

            # The SDC burst day directory is only looked at if there is no file
            # directly in the month directory
            if objects:
                break

        day += datetime.timedelta(days=1)

    # Files starting within the time interval, and the last one starting before
    # (or at) its start, which covers it
    times = sorted(files, key=_file_time)
    starts = [_file_time(time_tag) for time_tag in times]
    i_first = max([i for i, start in enumerate(starts) if start <= t_start] or [0])

    files_out = []

    for time_tag, start in zip(times[i_first:], starts[i_first:]):
        if start >= t_end:
            break

        obj = files[time_tag][1]
        files_out.append(
            {
                "s3_obj": obj,
                "timetag": start.isoformat(),
                "full_name": obj.key,
                "file_size": obj.size,
            },
        )

    return files_out
