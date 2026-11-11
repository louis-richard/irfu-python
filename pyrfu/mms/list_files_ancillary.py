#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import glob
import os
import re
from typing import Optional, Union

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime import iso86012datetime
from ..pyrf.iso86012datetime64 import iso86012datetime64
from .db_init import _read_config

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

# Products of the whole constellation: one file for the four spacecraft
# (MMS_DEFQ_2017187_2017188.V00, in ancillary/mms/defq; sc_id=mms on the SDC)
_CONSTELLATION_PRODUCTS = ("defq", "predq")


def _sc_name(product: str, mms_id: Union[str, int]) -> str:
    r"""Spacecraft part of the names of the files of an ancillary product:
    mms1-4, or mms for the products of the constellation."""
    return "mms" if product.lower() in _CONSTELLATION_PRODUCTS else f"mms{mms_id}"


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
        Spacecraft index (not used for "defq" and "predq", which cover the
        four spacecraft).
    product : str
        Ancillary type: {"predatt", "predeph", "predq", "defatt", "defeph",
        "defq"}.
    data_path : str, Optional
        Path of MMS data. Default uses the local path of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

    Returns
    -------
    file_names : list
        Ancillary files in interval.

    Raises
    ------
    FileNotFoundError
        If the data path doesn't exist.

    """

    # Check path
    root_path = os.path.normpath(data_path or _read_config()["local"])

    # Make sure that the data path exists
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"{root_path} doesn't exist")

    mms_id = str(mms_id)

    # Check time interval
    if isinstance(tint, list):
        tint_array = np.array(tint)
    else:
        raise TypeError("tint must be a list!!")

    # Convert time interval to ISO 8601
    if isinstance(tint_array[0], str):
        tint_iso8601 = datetime642iso8601(iso86012datetime64(tint_array))
        t_start, t_end = iso86012datetime(tint_iso8601)
    else:
        raise TypeError("Values must be in str!!")

    # PAD time interval to handle ancillary file start after midnight
    t_start -= datetime.timedelta(days=1)

    # directory and file name search patterns
    # For now
    # -all ancillary data is in one directory:
    #       mms\ancillary
    # -assume file names are of the form:
    #   SPACECRAFT_FILETYPE_startDate_endDate.version
    #   where SPACECRAFT is [MMS1, MMS2, MMS3, MMS4] in uppercase, or MMS for
    #   the products of the constellation (DEFQ, PREDQ)
    #   and FILETYPE is either DEFATT, PREDATT, DEFEPH, PREDEPH, DEFQ, PREDQ
    #   in uppercase
    #   and start/endDate is YYYYDOY
    #   and version is Vnn (.V00, .V01, etc..)
    sc_name = _sc_name(product, mms_id)
    stem = f"{sc_name.upper()}_{product.upper()}"
    directory = os.path.join(root_path, "ancillary", sc_name, product)

    # find all files in the directory (escaped: a data path with glob
    # characters, e.g. mms_data[2], found no file)
    files = glob.glob(os.path.join(glob.escape(directory), f"{stem}_*.V??"))

    # Latest version of each file in the time interval, the file names matched
    # without the directory (the version was picked by fnmatch on the paths)
    file_regex = re.compile(
        rf"{re.escape(stem)}_([0-9]{{7}})_([0-9]{{7}})\.V([0-9]{{2}})"
    )
    latest = {}

    for file in files:
        matches = file_regex.fullmatch(os.path.basename(file))

        if matches is None:
            continue

        start, end, version = matches.groups()
        start_time = datetime.datetime.strptime(start, "%Y%j")
        end_time = datetime.datetime.strptime(end, "%Y%j")

        if start_time < t_end and end_time >= t_start:
            if (start, end) not in latest or version > latest[start, end][0]:
                latest[start, end] = (version, file)

    file_names = sorted(file for _, file in latest.values())

    return file_names
