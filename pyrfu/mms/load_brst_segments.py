#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import csv
import io
import warnings

# Third party imports
import numpy as np
import pycdfpp
import requests

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

URL = (
    "https://lasp.colorado.edu/mms/sdc/public/service/latis/"
    "mms_burst_data_segment.csv"
)

# TAI nanoseconds since 1958-01-01 at J2000 TT (TT2000 = 0)
TAI_J2000_NS = 1325419167816000000

# There appears to be an extra 10 seconds of data, consistently, not included
# in the segment end times (as pyspedas mms_load_brst_segments)
END_OFFSET = 10


def _datetime642tai(time):
    tt2000 = pycdfpp.to_tt2000(time.astype("datetime64[ns]"))["nseconds"]
    return (tt2000.astype(np.int64) + TAI_J2000_NS) // 10**9


def _tai2datetime64(tai):
    tt2000 = np.asarray(tai, dtype=np.int64) * 10**9 - TAI_J2000_NS
    return pycdfpp.to_datetime64(tt2000.view([("nseconds", "<i8")]))


def load_brst_segments(
    tint, data_path: str = None, download: bool = None, timeout: float = 60.0
):
    r"""Load burst segment time intervals associated with the input time
    interval `tint`.

    The burst segments are read from the MMS SDC burst data segment service
    (as pyspedas mms_load_brst_segments). Only the complete segments are kept,
    and 10 s are added to their end times.

    Parameters
    ----------
    tint : list
        Time interval to look for burst segments.
    data_path : str, Optional
        Deprecated and ignored: the segments are not cached any more.
    download : bool, Optional
        Deprecated and ignored: the segments are always downloaded.
    timeout : float, Optional
        Timeout of the request in seconds. Default is 60.

    Returns
    -------
    brst_segments : list
        Segments of burst mode data overlapping `tint`, as [start, end] in
        ISO 8601 format, sorted by start time.

    Raises
    ------
    requests.HTTPError
        If the request to the MMS SDC fails.

    """

    if data_path is not None or download is not None:
        warnings.warn(
            "data_path and download are deprecated and ignored, and will be removed "
            "in a future version: the burst segments are read from the MMS SDC.",
            FutureWarning,
            stacklevel=2,
        )

    l_bound, r_bound = iso86012datetime64(np.array(tint))

    # Segments overlapping tint (with the end offset)
    tai_l = _datetime642tai(np.array([l_bound]))[0] - END_OFFSET
    tai_r = _datetime642tai(np.array([r_bound]))[0]
    query = f"?TAISTARTTIME<={tai_r}&TAIENDTIME>={tai_l}"

    response = requests.get(URL + query, timeout=timeout)
    response.raise_for_status()

    reader = csv.reader(io.StringIO(response.text))
    header = next(reader)
    i_start, i_end, i_status = [
        [i for i, name in enumerate(header) if name.startswith(key)][0]
        for key in ["TAISTARTTIME", "TAIENDTIME", "STATUS"]
    ]

    rows = [row for row in reader if row and row[i_status] == "COMPLETE+FINISHED"]

    if not rows:
        return []

    tai_start = np.array([int(row[i_start]) for row in rows], dtype=np.int64)
    tai_end = np.array([int(row[i_end]) for row in rows], dtype=np.int64)

    idx = np.argsort(tai_start, kind="stable")
    start = _tai2datetime64(tai_start[idx])
    end = _tai2datetime64(tai_end[idx] + END_OFFSET)

    # Select the pairs together so that starts and ends stay matched
    in_tint = (end >= l_bound) & (start <= r_bound)

    brst_segments = [
        [str(t_) for t_ in datetime642iso8601(np.array([t_s, t_e]))]
        for t_s, t_e in zip(start[in_tint], end[in_tint])
    ]

    return brst_segments
