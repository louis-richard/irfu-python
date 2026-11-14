#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Built-in imports
import functools
import re
import urllib.parse
import warnings
from bisect import bisect_left
from datetime import timedelta

# 3rd party imports
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.iso86012datetime64 import iso86012datetime64
from ..pyrf.time_clip import _time_interval
from .db_init import _config_stamp, _get_credential, _read_config
from .list_files_aws import _tint_bounds

LASP_PUBL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"
LASP_SITL = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"

# Test url from example in https://lasp.colorado.edu/mms/sdc/public/about/how-to/
TEST_URL = "file_names/science?start_date=2015-04-10&end_date=2015-04-11&sc_id=mms2"


# Timeouts of the SDC requests [s]: to connect, and between bytes (not for the
# whole download, so that large files can be downloaded)
SDC_TIMEOUT = (30, 300)


def _login_lasp():
    r"""Login to LASP colorado. The session is cached, so that the
    connectivity/credential probe only happens once, until the MMS configuration
    file changes (e.g., after mms.db_init). The session is shared: do not close it.
    """
    return _login_lasp_cached(_config_stamp())


@functools.lru_cache(maxsize=1)
def _login_lasp_cached(config_stamp):
    r"""Login to LASP colorado (cached by the stamp of the configuration file,
    see db_init._config_stamp)."""
    del config_stamp  # cache key only

    config = _read_config()

    session = requests.Session()

    # The credentials are only read and sent for the SITL (team) access, never
    # to the public SDC
    if config["sdc"]["rights"] == "public":
        lasp_url = LASP_PUBL
    elif config["sdc"]["rights"] == "sitl":
        credential = _get_credential(config["sdc"]["username"])

        if not credential or not credential.username or not credential.password:
            raise EnvironmentError(
                "Incomplete credentials please update using mms.db_init()"
            )

        lasp_url = LASP_SITL
        session.auth = (credential.username, credential.password)
    else:
        raise EnvironmentError(
            "Invalid MMS SDC rights, please update using mms.db_init()"
        )

    # Retry with backoff on rate limiting / transient server errors,
    # honoring the Retry-After header when LASP sends one.
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    headers = {"User-Agent": "pyrfu"}

    response = session.get(
        urllib.parse.urljoin(lasp_url, TEST_URL),
        verify=True,
        timeout=5,
        headers=headers,
    )
    response.raise_for_status()

    return session, headers, lasp_url


def _construct_url_json_list(tint, mms_id, var, lasp_url):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    # Files starting before the end (the query end is inclusive), to the second
    # above (a file starting in its last fraction of a second was left out)
    t_start, t_end = _tint_bounds(tint)
    start_date = t_start.strftime("%Y-%m-%d")
    end_date = (t_end - timedelta(seconds=1)).strftime("%Y-%m-%d-%H-%M-%S")

    # lasp_url ends with "/" (it was doubled)
    url = f"{lasp_url}file_info/science"
    url = f"{url}?start_date={start_date}&end_date={end_date}&sc_id=mms{mms_id}"

    url = f"{url}&instrument_id={var['inst']}"
    url = f"{url}&data_rate_mode={var['tmmode']}"
    url = f"{url}&data_level={var['lev']}"

    if var["dtype"]:
        url = f"{url}&descriptor={var['dtype']}"

    return url


def _file_start(digits):
    r"""Start (UTC, datetime64) of a file from the digits of its name,
    YYYYMMDD or YYYYMMDDhhmmss (dateutil read them as local time)."""
    hms = digits[8:].ljust(6, "0")
    return np.datetime64(
        f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}T{hms[:2]}:{hms[2:4]}:{hms[4:6]}",
        "ns",
    )


def _files_in_interval(in_files, trange):
    r"""Filters the file list returned by the SDC to the requested time
    range. This filter is purposefully liberal, it regularly grabs an extra
    file due to special cases
    """

    file_name = r"mms.*_([0-9]{8,14})_v(\d+).(\d+).(\d+).cdf"

    file_times = []

    regex = re.compile(file_name)

    for file in in_files:
        matches = regex.match(file["file_name"])
        if matches:
            file_times.append(
                (
                    file["file_name"],
                    _file_start(matches.groups()[0]),
                    file["timetag"],
                    file["file_size"],
                ),
            )

    # sort in time
    sorted_files = sorted(file_times, key=lambda x: x[1])

    times = [t[1] for t in sorted_files]

    # All in UTC: dateutil read the times as local time, so that on the night
    # of a change to summer time the files were out of order and some holding
    # data left out
    idx_min = bisect_left(times, iso86012datetime64(np.array([trange[0]]))[0])

    def mkout(f):
        return {"file_name": f[0], "timetag": f[2], "size": f[3]}

    if idx_min == 0:
        files = list(map(mkout, sorted_files[idx_min:]))
    else:
        files = list(map(mkout, sorted_files[idx_min - 1 :]))

    return files


def _make_urls_cdfs(lasp_url, files):
    for file in files:
        file["url"] = f"{lasp_url}download/science?file={file['file_name']}"

    return files


def list_files_sdc(tint, mms_id, var):
    r"""Find availables files from the LASP SDC for the target instrument,
    data type, data rate, mms_id and level during the target time interval.

    Parameters
    ----------
    tint : array_like or xarray.DataArray
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`.
    mms_id : str or int
        Index of the spacecraft
    var : dict
        Dictionary containing 4 keys
            * var["inst"] : name of the instrument
            * var["tmmode"] : data rate
            * var["lev"] : data level
            * var["dtype"] : data type

    The SDC rights (public or SITL) and username are those of the MMS
    configuration (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

    Returns
    -------
    file_names : list
        List of files corresponding to the parameters in the selected time
        interval

    """

    # Time interval in ISO 8601, from the forms of time_clip (a time series
    # failed)
    tint = datetime642iso8601(np.array(_time_interval(tint)))

    # Start session on MMS's SDC
    sdc_session, headers, lasp_url = _login_lasp()

    url_json_cdfs = _construct_url_json_list(tint, mms_id, var, lasp_url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        response = sdc_session.get(
            url_json_cdfs, verify=True, headers=headers, timeout=SDC_TIMEOUT
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses
        http_json = response.json()

    file_names = _files_in_interval(http_json["files"], tint)

    file_names = _make_urls_cdfs(lasp_url, file_names)

    return file_names
