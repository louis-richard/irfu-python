#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import re
import warnings
from bisect import bisect_left
from datetime import datetime, timedelta

# 3rd party imports
import numpy as np
import requests
from dateutil.parser import parse

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

LASP_PUBL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"
LASP_SITL = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"


def _login_lasp(user: str = "", password: str = ""):
    r"""Login to LASP colorado."""

    if not user:
        lasp_url = LASP_PUBL
    else:
        lasp_url = LASP_SITL

    session = requests.Session()
    session.auth = (user, password)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        _ = session.post("https://lasp.colorado.edu", verify=True, timeout=5)
        testget = session.get(lasp_url, verify=True, timeout=5)

    assert testget != "401", "Login failed!!"

    headers = {"User-Agent": "pyrfu"}

    return session, headers, lasp_url


def _construct_url_json_list(tint, mms_id, var, lasp_url):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    tint = np.array(tint).astype("<M8[ns]").astype(str)
    tint = [datetime.strptime(t_[:-3], "%Y-%m-%dT%H:%M:%S.%f") for t_ in tint]
    start_date = tint[0].strftime("%Y-%m-%d")
    end_date = (tint[1] - timedelta(seconds=1)).strftime("%Y-%m-%d-%H-%M-%S")

    url = f"{lasp_url}/file_info/science"
    url = f"{url}?start_date={start_date}&end_date={end_date}&sc_id=mms{mms_id}"

    url = f"{url}&instrument_id={var['inst']}"
    url = f"{url}&data_rate_mode={var['tmmode']}"
    url = f"{url}&data_level={var['lev']}"

    if var["dtype"]:
        url = f"{url}&descriptor={var['dtype']}"

    return url


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
                    parse(matches.groups()[0]).timestamp(),
                    file["timetag"],
                    file["file_size"],
                ),
            )

    # sort in time
    sorted_files = sorted(file_times, key=lambda x: x[1])

    times = [t[1] for t in sorted_files]

    idx_min = bisect_left(times, parse(trange[0]).timestamp())

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


def list_files_sdc(tint, mms_id, var, login: str = "", password: str = ""):
    r"""Find availables files from the LASP SDC for the target instrument,
    data type, data rate, mms_id and level during the target time interval.

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
    login : str, Optional
        Login to LASP MMS SITL. Default downloads from
        https://lasp.colorado.edu/mms/sdc/public/
    password : str, Optional
        Password to LASP MMS SITL. Default downloads from
        https://lasp.colorado.edu/mms/sdc/public/

    Returns
    -------
    file_names : list
        List of files corresponding to the parameters in the selected time
        interval

    """
    sdc_session, headers, lasp_url = _login_lasp(login, password)

    url_json_cdfs = _construct_url_json_list(tint, mms_id, var, lasp_url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        http_json = sdc_session.get(url_json_cdfs, verify=True, headers=headers).json()

    file_names = _files_in_interval(http_json["files"], tint)
    sdc_session.close()

    file_names = _make_urls_cdfs(lasp_url, file_names)

    return file_names
