#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import re
import urllib
import warnings
from bisect import bisect_left
from datetime import datetime, timedelta

# 3rd party imports
import keyring
import numpy as np
import requests
from dateutil.parser import parse

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from .db_init import MMS_CFG_PATH

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

LASP_PUBL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"
LASP_SITL = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"

# Test url from example in https://lasp.colorado.edu/mms/sdc/public/about/how-to/
TEST_URL = "file_names/science?start_date=2015-04-10&end_date=2015-04-11&sc_id=mms2"


def _login_lasp():
    r"""Login to LASP colorado."""

    with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
        config = json.load(fs)

    # Read credentials for username
    credential = keyring.get_credential("mms-sdc", config["sdc"]["username"])

    if credential:
        username, password = credential.username, credential.password
    else:
        username, password = "", ""

    if config["sdc"]["rights"] == "public":
        lasp_url = LASP_PUBL
    elif config["sdc"]["rights"] == "sitl" and username and password:
        lasp_url = LASP_SITL
    else:
        raise EnvironmentError(
            "Incomplete credentials please update using mms.db_init()"
        )

    session = requests.Session()
    session.auth = (username, password)

    headers = {"User-Agent": "pyrfu"}

    try:
        _ = session.post("https://lasp.colorado.edu", verify=True, timeout=5)
        response = session.get(
            urllib.parse.urljoin(lasp_url, TEST_URL),
            verify=True,
            timeout=5,
            headers=headers,
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.RequestException as e:
        print(f"Error login to {lasp_url}: {e}")

    return session, headers, lasp_url


def _construct_url_json_list(tint, mms_id, var, lasp_url):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

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


def list_files_sdc(tint, mms_id, var):
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
    username : str, Optional
        Login to LASP MMS SITL. Default downloads from
        https://lasp.colorado.edu/mms/sdc/public/

    Returns
    -------
    file_names : list
        List of files corresponding to the parameters in the selected time
        interval

    """

    # Make sure time interval is in iso8601 string format
    tint = np.array(tint).astype("datetime64[ns]")
    tint = datetime642iso8601(tint)

    # Start session on MMS's SDC
    sdc_session, headers, lasp_url = _login_lasp()

    url_json_cdfs = _construct_url_json_list(tint, mms_id, var, lasp_url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        http_json = sdc_session.get(url_json_cdfs, verify=True, headers=headers).json()

    file_names = _files_in_interval(http_json["files"], tint)
    sdc_session.close()

    file_names = _make_urls_cdfs(lasp_url, file_names)

    return file_names
