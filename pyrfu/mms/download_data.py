#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os
import pkg_resources
import re
import warnings

from bisect import bisect_left
from datetime import datetime, timedelta
from dateutil.parser import parse
from shutil import copyfileobj, copy
from tempfile import NamedTemporaryFile

# 3rd party imports
import requests
import numpy as np

from tqdm import tqdm

# Local imports
from .tokenize import tokenize

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.13"
__status__ = "Prototype"

lasp = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"


def _login_lasp(user: str, password: str):
    r"""Login to LASP colorado.
    """

    session = requests.Session()
    session.auth = (user, password)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        _ = session.post("https://lasp.colorado.edu", verify=True, timeout=5)
        testget = session.get(lasp, verify=True, timeout=5)

    assert testget != "401", "Login failed!!"

    return session, user


def _construct_url(tint, ic, var):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    tint = np.array(tint).astype("<M8[ns]").astype(str)
    tint = [datetime.strptime(t_[:-3], "%Y-%m-%dT%H:%M:%S.%f") for t_ in tint]
    start_date = tint[0].strftime("%Y-%m-%d")
    end_date = (tint[1] - timedelta(seconds=1)).strftime("%Y-%m-%d-%H-%M-%S")

    url = f"{lasp}/file_info/science"
    url = f"{url}?start_date={start_date}&end_date={end_date}&sc_id=mms{ic}"

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
            file_times.append((file["file_name"],
                               parse(matches.groups()[0]).timestamp(),
                               file["timetag"], file["file_size"]))

    # sort in time
    sorted_files = sorted(file_times, key=lambda x: x[1])

    times = [t[1] for t in sorted_files]

    idx_min = bisect_left(times, parse(trange[0]).timestamp())

    mkout = lambda f: {"file_name": f[0], "timetag": f[2], "size": f[3]}

    if idx_min == 0:
        return list(map(mkout, sorted_files[idx_min:]))
    else:
        return list(map(mkout, sorted_files[idx_min - 1:]))


def _make_path(file, var, mms_id, data_path: str = ""):
    r"""Construct path of the data file using the standard convention.
    """

    file_date = parse(file["timetag"])

    if not data_path:
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r") as f:
            config = json.load(f)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    assert os.path.exists(data_path), "local data directory doesn't exist!"

    path_list = [data_path, f"mms{mms_id}", var["inst"], var["tmmode"],
                 var["lev"], var["dtype"],
                 *file_date.strftime("%Y-%m").split("-")]

    if var["tmmode"].lower() == "brst":
        path_list.append(file_date.strftime("%d"))

    out_path = os.path.join(*path_list)
    out_file = os.path.join(*path_list, file["file_name"])

    download_url = f"{lasp}download/science?file={file['file_name']}"

    return out_path, out_file, download_url


def download_data(var_str, tint, mms_id, login, password, data_path: str = ""):
    r"""Downloads files containing field `var_str` over the time interval
    `tint` for the spacecraft `mms_id`. The files are saved to `data_path`.

    Parameters
    ----------
    var_str : str
        Input key of variable.
    tint : list of str
        Time interval.
    mms_id : str or int
        Index of the target spacecraft.
    login : str
        Login to LASP.
    password : str
        Password to LASP.
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu/mms/config.json`

    """
    sdc_session, user = _login_lasp(login, password)

    headers = {}
    try:
        release_version = pkg_resources.get_distribution("pyrfu").version
    except pkg_resources.DistributionNotFound:
        release_version = "bleeding edge"

    headers["User-Agent"] = f"pyrfu {release_version}"

    var = tokenize(var_str)

    root_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.sep.join([root_path, "mms_keys.json"]), "r") as json_file:
        keys_ = json.load(json_file)

    var["dtype"] = keys_[var["inst"]][var_str.lower()]["dtype"]

    url = _construct_url(tint, mms_id, var)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        http_json = sdc_session.get(url, verify=True, headers=headers).json()

    files_in_interval = _files_in_interval(http_json["files"], tint)

    for file in files_in_interval:
        out_path, out_file, dwl_url = _make_path(file, var, mms_id, data_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            fsrc = sdc_session.get(dwl_url, stream=True, verify=True,
                                   headers=headers)

        ftmp = NamedTemporaryFile(delete=False)

        with tqdm.wrapattr(fsrc.raw, "read", total=file["size"]) as fsrc_raw:
            with open(ftmp.name, "wb") as f:
                copyfileobj(fsrc_raw, f)

        os.makedirs(out_path, exist_ok=True)

        # if the download was successful, copy to data directory
        copy(ftmp.name, out_file)
        fsrc.close()
        ftmp.close()

    sdc_session.close()
    return
