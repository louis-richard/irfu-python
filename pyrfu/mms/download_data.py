#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
import re
import warnings
from bisect import bisect_left
from datetime import datetime, timedelta
from shutil import copy, copyfileobj
from tempfile import NamedTemporaryFile

# 3rd party imports
import numpy as np
import pkg_resources
import requests
import tqdm
from dateutil.parser import parse

# Local imports
from pyrfu.mms import tokenize

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

LASP_PUBL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"
LASP_SITL = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"


def _login_lasp(user: str, password: str, lasp_url: str):
    r"""Login to LASP colorado."""

    session = requests.Session()
    session.auth = (user, password)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        _ = session.post("https://lasp.colorado.edu", verify=True, timeout=5)
        testget = session.get(lasp_url, verify=True, timeout=5)

    assert testget != "401", "Login failed!!"

    return session, user


def _construct_url(tint, mms_id, var, lasp_url):
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


def _make_path(file, var, mms_id, lasp_url, data_path: str = ""):
    r"""Construct path of the data file using the standard convention."""

    file_date = parse(file["timetag"])

    if not data_path:
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    assert os.path.exists(data_path), "local data directory doesn't exist!"

    path_list = [
        data_path,
        f"mms{mms_id}",
        var["inst"],
        var["tmmode"],
        var["lev"],
        var["dtype"],
        *file_date.strftime("%Y-%m").split("-"),
    ]

    if var["tmmode"].lower() == "brst":
        path_list.append(file_date.strftime("%d"))

    out_path = os.path.join(*path_list)
    out_file = os.path.join(*path_list, file["file_name"])

    download_url = f"{lasp_url}download/science?file={file['file_name']}"

    return out_path, out_file, download_url


def download_data(
    var_str, tint, mms_id, login: str = "", password: str = "", data_path: str = ""
):
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
    login : str, Optional
        Login to LASP MMS SITL. Default downloads from
        https://lasp.colorado.edu/mms/sdc/public/
    password : str, Optional
        Password to LASP MMS SITL. Default downloads from
        https://lasp.colorado.edu/mms/sdc/public/
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu/mms/config.json`

    """

    if not login:
        lasp_url = LASP_PUBL
    else:
        lasp_url = LASP_SITL

    sdc_session, _ = _login_lasp(login, password, lasp_url)

    headers = {}
    try:
        release_version = pkg_resources.get_distribution("pyrfu").version
    except pkg_resources.DistributionNotFound:
        release_version = "bleeding edge"

    headers["User-Agent"] = f"pyrfu {release_version}"

    var = tokenize(var_str)

    root_path = os.path.dirname(os.path.abspath(__file__))

    with open(
        os.sep.join([root_path, "mms_keys.json"]), "r", encoding="utf-8"
    ) as json_file:
        keys_ = json.load(json_file)

    var["dtype"] = keys_[var["inst"]][var_str.lower()]["dtype"]

    url = _construct_url(tint, mms_id, var, lasp_url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        http_json = sdc_session.get(url, verify=True, headers=headers).json()

    files_in_interval = _files_in_interval(http_json["files"], tint)

    for file in files_in_interval:
        out_path, out_file, dwl_url = _make_path(file, var, mms_id, lasp_url, data_path)

        logging.info("Downloading %s from %s...", os.path.basename(out_file), dwl_url)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            with sdc_session.get(
                dwl_url,
                stream=True,
                verify=True,
                headers=headers,
            ) as fsrc:
                with NamedTemporaryFile(delete=False) as ftmp:
                    with tqdm.tqdm.wrapattr(
                        fsrc.raw,
                        "read",
                        total=file["size"],
                        ncols=60,
                    ) as fsrc_raw:
                        with open(ftmp.name, "wb") as fs:
                            copyfileobj(fsrc_raw, fs)

                os.makedirs(out_path, exist_ok=True)

                # if the download was successful, copy to data directory
                copy(ftmp.name, out_file)

    sdc_session.close()
