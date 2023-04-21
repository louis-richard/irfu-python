#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os
import warnings

from datetime import datetime, timedelta
from shutil import copyfileobj, copy
from tempfile import NamedTemporaryFile

import pkg_resources

# 3rd party imports
import numpy as np
import requests
import tqdm

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.22"
__status__ = "Prototype"

LASP = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"


def _login_lasp(user: str, password: str):
    r"""Login to LASP colorado."""

    session = requests.Session()
    session.auth = (user, password)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        _ = session.post("https://lasp.colorado.edu", verify=True, timeout=5)
        testget = session.get(LASP, verify=True, timeout=5)

    assert testget != "401", "Login failed!!"

    return session, user


def _construct_url(tint, mms_id, product):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    tint = np.array(tint).astype("<M8[ns]").astype(str)
    tint = [datetime.strptime(t_[:-3], "%Y-%m-%dT%H:%M:%S.%f") for t_ in tint]
    start_date = (tint[0] - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (tint[1] + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"{LASP}/file_info/ancillary"
    url = f"{url}?start_date={start_date}&end_date={end_date}&sc_id=mms{mms_id}"

    url = f"{url}&product={product}"

    return url


def _make_path(file, product, mms_id, data_path: str = ""):
    r"""Construct path of the data file using the standard convention."""

    if not data_path:
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    assert os.path.exists(data_path), "local data directory doesn't exist!"

    path_list = [
        data_path,
        "ancillary",
        f"mms{mms_id}",
        product,
    ]

    out_path = os.path.join(*path_list)
    out_file = os.path.join(*path_list, file["file_name"])

    download_url = f"{LASP}download/ancillary?file={file['file_name']}"

    return out_path, out_file, download_url


def download_ancillary(
    product,
    tint,
    mms_id,
    login,
    password,
    data_path: str = "",
):
    r"""Downloads files containing field `var_str` over the time interval
    `tint` for the spacecraft `mms_id`. The files are saved to `data_path`.

    Parameters
    ----------
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    tint : list of str
        Time interval
    mms_id : str or int
        Spacecraft index
    login : str
        Login to LASP.
    password : str
        Password to LASP.
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu/mms/config.json`

    """

    sdc_session, _ = _login_lasp(login, password)

    headers = {}
    try:
        release_version = pkg_resources.get_distribution("pyrfu").version
    except pkg_resources.DistributionNotFound:
        release_version = "bleeding edge"

    headers["User-Agent"] = f"pyrfu {release_version}"

    url = _construct_url(tint, mms_id, product)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        http_json = sdc_session.get(url, verify=True, headers=headers).json()

    files_in_interval = http_json["files"]

    for file in files_in_interval:
        out_path, out_file, dwl_url = _make_path(
            file,
            product,
            mms_id,
            data_path,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            fsrc = sdc_session.get(
                dwl_url,
                stream=True,
                verify=True,
                headers=headers,
            )

        ftmp = NamedTemporaryFile(delete=False)

        with tqdm.tqdm.wrapattr(
            fsrc.raw,
            "read",
            total=file["file_size"],
        ) as fsrc_raw:
            with open(ftmp.name, "wb") as fs:
                copyfileobj(fsrc_raw, fs)

        os.makedirs(out_path, exist_ok=True)

        # if the download was successful, copy to data directory
        copy(ftmp.name, out_file)
        fsrc.close()
        ftmp.close()

    sdc_session.close()
