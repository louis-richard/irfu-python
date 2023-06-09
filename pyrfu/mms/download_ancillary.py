#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from shutil import copy, copyfileobj
from tempfile import NamedTemporaryFile

# 3rd party imports
import numpy as np
import pkg_resources
import requests
import tqdm

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.22"
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


def _construct_url(tint, mms_id, product, lasp_url):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    tint = np.array(tint).astype("<M8[ns]").astype(str)
    tint = [datetime.strptime(t_[:-3], "%Y-%m-%dT%H:%M:%S.%f") for t_ in tint]
    start_date = (tint[0] - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (tint[1] + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"{lasp_url}/file_info/ancillary"
    url = f"{url}?start_date={start_date}&end_date={end_date}&sc_id=mms{mms_id}"

    url = f"{url}&product={product}"

    return url


def _make_path(file, product, mms_id, lasp_url, data_path: str = ""):
    r"""Construct path of the data file using the standard convention."""

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
        "ancillary",
        f"mms{mms_id}",
        product,
    ]

    out_path = os.path.join(*path_list)
    out_file = os.path.join(*path_list, file["file_name"])

    download_url = f"{lasp_url}download/ancillary?file={file['file_name']}"

    return out_path, out_file, download_url


def download_ancillary(
    product,
    tint,
    mms_id,
    login: str = "",
    password: str = "",
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

    url = _construct_url(tint, mms_id, product, lasp_url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        http_json = sdc_session.get(url, verify=True, headers=headers).json()

    files_in_interval = http_json["files"]

    for file in files_in_interval:
        out_path, out_file, dwl_url = _make_path(
            file,
            product,
            mms_id,
            lasp_url,
            data_path,
        )

        logging.info("Downloading %s from %s...", os.path.basename(out_file), dwl_url)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            fsrc = sdc_session.get(
                dwl_url,
                stream=True,
                verify=True,
                headers=headers,
            )

        with NamedTemporaryFile(delete=False) as ftmp:
            with tqdm.tqdm.wrapattr(
                fsrc.raw, "read", total=file["file_size"], ncols=60
            ) as fsrc_raw:
                with open(ftmp.name, "wb") as fs:
                    copyfileobj(fsrc_raw, fs)

        os.makedirs(out_path, exist_ok=True)

        # if the download was successful, copy to data directory
        copy(ftmp.name, out_file)
        fsrc.close()

    sdc_session.close()
