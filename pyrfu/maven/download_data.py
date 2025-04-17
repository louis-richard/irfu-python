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
import requests
import tqdm

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.10"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


LASP_PUBL = "https://lasp.colorado.edu/maven/sdc/public/files/api/v1/"


def _login_lasp(user: str, password: str):
    r"""Login to LASP colorado."""

    session = requests.Session()
    session.auth = (user, password)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        _ = session.post("https://lasp.colorado.edu", verify=True, timeout=5)

    return session


def _construct_url(tint, var, lasp_url):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    tint = np.array(tint).astype("<M8[ns]").astype(str)
    tint = [datetime.strptime(t_[:-3], "%Y-%m-%dT%H:%M:%S.%f") for t_ in tint]
    start_date = tint[0].strftime("%Y-%m-%d")
    end_date = tint[1].strftime("%Y-%m-%d")

    # If start and end are on the same add one day to end date
    if end_date == start_date:
        tint[1] += timedelta(days=1)
        end_date = tint[1].strftime("%Y-%m-%d")

    url = f"{lasp_url}search/science/fn_metadata/file_info?"

    url = f"{url}instrument={var['inst']}"
    url = f"{url}&level={var['level']}"
    url = f"{url}&start_date={start_date}&end_date={end_date}"

    if "file_extension" in var:
        url = f"{url}&file_extension={var['file_extension']}"

    # if "plan" in var:
    #     url = f"{url}&plan={var['plan']}"

    return url


def _make_path(file, var, lasp_url, data_path: str = ""):
    r"""Construct path of the data file using the standard convention."""

    if not data_path:
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MAVEN configuration file
        with open(os.path.join(pkg_path, "config.json"), "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    assert os.path.exists(data_path), "local data directory doesn't exist!"

    path_list = [
        data_path,  # root path to data
        var["inst"],  # instrument sub-directory
        var["level"],  # data level sub-directory
        file["file_name"].split("_")[4][:4],  # year sub-directory
        file["file_name"].split("_")[4][4:6],  # month sub-directory
    ]

    out_path = os.path.join(*path_list)
    out_file = os.path.join(*path_list, file["file_name"])

    download_url = (
        f"{lasp_url}search/science/fn_metadata/download?file={file['file_name']}"
    )

    return out_path, out_file, download_url


def download_data(var, tint, login: str = "", password: str = "", data_path: str = ""):
    r"""Downloads files containing field `var_str` over the time interval
    `tint`. The files are saved to `data_path`.

    Parameters
    ----------
    var : dict
        Hashtable containing:
            - inst: instrument acronym (acc, euv, iuv, kp, lpw, mag, ngi, pfp, sep, sta,
             swe, swi)
            - level: data levels (l1a, l1b, l2, l3, insitu (KP only), iuv (KP only))
    tint : list of str
        Time interval.
    login : str, Optional
        Login to LASP MAVEN. Default downloads from
        https://lasp.colorado.edu/maven/sdc/public/
    password : str, Optional
        Password to LASP MAVEN. Default downloads from
        https://lasp.colorado.edu/maven/sdc/public/
    data_path : str, Optional
        Path of MAVEN data. If None use `pyrfu/mms/config.json`

    """

    if not login:
        lasp_url = LASP_PUBL
    else:
        lasp_url = LASP_PUBL
        logging.info("login not impleted. Use public instead")

    sdc_session = _login_lasp(login, password)

    headers = {"User-Agent": "pyrfu"}

    url = _construct_url(tint, var, lasp_url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        http_json = sdc_session.get(url, verify=True, headers=headers).json()

    if not http_json["files"]:
        raise FileNotFoundError("No files to download!!")

    for file in http_json["files"]:
        out_path, out_file, dwl_url = _make_path(file, var, lasp_url, data_path)
        plan = out_file.split("/")[-1].split("_")[3][7:]
        if plan == var["plan"] and out_file[-3:] == "sts":

            logging.info(
                "Downloading %s from %s...", os.path.basename(out_file), dwl_url
            )

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
                            total=file["file_size"],
                            ncols=60,
                        ) as fsrc_raw:
                            with open(ftmp.name, "wb") as fs:
                                copyfileobj(fsrc_raw, fs)

                    os.makedirs(out_path, exist_ok=True)

                    # if the download was successful, copy to data directory
                    copy(ftmp.name, out_file)

    sdc_session.close()
