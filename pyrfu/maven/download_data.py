#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
import warnings
from datetime import datetime, timedelta

# 3rd party imports
import numpy as np
import requests

# Local imports
from ..mms.download_data import _download_file
from ..mms.list_files_sdc import SDC_TIMEOUT
from .db_init import config_path

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.10"
__status__ = "Prototype"

logger = logging.getLogger(__name__)

LASP_PUBL = "https://lasp.colorado.edu/maven/sdc/public/files/api/v1/"


def _login_lasp():
    r"""Session to the public MAVEN SDC (a POST to lasp.colorado.edu, with
    the login and password, was made first, unused)."""
    return requests.Session()


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
        # Read the configuration file (user directory)
        with open(config_path(), "r", encoding="utf-8") as fs:
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


def _wanted(file_name: str, var: dict) -> bool:
    r"""File of the request: of the plan (MAG coordinates, the end of the
    fourth field of the MAG file names, e.g. "pl" in
    mvn_mag_l2_2015001pl_20150101_v01_r01.sts) and of the file extension,
    when var has them. With a plan and no extension, the .sts files (only the
    MAG .sts files with the plan were downloaded, KeyError without it)."""
    if "plan" in var and file_name.split("_")[3][7:] != var["plan"]:
        return False

    extension = var.get("file_extension", "sts" if "plan" in var else None)
    return extension is None or file_name.endswith(f".{extension}")


def download_data(var, tint, login: str = "", password: str = "", data_path: str = ""):
    r"""Downloads files containing field `var_str` over the time interval
    `tint`. The files are saved to `data_path`.

    Parameters
    ----------
    var : dict
        Hashtable containing:

        - inst: instrument acronym (acc, euv, iuv, kp, lpw, mag, ngi, pfp, sep,
          sta, swe, swi)
        - level: data levels (l1a, l1b, l2, l3, insitu (KP only), iuv (KP only))
        - plan (Optional): MAG coordinates, e.g. "pl" (planetocentric) or
          "ss" (sun-state), the end of the fourth field of the MAG file names
          (mvn_mag_l2_2015001pl_20150101_v01_r01.sts); with a plan, the .sts
          files unless file_extension is given
        - file_extension (Optional): e.g. "cdf" or "sts"

        Without plan and file_extension, all the files found are downloaded.

    tint : list of str
        Time interval.
    login : str, Optional
        Login to LASP MAVEN: not implemented, the public data
        (https://lasp.colorado.edu/maven/sdc/public/) are downloaded.
    password : str, Optional
        Password to LASP MAVEN: not implemented, see `login`.
    data_path : str, Optional
        Path of MAVEN data. Default uses the local data directory of the
        MAVEN configuration (see `pyrfu.maven.db_init`).

    Raises
    ------
    FileNotFoundError
        If no file is found, or none of the plan and file extension (only the
        MAG .sts files of the plan were downloaded: the other instruments
        downloaded nothing, silently).
    requests.HTTPError
        If the search or a download fails (the error page was saved as the
        file).

    Notes
    -----
    Each file is downloaded to a temporary file next to it, moved to its name
    when complete: a failed download leaves no file (a temporary copy of each
    file was left in the temporary directory).

    """

    lasp_url = LASP_PUBL

    if login or password:
        logger.info("login not implemented: the public data are downloaded")

    sdc_session = _login_lasp()

    headers = {"User-Agent": "pyrfu"}

    url = _construct_url(tint, var, lasp_url)

    # (no timeout, and an error page was taken as the listing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        response = sdc_session.get(
            url, verify=True, headers=headers, timeout=SDC_TIMEOUT
        )
        response.raise_for_status()
        http_json = response.json()

    files = [f for f in http_json["files"] if _wanted(f["file_name"], var)]

    if not files:
        raise FileNotFoundError("No files to download!!")

    for file in files:
        _, out_file, dwl_url = _make_path(file, var, lasp_url, data_path)

        logger.info("Downloading %s from %s...", file["file_name"], dwl_url)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            _download_file(
                sdc_session, dwl_url, headers, out_file, file.get("file_size")
            )

    sdc_session.close()
