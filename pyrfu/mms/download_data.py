#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
import warnings
from shutil import copy, copyfileobj
from tempfile import NamedTemporaryFile

# 3rd party imports
import tqdm
from dateutil.parser import parse

# Local imports
from .get_data import _var_and_cdf_name
from .list_files_sdc import _login_lasp, list_files_sdc

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def _make_path_local(file, var, mms_id, data_path: str = ""):
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

    return out_path, out_file


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

    sdc_session, headers, _ = _login_lasp(login, password)

    var, _ = _var_and_cdf_name(var_str, mms_id)

    files_in_interval = list_files_sdc(tint, mms_id, var, login, password)

    for file in files_in_interval:
        out_path, out_file = _make_path_local(file, var, mms_id, data_path)

        logging.info(
            "Downloading %s from %s...", os.path.basename(out_file), file["url"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            with sdc_session.get(
                file["url"],
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
