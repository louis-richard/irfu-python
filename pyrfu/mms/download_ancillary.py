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

from .list_files_ancillary_sdc import list_files_ancillary_sdc

# Local imports
from .list_files_sdc import _login_lasp

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

LASP_PUBL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"
LASP_SITL = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"


def _make_path_local(file, product, mms_id, data_path: str = ""):
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

    return out_path, out_file


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

    sdc_session, headers, _ = _login_lasp(login, password)

    files_in_interval = list_files_ancillary_sdc(tint, mms_id, product, login, password)

    for file in files_in_interval:
        out_path, out_file = _make_path_local(
            file,
            product,
            mms_id,
            data_path,
        )

        logging.info(
            "Downloading %s from %s...", os.path.basename(out_file), file["url"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            fsrc = sdc_session.get(
                file["url"],
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
