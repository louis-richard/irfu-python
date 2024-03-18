#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
import warnings
from shutil import copy, copyfileobj
from tempfile import NamedTemporaryFile
from typing import Literal, Optional, Union

# 3rd party imports
import tqdm

# Local imports
from pyrfu.mms.db_init import MMS_CFG_PATH
from pyrfu.mms.list_files_ancillary_sdc import list_files_ancillary_sdc
from pyrfu.mms.list_files_sdc import _login_lasp

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

LASP_PUBL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"
LASP_SITL = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"


def _make_path_local(
    file: dict,
    product: Literal["predatt", "predeph", "defatt", "defeph"],
    mms_id: Union[str, int],
    data_path: Optional[str] = "",
) -> str:
    r"""Construct path of the data file using the standard convention.

    Parameters
    ----------
    file : dict
        File information.
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    mms_id : str or int
        Spacecraft index.
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu/mms/config.json`.

    Returns
    -------
    str
        Full path of the data file.

    Raises
    ------
    FileNotFoundError
        If the local data directory doesn't exist.

    """
    if not data_path:
        # Read the current version of the MMS configuration file
        with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local"])
    else:
        data_path = os.path.normpath(data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError("local data directory doesn't exist!")

    path_list = [
        data_path,
        "ancillary",
        f"mms{mms_id}",
        product,
    ]

    return os.path.join(*path_list, file["file_name"])


def download_ancillary(
    product: Literal["predatt", "predeph", "defatt", "defeph"],
    tint: list,
    mms_id: Union[str, int],
    data_path: Optional[str] = "",
):
    r"""Download files from MMS SDC.

    Download ancillary files containing field `var_str` over the time interval `tint`
    for the spacecraft `mms_id` to `data_path`.

    Parameters
    ----------
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    tint : list
        Time interval
    mms_id : str or int
        Spacecraft index
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu/mms/config.json`

    """
    # List files in MMS SDC that match the request
    files_in_interval = list_files_ancillary_sdc(tint, mms_id, product)

    # Start session on MMS SDC ("public" or "sitl")
    sdc_session, headers, _ = _login_lasp()

    for file in files_in_interval:
        # Create local path following tree structure for the CDF files
        out_file = _make_path_local(file, product, mms_id, data_path)
        out_path = os.path.dirname(out_file)

        logging.info(
            "Downloading %s from %s...", os.path.basename(out_file), file["url"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            fsrc = sdc_session.get(
                file["url"], stream=True, verify=True, headers=headers
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
