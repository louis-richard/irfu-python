#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
from typing import Optional, Union

# 3rd party imports
import tqdm
from dateutil.parser import parse

# Local imports
from pyrfu.mms.db_init import MMS_CFG_PATH
from pyrfu.mms.get_data import _var_and_cdf_name
from pyrfu.mms.list_files_sdc import SDC_TIMEOUT, _login_lasp, list_files_sdc

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def _make_path_local(
    file: dict, var: dict, mms_id: Union[str, int], data_path: Optional[str] = ""
):
    r"""Construct path of the data file using the standard convention.

    Parameters
    ----------
    file : dict
        File information.
    var : dict
        Variable information.
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
    file_date = parse(file["timetag"])

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
        f"mms{mms_id}",
        var["inst"],
        var["tmmode"],
        var["lev"],
        var["dtype"],
        *file_date.strftime("%Y-%m").split("-"),
    ]

    if var["tmmode"].lower() == "brst":
        path_list.append(file_date.strftime("%d"))

    return os.path.join(*path_list, file["file_name"])


def _download_file(
    session, url: str, headers: dict, out_file: str, size: Optional[int] = None
):
    r"""Download url to out_file, through a temporary file next to it so that a
    failed or interrupted download never leaves a partial file."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    part_file = f"{out_file}.part"

    try:
        with session.get(
            url, stream=True, verify=True, headers=headers, timeout=SDC_TIMEOUT
        ) as response:
            response.raise_for_status()

            with (
                open(part_file, "wb") as fs,
                tqdm.tqdm(total=size, unit="B", unit_scale=True, ncols=60) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fs.write(chunk)
                    progress.update(len(chunk))

        os.replace(part_file, out_file)
    finally:
        if os.path.exists(part_file):
            os.remove(part_file)


def download_data(
    var_str: str, tint: list, mms_id: Union[str, int], data_path: Optional[str] = ""
):
    r"""Download files from MMS SDC.

    Download data files containing field `var_str` over the time interval `tint` for
    the spacecraft `mms_id`. The files are saved to `data_path`.

    Parameters
    ----------
    var_str : str
        Input key of variable.
    tint : list
        Time interval.
    mms_id : str or int
        Index of the target spacecraft.
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu/mms/config.json`

    """
    var, _ = _var_and_cdf_name(var_str, mms_id)

    files_in_interval = list_files_sdc(tint, mms_id, var)

    sdc_session, headers, _ = _login_lasp()

    for file in files_in_interval:
        out_file = _make_path_local(file, var, mms_id, data_path)

        logging.info(
            "Downloading %s from %s...", os.path.basename(out_file), file["url"]
        )

        _download_file(sdc_session, file["url"], headers, out_file, file.get("size"))

    sdc_session.close()
