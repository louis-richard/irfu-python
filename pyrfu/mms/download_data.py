#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os
import uuid
from typing import Optional, Union

# 3rd party imports
import tqdm
from dateutil.parser import parse

# Local imports
from pyrfu.mms.db_init import _read_config
from pyrfu.mms.get_data import _var_and_cdf_name
from pyrfu.mms.list_files_sdc import SDC_TIMEOUT, _login_lasp, list_files_sdc

logger = logging.getLogger(__name__)


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
        Path of MMS data. Default uses the local path of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

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

    data_path = os.path.normpath(data_path or _read_config()["local"])

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
    failed or interrupted download never leaves a partial file. The temporary
    file is the download's own: concurrent downloads of a file shared
    out_file.part, and the second one raised FileNotFoundError."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    part_file = f"{out_file}.{uuid.uuid4().hex}.part"

    try:
        with session.get(
            url, stream=True, verify=True, headers=headers, timeout=SDC_TIMEOUT
        ) as response:
            response.raise_for_status()

            with (
                open(part_file, "xb") as fs,
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
        Path of MMS data. Default uses the local path of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

    """
    var, _ = _var_and_cdf_name(var_str, mms_id)

    files_in_interval = list_files_sdc(tint, mms_id, var)

    sdc_session, headers, _ = _login_lasp()

    for file in files_in_interval:
        out_file = _make_path_local(file, var, mms_id, data_path)

        logger.info(
            "Downloading %s from %s...", os.path.basename(out_file), file["url"]
        )

        _download_file(sdc_session, file["url"], headers, out_file, file.get("size"))
