#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import shutil
from typing import Mapping, Optional, Union

# Local imports
from pyrfu.mms.db_init import _read_config
from pyrfu.mms.list_files import list_files


def copy_files(
    var: Mapping[str, str],
    tint: list[str],
    mms_id: Union[int, str],
    tar_path: str,
    data_path: Optional[str] = "",
) -> None:
    r"""Copy files from the local data path to the target path.

    Parameters
    ----------
    var : dict
        Dictionary containing 4 keys
            * var["inst"] : name of the instrument.
            * var["tmmode"] : data rate.
            * var["lev"] : data level.
            * var["dtype"] : data type.
    tint : array_like or xarray.DataArray
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`.
    mms_id : int or str
        Index of the spacecraft.
    tar_path : str
        Target path to put files.
    data_path : str, Optional
        Local path to MMS data. Default uses the local path of the MMS
        configuration (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

    """
    # Normalize the target path and make sure it exists.
    tar_path = os.path.normpath(tar_path)

    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"{tar_path} doesn't exist!!")

    normed_path = os.path.normpath(data_path or _read_config()["local"])

    # Make sure the local path exists.
    if not os.path.exists(normed_path):
        raise FileNotFoundError(f"{normed_path} doesn't exist!!")

    # List files that matches the requirements (instrument, date level,
    # data type, data rate) in the time interval for the target spacecraft.
    files = list_files(tint, mms_id, var, data_path=normed_path)

    for file in files:
        # Make paths
        relative_path = os.path.relpath(file, normed_path)
        directory_path = os.path.join(tar_path, os.path.dirname(relative_path))
        target_file = os.path.join(directory_path, os.path.basename(file))

        # Create directories in target path
        os.makedirs(directory_path, exist_ok=True)

        # Copy file
        shutil.copy(file, target_file)
