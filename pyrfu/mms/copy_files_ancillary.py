#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os
import shutil
from typing import Literal, Optional, Union

# Local imports
from pyrfu.mms.db_init import MMS_CFG_PATH
from pyrfu.mms.list_files_ancillary import list_files_ancillary

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def copy_files_ancillary(
    product: Literal["predatt", "predeph", "defatt", "defeph"],
    tint: list[str],
    mms_id: Union[int, str],
    tar_path: str,
    data_path: Optional[str] = "",
) -> None:
    r"""Copy ancillary files from local as defined in config.json to the target path.

    Parameters
    ----------
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    tint : list
        Time interval.
    mms_id : str or int
        Index of the spacecraft.
    tar_path : str
        Target path.
    data_path : str, Optional
        Local path to MMS data. Default uses that provided in pyrfu.mms.config.json

    """
    # Normalize the target path and make sure it exists.
    tar_path = os.path.normpath(tar_path)
    assert os.path.exists(tar_path), f"{tar_path} doesn't exist!!"

    if not data_path:
        # Read the current version of the MMS configuration file
        with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
            config = json.load(fs)

        root_path = os.path.normpath(config["local"])
    else:
        root_path = os.path.normpath(data_path)

    # Make sure the local path exists.
    assert os.path.exists(root_path), f"{root_path} doesn't exist!!"

    # List files that matches the requirements (instrument, date level,
    # data type, data rate) in the time interval for the target spacecraft.
    files = list_files_ancillary(tint, mms_id, product, data_path=root_path)

    for file in files:
        # Make paths
        relative_path = os.path.relpath(file, root_path)
        path = os.path.join(tar_path, os.path.dirname(relative_path))
        target_file = os.path.join(path, os.path.basename(file))

        # Create directories in target path
        os.makedirs(path, exist_ok=True)

        # Copy file
        shutil.copy2(file, target_file)
