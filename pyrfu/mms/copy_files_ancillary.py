#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# Built-in imports
import os
import subprocess

# Local imports
from .list_files_ancillary import list_files_ancillary

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.22"
__status__ = "Prototype"


def copy_files_ancillary(product, tint, mms_id, tar_path: str = "./data/"):
    r"""Copy ancillary files from local_data_dir as defined in config.json to
    the target path.

    Parameters
    ----------
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    tint : list of str
        Time interval.
    mms_id : str or int
        Index of the spacecraft.
    tar_path : str, Optional
        Target path. Default is './data/'.

    """

    # Normalize the target path and make sure it exists.
    tar_path = os.path.normpath(tar_path)
    assert os.path.exists(tar_path), f"{tar_path} doesn't exist!!"

    # Get the local_data_dir path from config.json.
    pkg_path = os.path.dirname(os.path.abspath(__file__))

    # Read the current version of the MMS configuration file
    with open(os.path.join(pkg_path, "config.json"), "r", encoding="utf-8") as fs:
        config = json.load(fs)

    # Normalize the local_data_dir path and make sure it exists.
    mms_path = os.path.normpath(config["local_data_dir"]) + "/"
    assert os.path.exists(mms_path), f"{mms_path} doesn't exist!!"

    # List files that matches the requirements (instrument, date level,
    # data type, data rate) in the time interval for the target spacecraft.
    files = list_files_ancillary(tint, mms_id, product)

    for file in files:
        relative_path = os.path.split(file)[0].replace(mms_path, "")
        path = os.path.join(tar_path, relative_path)
        target_file = os.path.join(path, os.path.split(file)[1])

        if not os.path.exists(path):
            os.makedirs(path)

        with subprocess.Popen(
            f"cp {file} {target_file}", stdout=subprocess.PIPE, shell=True
        ) as s_proc:
            (_, _) = s_proc.communicate()

            # This makes the wait possible
            _ = s_proc.wait()
