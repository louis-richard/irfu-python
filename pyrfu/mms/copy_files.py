#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import json
import subprocess

# Local imports
from .list_files import list_files

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.11"
__status__ = "Prototype"


def copy_files(var, tint, mms_id, tar_path: str = "./data/"):
    r"""Copy files from local_data_dir as defined in config.json to the target 
    path.

    Parameters
    ----------
    var : dict
        Dictionary containing 4 keys
            * var["inst"] : name of the instrument.
            * var["tmmode"] : data rate.
            * var["lev"] : data level.
            * var["dtype"] : data type.
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
    with open(os.path.join(pkg_path, "config.json"), "r") as f:
        config = json.load(f)

    # Normalize the local_data_dir path and make sure it exists.
    mms_path = os.path.normpath(config["local_data_dir"]) + "/"
    assert os.path.exists(mms_path), f"{mms_path} doesn't exist!!"

    # List files that matches the requirements (instrument, date level,
    # data type, data rate) in the time interval for the target spacecraft.
    files = list_files(tint, mms_id, var)

    for file in files:
        relative_path = os.path.split(file)[0].replace(mms_path, "")
        path = os.path.join(tar_path, relative_path)
        target_file = os.path.join(path, os.path.split(file)[1])

        if not os.path.exists(path):
            os.makedirs(path)

        p = subprocess.Popen(f"cp {file} {target_file}",
                             stdout=subprocess.PIPE,
                             shell=True)
        (_, _) = p.communicate()

        # This makes the wait possible
        _ = p.wait()

    return
