#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import subprocess

# Local imports
from .mms_config import CONFIG
from .list_files import list_files

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def copy_files(var, tint, mms_id, target_dir: str = "./data/"):
    r"""Copy files from NAS24 to the target path

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
    target_dir : str, Optional
        Target path. Default is './data/'.

    """

    mms_path = CONFIG["local_data_dir"] + "/"

    files = list_files(tint, mms_id, var)

    for file in files:
        relative_path = os.path.split(file)[0].replace(mms_path, "")
        path = os.path.join(target_dir, relative_path)
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
