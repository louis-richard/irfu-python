#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import json

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def db_init(local_data_dir):
    r"""Setup the default path of MMS data.

    Parameters
    ----------
    local_data_dir : str
        Path to the data.

    """

    # Path to the configuration file.
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Read the current version of the configuration
    with open(os.path.join(root_path, "config.json"), "r") as f:
        config = json.load(f)

    # Overwrite the configuration file with the new path
    with open(os.path.join(root_path, "config.json"), "w") as f:
        config["local_data_dir"] = os.path.normpath(local_data_dir)
        json.dump(config, f)

    return
