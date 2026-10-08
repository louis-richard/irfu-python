#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os

# Local imports
from .._user_config import user_config_path

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Default configuration distributed with the package
_PACKAGE_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)


def config_path() -> str:
    r"""Path of the SolO configuration file, in the user configuration
    directory (created from the package configuration the first time)."""
    return user_config_path("solo_config.json", _PACKAGE_CFG_PATH)


def db_init(local_data_dir):
    r"""Setup the default path of SolO data.

    Parameters
    ----------
    local_data_dir : str
        Path to the data.

    """

    # Normalize the path and make sure that it exists
    local_data_dir = os.path.normpath(local_data_dir)
    assert os.path.exists(
        local_data_dir,
    ), f"{local_data_dir} doesn't exists!!"

    # Read the current version of the configuration (user directory)
    with open(config_path(), "r", encoding="utf-8") as fs:
        config = json.load(fs)

    # Overwrite the configuration file with the new path
    with open(config_path(), "w", encoding="utf-8") as fs:
        config["local_data_dir"] = local_data_dir
        json.dump(config, fs)
