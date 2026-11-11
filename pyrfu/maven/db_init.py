#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os

# Local imports
from .._user_config import set_local_data_dir, user_config_path

# Default configuration distributed with the package
_PACKAGE_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)


def config_path() -> str:
    r"""Path of the MAVEN configuration file, in the user configuration
    directory (created from the package configuration the first time)."""
    return user_config_path("maven_config.json", _PACKAGE_CFG_PATH)


def db_init(local_data_dir):
    r"""Setup the default path of MAVEN data.

    Parameters
    ----------
    local_data_dir : str
        Path to the data.

    Raises
    ------
    FileNotFoundError
        If local_data_dir doesn't exist (it was not written to the
        configuration, but with python -O it was).

    """

    # In the configuration of the user directory, at once (a write that failed
    # left a partial file)
    set_local_data_dir(config_path, local_data_dir)
