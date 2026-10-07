#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import shutil

# 3rd party imports
import platformdirs

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2026"
__license__ = "MIT"
__version__ = "2.4.21"
__status__ = "Prototype"


def user_config_path(file_name: str, package_cfg_path: str) -> str:
    r"""Path of a configuration file in the user configuration directory (e.g.
    ~/Library/Application Support/pyrfu on macOS, ~/.config/pyrfu on Linux),
    created from the configuration file of the package the first time. The
    package file is used if the user directory cannot be written.

    Parameters
    ----------
    file_name : str
        Name of the configuration file in the user directory.
    package_cfg_path : str
        Path of the default configuration file distributed with the package.

    Returns
    -------
    path : str
        Path of the configuration file to read and write.

    """
    path = os.path.join(platformdirs.user_config_dir("pyrfu"), file_name)

    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copyfile(package_cfg_path, path)
        except OSError:
            return package_cfg_path

    return path
