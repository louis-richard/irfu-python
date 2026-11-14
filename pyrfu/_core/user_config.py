#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import contextlib
import json
import os
import shutil
import tempfile

# 3rd party imports
import platformdirs


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


def atomic_write(path: str, content) -> None:
    r"""Writes `content` (bytes or str) to `path` through a temporary file in
    the same directory, so that no reader sees a partial file."""
    mode = "wb" if isinstance(content, bytes) else "w"
    handle, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")

    try:
        with os.fdopen(handle, mode) as file:
            file.write(content)

        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def read_config(path: str) -> dict:
    r"""Content of the configuration file `path`."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_config(path: str, config: dict) -> None:
    r"""Writes the configuration `config` to `path` at once (a write that
    fails leaves the previous file)."""
    atomic_write(path, json.dumps(config))


def local_data_dir(path: str) -> str:
    r"""Local data directory of the configuration file `path` (MAVEN, SolO)."""
    return os.path.normpath(read_config(path)["local_data_dir"])


def set_local_data_dir(config_path, data_dir: str) -> None:
    r"""Sets the local data directory in the configuration file of
    `config_path()` (MAVEN, SolO), called once data_dir is checked, so that no
    configuration file is created for a wrong path.

    Raises
    ------
    FileNotFoundError
        If data_dir doesn't exist.

    """
    data_dir = os.path.normpath(data_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} doesn't exist")

    path = config_path()
    config = read_config(path)
    config["local_data_dir"] = data_dir
    write_config(path, config)
