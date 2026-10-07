#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
import shutil
from typing import Literal, Optional

# 3rd party imports
import keyring
import keyring.errors
import platformdirs
from keyring.backends.chainer import ChainerBackend
from keyrings.alt.file import PlaintextKeyring

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logger = logging.getLogger(__name__)

# Default configuration distributed with the package
_PACKAGE_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)


def _user_config_path() -> str:
    r"""Path of the MMS configuration file in the user configuration directory
    (e.g. ~/Library/Application Support/pyrfu on macOS, ~/.config/pyrfu on Linux).
    It is created from the package configuration file the first time, and the
    package file is used if the user directory cannot be written."""
    path = os.path.join(platformdirs.user_config_dir("pyrfu"), "mms_config.json")

    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copyfile(_PACKAGE_CFG_PATH, path)
        except OSError:
            return _PACKAGE_CFG_PATH

    return path


MMS_CFG_PATH = _user_config_path()

# Service name of the MMS SDC credentials in the keyring
SDC_SERVICE = "mms-sdc"


def _secure_keyring():
    r"""System keyring (e.g. macOS Keychain, Windows Credential Locker, Secret
    Service) if one is available, None otherwise."""
    backend = keyring.get_keyring()
    backends = backend.backends if isinstance(backend, ChainerBackend) else [backend]

    if any(b.priority >= 1 for b in backends):
        return backend

    return None


def _get_credential(username: str):
    r"""MMS SDC credential of username, from the system keyring or, for the
    credentials saved by previous versions or without system keyring, from the
    plaintext keyring file."""
    secure = _secure_keyring()
    credential = secure.get_credential(SDC_SERVICE, username) if secure else None

    if credential is None:
        credential = PlaintextKeyring().get_credential(SDC_SERVICE, username)

    return credential


def _set_password(username: str, password: str) -> None:
    r"""Save the MMS SDC credential in the system keyring, or in the plaintext
    keyring file (with a warning) if there is no system keyring."""
    secure = _secure_keyring()
    plaintext = PlaintextKeyring()

    if secure is not None:
        logger.info("Updating MMS SDC credentials in the system keyring...")
        secure.set_password(SDC_SERVICE, username, password)

        # Remove the copy saved in plain text by previous versions
        try:
            plaintext.delete_password(SDC_SERVICE, username)
        except keyring.errors.PasswordDeleteError:
            pass
    else:
        logger.warning(
            "No system keyring available: MMS SDC credentials saved in plain text "
            "in %s",
            plaintext.file_path,
        )
        plaintext.set_password(SDC_SERVICE, username, password)


# Settings used when there is no configuration yet, or with reset=True
_DEFAULT_CONFIG = {
    "default": "local",
    "local": ".",
    "sdc": {"rights": "public", "username": "username"},
    "aws": "",
}


def _read_config() -> dict:
    r"""Current MMS configuration, completed with the default settings."""
    config = json.loads(json.dumps(_DEFAULT_CONFIG))

    try:
        with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
            current = json.load(fs)
    except (OSError, ValueError):
        return config

    for key in ["default", "local", "aws"]:
        if key in current:
            config[key] = current[key]

    if isinstance(current.get("sdc"), dict):
        config["sdc"].update(
            {k: v for k, v in current["sdc"].items() if k in ["rights", "username"]}
        )

    return config


def db_init(
    default: Optional[Literal["local", "sdc", "aws"]] = None,
    local: Optional[str] = None,
    sdc: Optional[Literal["public", "sitl"]] = None,
    sdc_username: Optional[str] = None,
    sdc_password: Optional[str] = None,
    aws: Optional[str] = None,
    reset: bool = False,
) -> None:
    r"""Manage the MMS data access configuration.

    The default resource to access MMS data, the local path to use, the MMS SDC
    rights and username, and the Amazon Web Services (AWS) bucket name are
    saved in the MMS configuration file of the user configuration directory
    (`pyrfu.mms.MMS_CFG_PATH`). The MMS SDC password is saved in the system
    keyring (macOS Keychain, Windows Credential Locker, Secret Service on
    Linux) or, if there is none, in plain text in the keyring file in your
    home directory. It is never written in the configuration file.

    Only the settings that are given are changed; the others keep their
    current value (or the default one if there is no configuration yet, or
    with `reset=True`).

    Parameters
    ----------
    default : {"local", "sdc", "aws"}, Optional
        Name of the default resource to access the MMS data. Default is local.
    local : str, Optional
        Local path to MMS data. Default is the current directory.
    sdc : {"public", "sitl"}, Optional
        Rights to access MMS data from SDC. Default is public. "sitl" needs the
        credentials of `sdc_username` in the keyring.
    sdc_username : str, Optional
        MMS SDC username. Given alone, the configuration uses this user, whose
        password must already be in the keyring.
    sdc_password : str, Optional
        MMS SDC password of `sdc_username`, saved in the keyring. It needs
        `sdc_username`.
    aws : str, Optional
        Bucket name and prefix to MMS data in AWS S3, as "bucket/prefix". Empty
        (the default) uses the public MMS archive on NASA HelioCloud
        ("gov-nasa-hdrl-data1/spdf/cdaweb/data/mms", no AWS credentials needed).
    reset : bool, Optional
        Start from the default settings instead of the current ones. The
        keyring is not changed. Default is False.

    Raises
    ------
    NotImplementedError
        If the default resource is not implemented.
    FileNotFoundError
        If the local path doesn't exist.
    ValueError
        If the SDC rights are not "public" or "sitl", or if `sdc_password` is
        given without `sdc_username`.

    Examples
    --------
    >>> from pyrfu import mms

    Use the local data in /data/mms by default (the other settings are kept)

    >>> mms.db_init(default="local", local="/data/mms")

    Save the MMS SDC team credentials and use them

    >>> mms.db_init(sdc="sitl", sdc_username="user", sdc_password="password")

    """
    config = json.loads(json.dumps(_DEFAULT_CONFIG)) if reset else _read_config()

    # Check and set the given settings only
    if default is not None:
        if default.lower() not in ["local", "sdc", "aws"]:
            raise NotImplementedError(f"Resource {default} is not implemented!!")

        config["default"] = default.lower()

    if local is not None or reset:
        # Normalize the path and make sure that it exists
        local = os.path.normpath(
            os.path.abspath(config["local"] if local is None else local)
        )

        if not os.path.exists(local):
            raise FileNotFoundError(f"{local} doesn't exists!!")

        config["local"] = local

    if sdc is not None:
        if sdc.lower() not in ["public", "sitl"]:
            raise ValueError("sdc must be 'public' or 'sitl'!!")

        config["sdc"]["rights"] = sdc.lower()

    if sdc_password is not None and sdc_username is None:
        raise ValueError("sdc_password needs sdc_username")

    if sdc_username is not None:
        config["sdc"]["username"] = sdc_username

    if aws is not None:
        config["aws"] = aws

    # First configuration: the current directory, as an absolute path
    if not os.path.isabs(config["local"]):
        config["local"] = os.path.normpath(os.path.abspath(config["local"]))

    # Save the password in the keyring only if given (never a placeholder)
    if sdc_password is not None:
        _set_password(sdc_username, sdc_password)

    if config["sdc"]["rights"] == "sitl" and not _get_credential(
        config["sdc"]["username"]
    ):
        logger.warning(
            "No MMS SDC credentials for %s in the keyring: use "
            "mms.db_init(sdc_username=..., sdc_password=...)",
            config["sdc"]["username"],
        )

    logger.info("Updating MMS data access configuration in %s...", MMS_CFG_PATH)

    with open(MMS_CFG_PATH, "w", encoding="utf-8") as fs:
        json.dump(config, fs)
