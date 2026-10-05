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


def db_init(
    default: Optional[Literal["local", "sdc", "aws"]] = "local",
    local: Optional[str] = ".",
    sdc: Optional[str] = "public",
    sdc_username: Optional[str] = "username",
    sdc_password: Optional[str] = "password",
    aws: Optional[str] = "",
) -> None:
    r"""Manage the MMS data access configuration.

    The default resource to access MMS data, the local path to use and the Amazon
    Web Services (AWS) bucket name are saved in the MMS configuration file of the
    user configuration directory (`pyrfu.mms.MMS_CFG_PATH`), and the MMS SDC
    credentials in the system keyring (macOS Keychain, Windows Credential
    Locker, Secret Service on Linux) or, if there is none, in plain text in the
    keyring file in your home directory.

    Parameters
    ----------
    default : {"local", "sdc", "aws"}, Optional
        Name of the default resource to access the MMS data. Default is local.
    local : str, Optional
        Local path to MMS data. Default is the current directory.
    sdc : {"public", "sitl"}, Optional
        Rights to access MMS data from SDC. If "sitl" please make sure to register
        valid SDC credential. Default is public.
    sdc_username : str, Optional
        MMS SDC credential username. Default is "username".
    sdc_password : str, Optional
        MMS SDC credential password. Default is "password".
    aws : str, Optional
        Bucket name and prefix to MMS data in AWS S3, as "bucket/prefix". Default
        is empty, which uses the public MMS archive on NASA HelioCloud
        ("gov-nasa-hdrl-data1/spdf/cdaweb/data/mms", no AWS credentials needed).

    Raises
    ------
    NotImplementedError
        If the default resource is not implemented.
    FileNotFoundError
        If the local path doesn't exist.
    ValueError
        If the SDC rights are not "public" or "sitl".

    """
    # Check default
    if default.lower() not in ["local", "sdc", "aws"]:
        raise NotImplementedError(f"Resource {default} is not implemented!!")

    # Normalize the path and make sure that it exists
    local = os.path.normpath(os.path.abspath(local))

    if not os.path.exists(local):
        raise FileNotFoundError(f"{local} doesn't exists!!")

    # Check MMS SDC rights
    if sdc.lower() not in ["public", "sitl"]:
        raise ValueError("sdc must be 'public' or 'sitl'!!")

    config = {
        "default": default.lower(),
        "local": local,
        "sdc": {"rights": sdc.lower(), "username": sdc_username},
        "aws": aws,
    }

    logger.info("Updating MMS data access configuration in %s...", MMS_CFG_PATH)

    # Overwrite the configuration file with the new path
    with open(MMS_CFG_PATH, "w", encoding="utf-8") as fs:
        json.dump(config, fs)

    # Read credentials for sdc_username
    credential = _get_credential(sdc_username)

    if (
        not credential
        or credential.username == "username"
        or credential.password == "password"
    ):
        # if credentials are empty overwrite anyway
        username, password = sdc_username, sdc_password
    elif sdc_username == "username" or sdc_password == "password":
        # if existing credentials and incomplete arguments do not overwrite
        username, password = credential.username, credential.password
    else:
        # if existing credentials and complete arguments overwrite
        username, password = sdc_username, sdc_password

    _set_password(username, password)
