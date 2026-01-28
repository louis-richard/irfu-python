#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
from typing import Literal, Optional

# 3rd party imports
import keyring
from keyrings.alt.file import PlaintextKeyring

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

MMS_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def db_init(
    default: Optional[Literal["local", "sdc", "aws"]] = "local",
    local: Optional[str] = ".",
    sdc: Optional[str] = "public",
    sdc_username: Optional[str] = "username",
    sdc_password: Optional[str] = "password",
    aws: Optional[str] = "",
) -> None:
    r"""Manage the MMS data access configuration.

    The default resource to access MMS data, the local path to use, the MMS SDC
    credentials saved in encrypted file in your home directory, and the Amazon Web
    Services (AWS) bucket name.

    Parameters
    ----------
    default : {"local", "sdc", "aws"}, Optional
        Name of the default resource to access the MMS data. Default is local.
    local : str, Optional
        Local path to MMS data. Default is /Volumes/mms.
    sdc : {"public", "sitl"}, Optional
        Rights to access MMS data from SDC. If "sitl" please make sure to register
        valid SDC credential. Default is public.
    sdc_username : str, Optional
        MMS SDC credential username. Default is "username".
    sdc_password : str, Optional
        MMS SDC credential password. Default is "password".
    aws : str, Optional
        Bucket name and prefix to MMS data in AWS S3. Default is empty.

    Raises
    ------
    NotImplementedError
        If the default resource is not implemented.
    FileNotFoundError
        If the local path doesn't exist.
    ValueError
        If the SDC rights are not "public" or "sitl".

    """
    keyring.set_keyring(PlaintextKeyring())

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

    logging.info("Updating MMS data access configuration in %s...", MMS_CFG_PATH)

    # Overwrite the configuration file with the new path
    with open(MMS_CFG_PATH, "w", encoding="utf-8") as fs:
        json.dump(config, fs)

    # Read credentials for sdc_username
    credential_path = str(keyring.util.platform_.config_root())
    credential = keyring.get_credential("mms-sdc", sdc_username)

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

    logging.info("Updating MMS SDC credentials in %s...", credential_path)

    keyring.set_password("mms-sdc", username, password)
