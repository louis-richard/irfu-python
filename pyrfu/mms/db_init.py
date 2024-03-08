#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os

import keyring

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.5.0"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

MMS_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def db_init(
    default: str = "local",
    local: str = "../data",
    sdc: str = "public",
    sdc_username: str = "",
    sdc_password: str = "",
    aws: str = "",
):
    r"""Setup the default resource to access MMS data. MMS SDC username and password
    are stored in secured credentials in encrypted file in your home directory.

    Parameters
    ----------
    default : {"local", "sdc", "aws"}, Optional
        Name of the default resource to access the MMS data. Default is local.
    local : str
        Local path to MMS data. Default is /Volumes/mms.
    sdc : {"public", "sitl"}, Optional
        Rights to access MMS data from SDC. If "sitl" please make sure to register
        valid SDC credential. Default is public.
    sdc_username : str, Optional
        MMS SDC credential username. Default is empty.
    sdc_password : str, Optional
        MMS SDC credential password. Default is empty.
    aws : str, Optional
        Bucket name and prefix to MMS data in AWS S3. Default is empty.

    """

    # Check default
    if default.lower() not in ["local", "sdc", "aws"]:
        raise NotImplementedError(f"Resource {default} is not implemented!!")

    # Normalize the path and make sure that it exists
    local = os.path.normpath(os.path.abspath(local))
    assert os.path.exists(local), f"{local} doesn't exists!!"

    # Check MMS SDC rights
    assert sdc.lower() in ["public", "sitl"], "sdc must be 'public' or 'sitl'!!"

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

    if not (credential and credential.username and credential.password):
        # if credentials are empty overwrite anyway
        username, password = sdc_username, sdc_password
    elif not sdc_username or not sdc_password:
        # if existing credentials and incomplete arguments do not overwrite
        username, password = credential.username, credential.password
    else:
        # if existing credentials and complete arguments overwrite
        username, password = sdc_username, sdc_password

    logging.info("Updating MMS SDC credentials in %s...", credential_path)

    keyring.set_password("mms-sdc", username, password)
