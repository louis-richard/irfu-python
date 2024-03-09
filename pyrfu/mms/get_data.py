#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os

# 3rd party imports
import requests
from botocore.exceptions import ClientError

# Local imports
from ..pyrf.dist_append import dist_append
from ..pyrf.ts_append import ts_append
from ..pyrf.ttns2datetime64 import ttns2datetime64
from .db_init import MMS_CFG_PATH
from .get_dist import get_dist
from .get_ts import get_ts
from .list_files import list_files
from .list_files_aws import list_files_aws
from .list_files_sdc import _login_lasp, list_files_sdc
from .tokenize import tokenize

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def _var_and_cdf_name(var_str, mms_id):
    var = tokenize(var_str)
    cdf_name = f"mms{mms_id}_{var['cdf_name']}"
    return var, cdf_name


def _check_times(inp):
    if inp.time.data.dtype == "int64":
        out = inp.assign_coords(time=ttns2datetime64(inp.time.data))
    else:
        out = inp
    return out


def _list_files_sources(source, tint, mms_id, var, data_path):
    if source == "local":
        file_names = list_files(tint, mms_id, var, data_path)
        sdc_session, headers = None, {}
    elif source == "sdc":
        file_names = [file.get("url") for file in list_files_sdc(tint, mms_id, var)]
        sdc_session, headers, _ = _login_lasp()
    elif source == "aws":
        file_names = [file.get("s3_obj") for file in list_files_aws(tint, mms_id, var)]
        sdc_session, headers = None, {}
    else:
        raise NotImplementedError("AWS is not yet implemented!!")

    return file_names, sdc_session, headers


def _get_file_content_sources(source, file_name, sdc_session, headers):
    if source == "local":
        file_path = os.path.normpath(file_name)
        with open(file_path, "rb") as file:
            file_content = file.read()
    elif source == "sdc":
        try:
            response = sdc_session.get(file_name, timeout=None, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            file_content = response.content
        except requests.RequestException as e:
            print(f"Error retrieving file from {file_name}: {e}")
    elif source == "aws":
        try:
            response = file_name.get()
            file_content = response["Body"].read()
        except ClientError as err:
            if err.response["Error"]["Code"] == "InternalError":  # Generic error
                logging.error("Error Message: %s", err.response["Error"]["Message"])

                response_meta = err.response.get("ResponseMetadata")
                logging.error("Request ID: %s", response_meta.get("RequestId"))
                logging.error("Http code: %s", response_meta.get("HTTPStatusCode"))
            else:
                raise err
    else:
        raise NotImplementedError(f"Resource {source} is not yet implemented!!")

    return file_content


def get_data(
    var_str,
    tint,
    mms_id,
    verbose: bool = True,
    data_path: str = "",
    source: str = "",
):
    r"""Load a variable. var_str must be in var (see below)

    Parameters
    ----------
    var_str : str
        Key of the target variable (use mms.get_data() to see keys.).
    tint : list of str
        Time interval.
    mms_id : str or int
        Index of the target spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Local path of MMS data. Default uses that provided in `pyrfu/mms/config.json`
    source: {"local", "sdc", "aws"}, Optional
        Ressource to fetch data from. Default uses default in `pyrfu/mms/config.json`

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the target variable of measured by the target
        spacecraft over the selected time interval.

    See also
    --------
    pyrfu.mms.get_ts : Read time series.
    pyrfu.mms.get_dist : Read velocity distribution function.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Index of MMS spacecraft

    >>> ic = 1

    Load magnetic field from FGM

    >>> b_xyz = mms.get_data("b_gse_fgm_brst_l2", tint_brst, ic)

    """

    mms_id = str(mms_id)

    var, cdf_name = _var_and_cdf_name(var_str, mms_id)

    # Read the current version of the MMS configuration file
    with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
        config = json.load(fs)

    source = source if source else config.get("default")

    file_names, sdc_session, headers = _list_files_sources(
        source, tint, mms_id, var, data_path
    )

    assert file_names, "No files found. Make sure that the data_path is correct"

    if verbose:
        logging.info("Loading %s...", cdf_name)

    out = None

    for file_name in file_names:
        file_content = _get_file_content_sources(
            source, file_name, sdc_session, headers
        )

        if "-dist" in var["dtype"]:
            out = dist_append(out, get_dist(file_content, cdf_name, tint))

        else:
            out = ts_append(out, get_ts(file_content, cdf_name, tint))

    out = _check_times(out)

    if sdc_session:
        sdc_session.close()

    return out
