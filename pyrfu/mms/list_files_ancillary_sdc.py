#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from datetime import datetime, timedelta

# 3rd party imports
import numpy as np

# Local imports
from .list_files_sdc import _login_lasp

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.11"
__status__ = "Prototype"


def _construct_url_json_list(tint, mms_id, product, lasp_url):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    tint = np.array(tint).astype("<M8[ns]").astype(str)
    tint = [datetime.strptime(t_[:-3], "%Y-%m-%dT%H:%M:%S.%f") for t_ in tint]
    start_date = (tint[0] - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (tint[1] + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"{lasp_url}/file_info/ancillary"
    url = f"{url}?start_date={start_date}&end_date={end_date}&sc_id=mms{mms_id}"

    url = f"{url}&product={product}"

    return url


def _make_urls_ancillaries(lasp_url, files):
    for file in files:
        file["url"] = f"{lasp_url}download/ancillary?file={file['file_name']}"

    return files


def list_files_ancillary_sdc(tint, mms_id, product):
    r"""Find available ancillary files from LASP SDC for the target product type.

    Parameters
    ----------
    tint : list of str
        Time interval
    mms_id : str or int
        Spacecraft index
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.

    Returns
    -------
    file_names : list
        Ancillary files in interval.

    """

    sdc_session, headers, lasp_url = _login_lasp()

    url_json_ancillaries = _construct_url_json_list(tint, mms_id, product, lasp_url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        response = sdc_session.get(url_json_ancillaries, verify=True, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        http_json = response.json()

    file_names = http_json["files"]
    sdc_session.close()

    file_names = _make_urls_ancillaries(lasp_url, file_names)

    return file_names
