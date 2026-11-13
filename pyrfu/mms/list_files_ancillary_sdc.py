#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from datetime import timedelta

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.iso86012datetime import iso86012datetime
from ..pyrf.time_clip import _time_interval
from .list_files_ancillary import _sc_name
from .list_files_sdc import SDC_TIMEOUT, _login_lasp


def _construct_url_json_list(tint, mms_id, product, lasp_url):
    r"""Construct the url that return a json-formatted string of science
    filenames that are available for download according to:
    https://lasp.colorado.edu/mms/sdc/team/about/how-to/
    """

    # In the forms of time_clip (a time series failed)
    tint = iso86012datetime(np.array(_time_interval(tint)))
    start_date = (tint[0] - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (tint[1] + timedelta(days=1)).strftime("%Y-%m-%d")

    # lasp_url ends with "/" (it was doubled)
    url = f"{lasp_url}file_info/ancillary"
    url = f"{url}?start_date={start_date}&end_date={end_date}"
    url = f"{url}&sc_id={_sc_name(product, mms_id)}"

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
    tint : array_like or xarray.DataArray
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`.
    mms_id : str or int
        Spacecraft index (not used for "defq" and "predq", which cover the
        four spacecraft).
    product : {"predatt", "predeph", "predq", "defatt", "defeph", "defq"}
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
        response = sdc_session.get(
            url_json_ancillaries, verify=True, headers=headers, timeout=SDC_TIMEOUT
        )
        response.raise_for_status()  # Raise an HTTPError for bad responses
        http_json = response.json()

    file_names = http_json["files"]

    file_names = _make_urls_ancillaries(lasp_url, file_names)

    return file_names
