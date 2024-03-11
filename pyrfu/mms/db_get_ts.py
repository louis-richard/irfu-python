#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging

# Local imports
from ..pyrf.ts_append import ts_append
from .db_init import MMS_CFG_PATH
from .get_data import _check_times, _get_file_content_sources, _list_files_sources
from .get_ts import get_ts

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


def _tokenize(dataset_name):
    dataset = dataset_name.split("_")

    # Index of the MMS spacecraft
    probe = dataset[0][-1]

    var = {"inst": dataset[1], "tmmode": dataset[2], "lev": dataset[3]}

    try:
        var["dtype"] = dataset[4]
    except IndexError:
        pass

    return probe, var


def db_get_ts(
    dataset_name,
    cdf_name,
    tint,
    verbose: bool = True,
    data_path: str = "",
    source: str = "",
):
    r"""Get variable time series in the cdf file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    cdf_name : str
        Name of the target field in cdf file.
    tint : array_like
        Time interval.
    verbose : bool, Optional
        Status monitoring. Default is verbose = True
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`
    source: {"local", "sdc", "aws"}, Optional
        Ressource to fetch data from. Default uses default in `pyrfu/mms/config.json`

    Returns
    -------
    out : xarray.DataArray
        Time series of the target variable.

    """

    mms_id, var = _tokenize(dataset_name)

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

        out = ts_append(out, get_ts(file_content, cdf_name, tint))

    out = _check_times(out)

    if sdc_session:
        sdc_session.close()

    return out
