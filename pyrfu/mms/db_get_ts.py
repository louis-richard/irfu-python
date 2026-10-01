#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
from functools import lru_cache  # CHANGED: to cache the parsed config file
from typing import Mapping, Optional, Tuple

# 3rd party imports
from xarray.core.dataarray import DataArray

from pyrfu.mms.db_init import MMS_CFG_PATH
from pyrfu.mms.get_data import (
    _check_times,
    _get_file_content_sources,
    _list_files_sources,
)
from pyrfu.mms.get_ts import get_ts

# Local imports
from pyrfu.pyrf.ts_append import ts_append

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def _tokenize(dataset_name: str) -> Tuple[str, Mapping[str, str]]:
    r"""Tokenize dataset name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    str
        MMS spacecraft identifier.
    dict
        Dictionary containing the instrument, telemetry mode, level, and data type.

    """
    dataset = dataset_name.split("_")

    # Index of the MMS spacecraft
    probe = dataset[0][-1]

    var = {"inst": dataset[1], "tmmode": dataset[2], "lev": dataset[3]}

    # Data type (empty for datasets without one, e.g., mms1_fgm_srvy_l2)
    var["dtype"] = dataset[4] if len(dataset) > 4 else ""

    return probe, var


@lru_cache(maxsize=1)
def _load_config() -> Mapping[str, str]:
    r"""Load and cache the MMS configuration file."""
    with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
        return json.load(fs)


def _resolve_source(source: Optional[str] = "default") -> str:
    r"""Resource to fetch the data from: `source` ("local", "sdc" or "aws", in any
    case), or the default resource of `pyrfu/mms/config.json` if it is empty or
    "default"."""
    if not source or source.lower() == "default":
        return _load_config().get("default")

    if source.lower() in ["local", "sdc", "aws"]:
        return source.lower()

    raise ValueError("Invalid source. Must be one of 'default', 'local', 'sdc', 'aws'")


def _db_get_ts_dict(
    dataset_name: str,
    cdf_names: list[str],
    tint: list[str],
    verbose: Optional[bool] = True,
    data_path: Optional[str] = "",
    source: Optional[str] = "default",
) -> dict[str, DataArray]:
    r"""Time series of several variables of a dataset, reading each file once
    (i.e., one download per file from the SDC or AWS for all the variables).

    See :func:`db_get_ts` for the parameters; returns a dictionary of the time
    series with `cdf_names` as keys.

    """
    mms_id, var = _tokenize(dataset_name)
    resource = _resolve_source(source)

    file_names, sdc_session, headers = _list_files_sources(
        resource, tint, mms_id, var, data_path
    )

    out = {}

    try:
        if not file_names:
            raise FileNotFoundError(f"No files found for {dataset_name}")

        if verbose:
            for cdf_name in cdf_names:
                logging.info("Loading %s...", cdf_name)

        for file_name in file_names:
            file_content = _get_file_content_sources(
                resource, file_name, sdc_session, headers
            )

            for cdf_name in cdf_names:
                try:
                    ts = get_ts(file_content, cdf_name, tint)
                except Exception:
                    logging.error("Failed to load %s from %s", cdf_name, file_name)
                    raise

                out[cdf_name] = ts_append(out[cdf_name], ts) if cdf_name in out else ts
    finally:
        if sdc_session:
            sdc_session.close()

    return {cdf_name: _check_times(ts) for cdf_name, ts in out.items()}


def db_get_ts(
    dataset_name: str,
    cdf_name: str,
    tint: list[str],
    verbose: Optional[bool] = True,
    data_path: Optional[str] = "",
    source: Optional[str] = "default",
) -> DataArray:
    r"""Get variable time series in the cdf file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    cdf_name : str
        Name of the target field in cdf file.
    tint : list
        Time interval.
    verbose : bool, Optional
        Status monitoring. Default is verbose = True
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`
    source: str, Optional
        Resource to fetch data from: {"default", "local", "sdc", "aws"}. Default uses
        default in `pyrfu/mms/config.json`

    Returns
    -------
    DataArray
        Time series of the target variable.

    Raises
    ------
    FileNotFoundError
        If no files are found for the dataset name.
    ValueError
        If the source is not supported.

    """
    out = _db_get_ts_dict(dataset_name, [cdf_name], tint, verbose, data_path, source)

    return out[cdf_name]
