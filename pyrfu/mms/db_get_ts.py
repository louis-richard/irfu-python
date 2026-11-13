#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
from typing import Mapping, Optional, Tuple

# 3rd party imports
import numpy as np
from pycdfpp import load
from xarray import DataArray

from pyrfu.mms.get_data import (
    _file_label,
    _get_file_content_sources,
    _list_files_sources,
    _resolve_source,
)
from pyrfu.mms.get_ts import get_ts

# Local imports
from pyrfu.pyrf.ts_append import ts_append

logger = logging.getLogger(__name__)


def _split_dataset_name(dataset_name: str) -> Tuple[str, Mapping[str, str]]:
    r"""Split a dataset name (e.g. "mms1_fgm_srvy_l2") into the spacecraft
    and the file specification (the public `tokenize` reads variable keys).

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


def _db_get_ts_dict(
    dataset_name: str,
    cdf_names: list[str],
    tint: list[str],
    verbose: Optional[bool] = True,
    data_path: Optional[str] = "",
    source: Optional[str] = "default",
    support_names: Optional[list[str]] = None,
) -> dict[str, DataArray]:
    r"""Time series of several variables of a dataset, reading each file once
    (i.e., one download per file from the SDC or AWS for all the variables).

    See :func:`db_get_ts` for the parameters; returns a dictionary of the time
    series with `cdf_names` as keys. The non-record-varying variables of
    `support_names` (e.g. the FEEPS energy bounds) are returned as the array of
    their first record in the first file that has them, without time.

    """
    mms_id, var = _split_dataset_name(dataset_name)
    resource = _resolve_source(source)

    file_names, sdc_session, headers = _list_files_sources(
        resource, tint, mms_id, var, data_path
    )

    out = {}

    if not file_names:
        raise FileNotFoundError(f"No files found for {dataset_name} in {resource}")

    if verbose:
        for cdf_name in cdf_names:
            logger.info("Loading %s...", cdf_name)

    support = {}

    for file_name in file_names:
        file_content = _get_file_content_sources(
            resource, file_name, sdc_session, headers
        )

        missing = [name for name in support_names or [] if name not in support]

        if missing:
            cdf = load(file_content)
            for name in missing:
                if name in cdf:
                    support[name] = np.array(cdf[name].values[0])

        for cdf_name in cdf_names:
            try:
                ts = get_ts(file_content, cdf_name, tint)
            except Exception:
                logger.error(
                    "Failed to load %s from %s", cdf_name, _file_label(file_name)
                )
                raise

            out[cdf_name] = ts_append(out[cdf_name], ts) if cdf_name in out else ts

    return out | support


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
    tint : array_like or xarray.DataArray
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`.
    verbose : bool, Optional
        Status monitoring. Default is verbose = True
    data_path : str, Optional
        Path of MMS data. Default uses the local path of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to fetch the data from, in any case. Default uses the default
        resource of the MMS configuration.

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
