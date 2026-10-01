#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
from typing import Optional

# 3rd party imports
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.mms.db_get_ts import _resolve_source, _tokenize
from pyrfu.mms.get_data import _get_file_content_sources, _list_files_sources
from pyrfu.mms.get_variable import get_variable

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def db_get_variable(
    dataset_name: str,
    cdf_name: str,
    tint: list[str],
    verbose: Optional[bool] = True,
    data_path: Optional[str] = "",
    source: Optional[str] = "default",
) -> DataArray:
    r"""Get variable in the cdf file.

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
    out : DataArray
       Variable of the target variable (from the first file of the time
       interval).

    Raises
    ------
    FileNotFoundError
        If no files are found for the dataset.
    ValueError
        If the source is not supported.

    """
    mms_id, var = _tokenize(dataset_name)
    resource = _resolve_source(source)

    file_names, sdc_session, headers = _list_files_sources(
        resource, tint, mms_id, var, data_path
    )

    try:
        if not file_names:
            raise FileNotFoundError(f"No files found for {cdf_name} in {resource}")

        if verbose:
            logging.info("Loading %s...", cdf_name)

        file_content = _get_file_content_sources(
            resource, file_names[0], sdc_session, headers
        )
    finally:
        if sdc_session:
            sdc_session.close()

    out = get_variable(file_content, cdf_name)

    return out
