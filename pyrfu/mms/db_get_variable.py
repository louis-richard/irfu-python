#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
from typing import Optional

# 3rd party imports
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.mms.get_variable import get_variable
from pyrfu.mms.list_files import list_files

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


def db_get_variable(
    dataset_name: str,
    cdf_name: str,
    tint: list[str],
    verbose: Optional[bool] = True,
    data_path: Optional[str] = "",
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

    Returns
    -------
    out : DataArray
       Variable of the target variable.

    Raises
    ------
    FileNotFoundError
        If no files are found for the dataset.

    """
    dataset = dataset_name.split("_")

    # Index of the MMS spacecraft
    probe = dataset[0][-1]

    var = {"inst": dataset[1], "tmmode": dataset[2], "lev": dataset[3]}

    try:
        var["dtype"] = dataset[4]
    except IndexError:
        pass

    files = list_files(tint, probe, var, data_path=data_path)

    if not files:
        raise FileNotFoundError(f"No files found for {cdf_name} in {data_path}")

    if verbose:
        logging.info("Loading %s...", cdf_name)

    out = get_variable(files[0], cdf_name)

    return out
