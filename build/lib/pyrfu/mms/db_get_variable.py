#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# Local imports
from .list_files import list_files
from .get_variable import get_variable

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def db_get_variable(dataset_name, cdf_name, tint, verbose: bool = True,
                    data_path: str = ""):
    r"""Get variable in the cdf file.

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

    Returns
    -------
    out : xarray.DataArray
       Variable of the target variable.

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

    if verbose:
        logging.info(f"Loading {cdf_name}...")

    out = get_variable(files[0], cdf_name)

    return out
