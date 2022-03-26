#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

# Local imports
from ..pyrf import ts_append

from .list_files import list_files
from .get_ts import get_ts

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


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


def db_get_ts(dataset_name, cdf_name, tint, verbose: bool = True,
              data_path: str = ""):
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

    Returns
    -------
    out : xarray.DataArray
        Time series of the target variable.

    """

    probe, var = _tokenize(dataset_name)

    files = list_files(tint, probe, var, data_path=data_path)

    if verbose:
        logging.info(f"Loading {cdf_name}...")

    out = None
    for i, file in enumerate(files):
        out = ts_append(out, get_ts(file, cdf_name, tint))

    return out
