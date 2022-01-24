#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import json
import logging

# Local imports
from ..pyrf import dist_append, ts_append, ttns2datetime64

from .tokenize import tokenize
from .list_files import list_files
from .get_ts import get_ts
from .get_dist import get_dist

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


def _var_and_cdf_name(var_str, mms_id):
    var = tokenize(var_str)

    root_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.sep.join([root_path, "mms_keys.json"]), "r") as json_file:
        keys_ = json.load(json_file)

    var["dtype"] = keys_[var["inst"]][var_str.lower()]["dtype"]
    cdf_name = f"mms{mms_id}_{keys_[var['inst']][var_str.lower()]['cdf_name']}"

    return var, cdf_name


def _check_times(inp):
    if inp.time.data.dtype == "int64":
        out = inp.assign_coords(time=ttns2datetime64(inp.time.data))
    else:
        out = inp
    return out


def get_data(var_str, tint, mms_id, verbose: bool = True,
             data_path: str = ""):
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
        Path of MMS data. If None use `pyrfu/mms/config.json`

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

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint_brst, ic)

    """

    mms_id = str(mms_id)

    var, cdf_name = _var_and_cdf_name(var_str, mms_id)

    files = list_files(tint, mms_id, var, data_path)

    assert files, "No files found. Make sure that the data_path is correct"

    if verbose:
        logging.info(f"Loading {cdf_name}...")

    out = None

    for file in files:
        if "-dist" in var["dtype"]:
            out = dist_append(out, get_dist(file, cdf_name, tint))

        else:
            out = ts_append(out, get_ts(file, cdf_name, tint))

    out = _check_times(out)

    return out
