#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bisect
import json
import logging

# Built-in imports
import os

# 3rd party imports
import pandas as pd

# Local imports
from pyrfu.mms.list_files_ancillary import list_files_ancillary
from pyrfu.pyrf.extend_tint import extend_tint
from pyrfu.pyrf.iso86012datetime import iso86012datetime

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


def load_ancillary(
    product,
    tint,
    mms_id,
    verbose: bool = True,
    data_path: str = "",
):
    r"""Loads ancillary data.

    Parameters
    ----------
    product : {"predatt", "predeph", "defatt", "defeph"}
        Ancillary type.
    tint : list of str
        Time interval
    mms_id : str or int
        Spacecraft index
    verbose : bool, Optional
        Set to True to follow the loading. Default is True
    data_path : str, Optional
        Path of MMS data. If None use `pyrfu.mms.mms_config.py`

    Returns
    -------
    out : xarray.Dataset
        Time series of the ancillary data

    """

    # Get path of files in interval
    tint_long = extend_tint(tint, [-86400, 86400])
    files_names = list_files_ancillary(tint_long, mms_id, product, data_path)

    # Convert time interval to datetime
    tint = iso86012datetime(tint)

    # Read length of header and columns names from .json file
    # Root path
    pkg_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.sep.join([pkg_path, "ancillary.json"]), "r", encoding="utf-8") as file:
        anc_dict = json.load(file)

    if verbose:
        logging.info("Loading ancillary %s files...", product)

    data_frame_dict = {}

    for i, file in enumerate(files_names):
        rows = pd.read_csv(
            file,
            # delim_whitespace=True,
            sep=r"\s+",
            header=None,
            skiprows=anc_dict[product]["header"],
        )

        # Remove footer
        rows = rows[:][:-1]

        # Convert time
        fmt = anc_dict[product]["time_format"]
        rows[0] = pd.to_datetime(rows[0], format=fmt)

        start_idx = bisect.bisect_left(rows[0][:], tint[0])
        end_idx = bisect.bisect_left(rows[0][:], tint[1])
        rows.columns = anc_dict[product]["columns_names"]

        data_frame_dict[i] = rows[:][start_idx:end_idx]

    data_frame = data_frame_dict[0]

    for k in list(data_frame_dict.keys())[1:]:
        data_frame = pd.concat([data_frame, data_frame_dict[k]], ignore_index=True)

    data_frame = data_frame.sort_values(by="time").set_index(["time"])
    dataset_out = data_frame.to_xarray()

    dataset_out = dataset_out.assign_coords(time=dataset_out.time.astype("<M8[ns]"))

    return dataset_out
