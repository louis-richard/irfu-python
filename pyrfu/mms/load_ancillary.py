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
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


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
    product : {"defatt", "defeph", "defq"}
        Ancillary type. The predicted products (predatt, predeph, predq) are
        not supported.
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
        Time series of the ancillary data, with strictly increasing times

    Raises
    ------
    ValueError
        If the product is not supported.
    FileNotFoundError
        If no file of the product covers the time interval.

    Notes
    -----
    DEFATT files repeat some times by design, and consecutive files overlap
    (by about 5 minutes, with phases different by up to 0.007 deg). As
    irfu-matlab, the last of the repeated rows of a file is kept
    (mms_load_ancillary), and in an overlap the rows of the earlier file
    (mms_db).

    """

    # Read length of header and columns names from .json file
    # Root path
    pkg_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.sep.join([pkg_path, "ancillary.json"]), "r", encoding="utf-8") as file:
        anc_dict = json.load(file)

    if product not in anc_dict:
        raise ValueError(
            f"Ancillary product {product!r} is not supported, use one of "
            f"{', '.join(anc_dict)}"
        )

    # Get path of files in interval
    tint_long = extend_tint(tint, [-86400, 86400])
    files_names = list_files_ancillary(tint_long, mms_id, product, data_path)

    if not files_names:
        raise FileNotFoundError(f"No {product} file found for MMS{mms_id} in {tint}")

    # Convert time interval to datetime
    tint = iso86012datetime(tint)

    if verbose:
        logger.info("Loading ancillary %s files...", product)

    data_frame_dict = {}

    for i, file in enumerate(files_names):
        rows = pd.read_csv(
            file,
            # delim_whitespace=True,
            sep=r"\s+",
            header=None,
            skiprows=anc_dict[product]["header"],
        )

        # Convert time, and remove the rows that are not data, such as the
        # DATA_STOP footer of DEFATT files (DEFEPH files have no footer)
        fmt = anc_dict[product]["time_format"]
        rows[0] = pd.to_datetime(rows[0], format=fmt, errors="coerce")
        rows = rows[rows[0].notna()].reset_index(drop=True)

        start_idx = bisect.bisect_left(rows[0][:], tint[0])
        end_idx = bisect.bisect_left(rows[0][:], tint[1])
        rows.columns = anc_dict[product]["columns_names"]

        # The last of the rows repeating a time, as mms_load_ancillary
        rows = rows[start_idx:end_idx].drop_duplicates(subset="time", keep="last")
        data_frame_dict[i] = rows

    data_frame = pd.concat(list(data_frame_dict.values()), ignore_index=True)

    # Consecutive files overlap: the rows of the earlier file, as mms_db (the
    # files are in time order, and the sort is stable)
    data_frame = data_frame.sort_values(by="time", kind="stable")
    data_frame = data_frame.drop_duplicates(subset="time", keep="first")
    data_frame = data_frame.set_index(["time"])
    dataset_out = data_frame.to_xarray()

    dataset_out = dataset_out.assign_coords(time=dataset_out.time.astype("<M8[ns]"))

    return dataset_out
