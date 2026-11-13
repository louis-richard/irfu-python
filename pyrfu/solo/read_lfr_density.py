#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os

# 3rd party imports
import numpy as np
import pycdfpp
from xarray import DataArray

from ..pyrf import read_cdf, time_clip, ts_append, ts_scalar
from ._list_files import _list_files

logger = logging.getLogger(__name__)


def _list_files_lfr_density_l3(
    tint: list, data_path: str = "", tree: bool = False
) -> list:
    """Latest version of the L3_rpw-bia-density files covering the time interval."""
    return _list_files(
        tint, "L3_rpw-bia-density", ["L3", "lfr_density"], data_path, tree
    )


def read_lfr_density(tint: list, data_path: str = "", tree: bool = False) -> DataArray:
    r"""Read L3 density data from LFR

    Parameters
    ----------
    tint : list
        Time interval
    data_path : str, Optional
        Path of Solar Orbiter data. Default uses the local_data_dir of the
        configuration in the user directory (see pyrfu.solo.db_init).
    tree : bool, Optional
        Flag for tree structured data repos. Default is False.

    Returns
    -------
    DataArray
        Time series of the density, from the latest version of each daily
        file, sorted in time.

    Raises
    ------
    TypeError
        If tint is not two strings, or tree not a bool.
    ValueError
        If tint has not two elements.
    FileNotFoundError
        If the data path doesn't exist, or no file covers the time interval
        (None was returned).

    """

    # List LFR density files in the data path.
    files = _list_files_lfr_density_l3(tint, data_path, tree)

    if not files:
        raise FileNotFoundError(f"no L3 LFR density file in {tint}")

    # Initialize spectrum output to None
    out = None

    for file in files:
        # Notify user
        logger.info("Loading %s...", os.path.split(file)[-1])

        # Read file content
        data_l3 = read_cdf(file)

        # Get time from Epoch
        epoch = data_l3["epoch"].data

        # Convert epoch to datetime64
        time = pycdfpp.to_datetime64(epoch)

        # Get density data and contruct time series.
        density = data_l3["density"].data
        density[density == -1e31] = np.nan
        out = ts_append(out, ts_scalar(time, density))

    # Sort in time
    out = out[np.argsort(out.time.data, kind="stable")]

    # Time clip
    out = time_clip(out, tint)

    return out
