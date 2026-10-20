#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os
from typing import Optional

# 3rd party imports
import numpy as np
import pycdfpp
from xarray.core.dataarray import DataArray

from ..pyrf import read_cdf, time_clip, ts_append, ts_scalar
from ._list_files import _list_files

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def _list_files_lfr_density_l3(
    tint: list, data_path: Optional[str] = "", tree: Optional[bool] = False
) -> list:
    """Latest version of the L3_rpw-bia-density files covering the time interval."""
    return _list_files(
        tint, "L3_rpw-bia-density", ["L3", "lfr_density"], data_path, tree
    )


def read_lfr_density(
    tint: list, data_path: Optional[str] = "", tree: Optional[bool] = False
) -> DataArray:
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

    """

    # List LFR density files in the data path.
    files = _list_files_lfr_density_l3(tint, data_path, tree)

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
    if out is not None:
        out = out[np.argsort(out.time.data, kind="stable")]

    # Time clip
    if out is not None:
        out = time_clip(out, tint)

    return out
