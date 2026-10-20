#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os
import re
from typing import Optional

# 3rd party imports
import numpy as np

# Local imports
from .db_init import config_path

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.1"
__status__ = "Prototype"


def _list_files(
    tint: list,
    product: str,
    tree_dirs: list,
    data_path: Optional[str] = "",
    tree: Optional[bool] = False,
) -> list:
    r"""Find the latest version of the daily files of a Solar Orbiter product
    covering the time interval.

    Parameters
    ----------
    tint : list
        Time interval.
    product : str
        Level and descriptor of the product, e.g., "L2_rpw-tnr-surv", for the
        files solo_L2_rpw-tnr-surv[-cdag]_YYYYMMDD_Vnn.cdf.
    tree_dirs : list of str
        Directories of the product under data_path in a tree structured data
        repo, e.g., ["L2", "thr"], followed by year and month.
    data_path : str, Optional
        Path of Solar Orbiter data. Default uses the local_data_dir of the
        configuration in the user directory (see pyrfu.solo.db_init).
    tree : bool, Optional
        Flag for tree structured data repos. Default is False, i.e., files
        searched in data_path and its subdirectories.

    Returns
    -------
    list
        Files of the dates in the time interval, sorted by date: the highest
        version of each date, across the -cdag and public file names (the
        public one for the same version).

    Notes
    -----
    irfu-matlab (solo_local_file_db) treats the -cdag and public files as
    two datasets, chosen by the caller (read_TNR.m reads -cdag only); here
    they are one, as a data repo may hold either.

    """

    # Check input types
    assert isinstance(tint, (list, np.ndarray)), "tint must be array_like"
    assert len(tint) == 2, "tint must contain two elements"
    assert isinstance(tint[0], str), "tint[0] must be a string"
    assert isinstance(tint[1], str), "tint[1] must be a string"
    assert isinstance(tree, bool), "tree must be a boolean"

    # Check path
    if not data_path:
        # Read the configuration file (user directory)
        with open(config_path(), "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    # Make sure that the data path exists
    assert os.path.exists(data_path), f"{data_path} doesn't exist!!"

    # Only this product: not the -10-seconds density, nor partial downloads
    file_regex = re.compile(
        rf"^solo_{re.escape(product)}(-cdag)?_([0-9]{{8}})_V([0-9]+)\.cdf$"
    )

    # Dates of the files covering the time interval (end excluded)
    d_start = np.datetime64(tint[0], "D")
    d_end = (np.datetime64(tint[1], "ns") - np.timedelta64(1, "ns")).astype(
        "datetime64[D]"
    )

    if tree:
        months = np.arange(
            d_start.astype("datetime64[M]"),
            d_end.astype("datetime64[M]") + np.timedelta64(1, "M"),
        )
        local_dirs = [
            os.path.join(data_path, *tree_dirs, *str(month).split("-"))
            for month in months
        ]
    else:
        local_dirs = [data_path]

    # Latest version of each date: (version, public), so that the public file
    # wins over the -cdag one of the same version
    files = {}

    for local_dir in local_dirs:
        for root, _, file_names in os.walk(local_dir):
            for file_name in file_names:
                matches = file_regex.match(file_name)

                if not matches:
                    continue

                cdag, date, version = matches.groups()
                date = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}", "D")

                if not d_start <= date <= d_end:
                    continue

                key = (int(version), cdag is None)

                if date not in files or key > files[date][0]:
                    files[date] = (key, os.path.join(root, file_name))

    return [files[date][1] for date in sorted(files)]
