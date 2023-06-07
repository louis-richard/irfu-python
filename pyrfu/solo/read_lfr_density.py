#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import json
import logging
import os
import re

# 3rd party imports
import numpy as np

from cdflib import cdfepoch
from dateutil import parser
from dateutil.rrule import rrule, DAILY

from ..pyrf import read_cdf, ts_append, ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "MIT"
__version__ = "2.3.22"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def _list_files_lfr_density_l3(tint, data_path: str = "", tree: bool = False):
    """Find files in the L2 data repo corresponding to the target time
    interval.

    Parameters
    ----------
    tint : list
        Time interval
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.solo.config.json`
    tree : bool, Optional
        Flag for tree structured data repos. Default is False.

    Returns
    -------
    files : list
        List of files corresponding to the parameters in the selected time
        interval

    """

    # Check path
    if not data_path:
        # pkg_path = os.path.dirname(os.path.abspath(__file__))
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    # Make sure that the data path exists
    assert os.path.exists(data_path), f"{data_path} doesn't exist!!"

    files_out = []

    # directory and file name search patterns:
    # - assume directories are of the form: [path]/L3/lfr_density/year/month/
    # - assume file names are of the form:
    #   solo_L3_rpw-bia-density-cdag_YYYYMMDD_version.cdf

    file_name = r"solo_L3_rpw-bia-density.*_([0-9]{8})_V[0-9]{2}.cdf"

    d_start = parser.parse(parser.parse(tint[0]).strftime("%Y-%m-%d"))
    until_ = parser.parse(tint[1]) - datetime.timedelta(seconds=1)
    days = rrule(DAILY, dtstart=d_start, until=until_)

    for date in days:
        if tree:
            local_dir = os.sep.join(
                [
                    data_path,
                    "L3",
                    "lfr_density",
                    date.strftime("%Y"),
                    date.strftime("%m"),
                ],
            )
        else:
            local_dir = data_path

        if os.name == "nt":
            full_path = os.sep.join(
                [re.escape(local_dir) + os.sep, file_name],
            )
        else:
            full_path = os.sep.join([re.escape(local_dir), file_name])

        regex = re.compile(full_path)

        for root, _, files in os.walk(local_dir):
            for file in files:
                this_file = os.sep.join([root, file])

                matches = regex.match(this_file)
                if matches:
                    this_time = parser.parse(matches.groups()[0])
                    if d_start <= this_time <= until_:
                        if this_file not in files_out:
                            files_out.append(os.sep.join([local_dir, file]))

    # sort in time
    if len(files_out) > 1:
        files_out = sorted(files_out)

    return files_out


def read_lfr_density(tint, data_path: str = "", tree: bool = False):
    r"""Read L3 density data from LFR

    Parameters
    ----------
    tint : list
        Time interval
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.solo.config.json`
    tree : bool, Optional
        Flag for tree structured data repos. Default is False.

    Returns
    -------
    out : xarray.DataArray
        Time series of the density.

    """

    # List LFR density files in the data path.
    files = _list_files_lfr_density_l3(tint, data_path, tree)

    # Initialize spectrum output to None
    out = None

    for file in files:
        # Notify user
        logging.info("Loading %s...", os.path.split(file)[-1])

        # Read file content
        data_l3 = read_cdf(file, tint)

        # Get time from Epoch
        epoch = data_l3["epoch"].data
        time = cdfepoch.to_datetime(epoch)

        # Get density data and contruct time series.
        density = data_l3["density"].data
        density[density == -1e31] = np.nan
        out = ts_append(out, ts_scalar(time, density))

    return out
