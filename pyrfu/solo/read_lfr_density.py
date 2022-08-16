#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import re
import json
import datetime

# 3rd party imports
from cdflib import cdfepoch
from dateutil import parser
from dateutil.rrule import rrule, DAILY

from ..pyrf import datetime2iso8601, read_cdf, ts_append, ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.22"
__status__ = "Prototype"


def _list_files_lfr_density_l3(tint, data_path: str = ""):
    """Find files in the L2 data repo corresponding to the target time interval.

    Parameters
    ----------
    tint : list
        Time interval
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.solo.config.json`

    Returns
    -------
    files : list
        List of files corresponding to the parameters in the selected time
        interval

    """

    # Check path
    if not data_path:
        # pkg_path = os.path.dirname(os.path.abspath(__file__))
        pkg_path = "/Users/louisr/Dropbox/Documents/PhD/irfu-python/pyrfu/solo"

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    # Make sure that the data path exists
    assert os.path.exists(data_path), f"{data_path} doesn't exist!!"

    files_out = []

    # directory and file name search patterns:
    # - assume directories are of the form: [data_path]/L3/lfr_density/year/month/
    # - assume file names are of the form: solo_L3_rpw-bia-density-cdag_YYYYMMDD_version.cdf

    file_name = r"solo_L3_rpw-bia-density-cdag_([0-9]{8})_V[0-9]{2}.cdf"

    d_start = parser.parse(parser.parse(tint[0]).strftime("%Y-%m-%d"))
    until_ = parser.parse(tint[1]) - datetime.timedelta(seconds=1)
    days = rrule(DAILY, dtstart=d_start, until=until_)

    for date in days:
        local_dir = os.sep.join(
            [data_path, "L3", "lfr_density", date.strftime("%Y"), date.strftime("%m")]
        )

        if os.name == "nt":
            full_path = os.sep.join([re.escape(local_dir) + os.sep, file_name])
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
        return sorted(files_out)
    else:
        return files_out


def read_lfr_density(tint):
    r"""Read L3 density data from LFR

    Parameters
    ----------
    tint : list
        Time interval
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.solo.config.json`

    Returns
    -------
    out : xarray.DataArray
        Time series of the density.

    """

    tint_ = list(map(parser.parse, tint))
    tint_ = list(map(datetime2iso8601, tint_))
    tint_ = list(map(cdfepoch.parse, tint_))

    files = _list_files_lfr_density_l3(tint)

    # Initialize spectrum output to None
    out = None

    for file in files:
        data_l3 = read_cdf(file, tint_)
        epoch = data_l3["epoch"].data
        time = cdfepoch.to_datetime(epoch, to_np=True)
        density = data_l3["density"].data
        out = ts_append(out, ts_scalar(time, density))

    return out
