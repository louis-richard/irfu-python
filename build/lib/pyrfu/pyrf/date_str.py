#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from datetime import datetime

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def date_str(tint, fmt: int = 1):
    r"""Creates a string corresponding to time interval for output plot naming.

    Parameters
    ----------
    tint : list of str
        Time interval.
    fmt : int
        Format of the output :
            * 1 : "%Y%m%d_%H%M"
            * 2 : "%y%m%d%H%M%S"
            * 3 : "%Y%m%d_%H%M%S"_"%H%M%S"
            * 4 : "%Y%m%d_%H%M%S"_"%Y%m%d_%H%M%S"

    Returns
    -------
    out : str
        String corresponding to the time interval in the desired format.

    """

    start_time = datetime.strptime(tint[0], '%Y-%m-%dT%H:%M:%S.%f')
    end_time = datetime.strptime(tint[1], '%Y-%m-%dT%H:%M:%S.%f')

    if fmt == 1:
        out = start_time.strftime("%Y%m%d_%H%M")
    elif fmt == 2:
        out = start_time.strftime("%y%m%d%H%M%S")
    elif fmt == 3:
        out = "_".join([start_time.strftime("%Y%m%d_%H%M%S"),
                        end_time.strftime("%H%M%S")])
    elif fmt == 4:
        out = "_".join([start_time.strftime("%Y%m%d_%H%M%S"),
                        end_time.strftime("%Y%m%d_%H%M%S")])
    else:
        raise ValueError("Unknown format")

    return out
