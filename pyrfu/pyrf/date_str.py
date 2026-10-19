#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def date_str(tint: list[str], fmt: int = 1) -> str:
    r"""Creates a string corresponding to time interval for output plot naming.

    Parameters
    ----------
    tint : list of str
        Time interval, in ISO 8601 format at any precision (e.g.,
        "2019-09-14T07:54:00.000").
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

    # Check input
    assert isinstance(tint, list), "tint must be a list"
    assert isinstance(tint[0], str), "1st element of tint must be a string"
    assert isinstance(tint[1], str), "2nd element of tint must be a string"
    assert fmt in range(1, 5), "fmt must be 1, 2, 3, or 4"

    # Any ISO 8601 precision ("...T07:54:00", "...00.000", "...00.000000000")
    start_time, end_time = [
        np.datetime64(t_, "ns").astype("datetime64[us]").item() for t_ in tint[:2]
    ]

    if fmt == 1:
        out = start_time.strftime("%Y%m%d_%H%M")
    elif fmt == 2:
        out = start_time.strftime("%y%m%d%H%M%S")
    elif fmt == 3:
        out = "_".join(
            [
                start_time.strftime("%Y%m%d_%H%M%S"),
                end_time.strftime("%H%M%S"),
            ],
        )
    else:
        out = "_".join(
            [
                start_time.strftime("%Y%m%d_%H%M%S"),
                end_time.strftime("%Y%m%d_%H%M%S"),
            ],
        )

    return out
