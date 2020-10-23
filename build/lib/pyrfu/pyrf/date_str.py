#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date_str.py

@author : Louis RICHARD
"""

from dateutil import parser


def date_str(tint=None, fmt=1):
    """Creates a string corresponding to time interval for output plot naming.

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

    assert tint is not None and isinstance(tint, list) and len(tint) == 2

    t1 = parser.parse(tint[0])
    t2 = parser.parse(tint[1])

    if fmt == 1:
        out = t1.strftime("%Y%m%d_%H%M")
    elif fmt == 2:
        out = t1.strftime("%y%m%d%H%M%S")
    elif fmt == 3:
        out = "_".join([t1.strftime("%Y%m%d_%H%M%S"), t2.strftime("%H%M%S")])
    elif fmt == 4:
        out = "_".join([t1.strftime("%Y%m%d_%H%M%S"), t2.strftime("%Y%m%d_%H%M%S")])
    else:
        raise ValueError("Unknown format")

    return out
