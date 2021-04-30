#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""extend_tint.py
@author: Louis Richard
"""

import numpy as np

from .iso86012datetime64 import iso86012datetime64
from .datetime642iso8601 import datetime642iso8601


def extend_tint(tint, ext=None):
    """Extends time interval.

    Parameters
    ----------
    tint : list of str
        Reference time interval to extend.

    ext : list of float or list of int
        Number of seconds to extend time interval
        [left extend, right extend].

    Returns
    -------
    tint_new : list of str
        Extended time interval.

    Examples
    --------
    >>> from pyrfu import pyrf

    Time interval

    >>> tints = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]

    Load spacecraft position

    >>> tints_long = pyrf.extend_tint(tint, [-100, 100])

    """

    if ext is None:
        ext = [-60, 60]

    # Convert extension to timedelta64 in s units
    ext = np.array(ext).astype("timedelta64[s]")

    # Original time interval to datetime64 format in ns units
    tint_ori = iso86012datetime64(np.array(tint))

    # New time interval in iso 8601 format
    tint_new = list(datetime642iso8601(tint_ori + ext))

    return tint_new
