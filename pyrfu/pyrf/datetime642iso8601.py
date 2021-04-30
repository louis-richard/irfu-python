#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 - 2021 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""datetime642iso8601.py
@author: Louis Richard
"""

import numpy as np


def datetime642iso8601(time):
    r"""Convert datetime64 in ns units to ISO 8601 time format .

    Parameters
    ----------
    time : ndarray
        Time in datetime64 in ns units.

    Returns
    -------
    time_iso8601 : ndarray
        Time in ISO 8601 format.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """

    # Convert to required precision
    time_datetime64 = time.astype("<M8[ns]")

    # Convert to string
    time_iso8601 = time_datetime64.astype(str)
    time_iso8601 = np.atleast_1d(np.squeeze(np.stack([time_iso8601])))

    return time_iso8601
