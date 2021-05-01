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

"""iso86012datetime.py
@author: Louis Richard
"""


import datetime
import numpy as np


def iso86012datetime(time):
    r"""Converts ISO 8601 time to datetime

    Parameters
    ----------
    time : ndarray or list
        Time

    Returns
    -------
    time_datetime : list
        Time in datetime format.

    """

    # Make sure that str is in ISO8601 format
    time = np.array(time).astype("<M8[ns]").astype(str)

    # ISO 8601 format with miliseconds precision (max precision for datetime)
    fmt = "%Y-%m-%dT%H:%M:%S.%f"

    # Convert to datetime format
    time_datetime = [datetime.datetime.strptime(t_[:-3], fmt) for t_ in time]

    return time_datetime
