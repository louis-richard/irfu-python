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

"""iso86012datetime64.py
@author: Louis Richard
"""


def iso86012datetime64(time):
    r"""Convert ISO8601 time format to datetime64 in ns units.

    Parameters
    ----------
    time : ndarray
        Time in ISO 8601 format

    Returns
    -------
    time_datetime64 : ndarray
        Time in datetime64 in ns units.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """

    time_datetime64 = time.astype("<M8[ns]")

    return time_datetime64
