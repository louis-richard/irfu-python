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

"""unix2datetime64.py
@author: Louis Richard
"""


def unix2datetime64(time):
    r"""Converts unix time to datetime64 in ns units.

    Parameters
    ----------
    time : ndarray
        Time in unix format.


    Returns
    -------
    time_datetime64 : ndarray
        Time in datetime64 format.


    See Also
    --------
    pyrfu.pyrf.datetime642unix

    """

    # Make sure that time is in ns format
    time_unix = (time * 1e9).astype("int64")

    # Convert to unix
    time_datetime64 = time_unix.astype("<M8[ns]")

    return time_datetime64
