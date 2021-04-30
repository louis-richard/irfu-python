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

"""datetime642unix.py
@author: Louis Richard
"""


def datetime642unix(time):
    r"""Converts datetime64 in ns units to unix time.

    Parameters
    ----------
    time : ndarray
        Time in datetime64 format.


    Returns
    -------
    time_unix : ndarray
        Time in unix format.


    See Also
    --------
    pyrfu.pyrf.unix2datetime64

    """

    # Make sure that time is in ns format
    time_ns = time.astype("<M8[ns]")

    # Convert to unix
    time_unix = time_ns.astype("int64") / 1e9

    return time_unix
