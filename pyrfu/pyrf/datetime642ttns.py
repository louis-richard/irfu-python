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

"""datetime642ttns.py
@author: Louis Richard
"""

import numpy as np

from cdflib import cdfepoch

from .datetime642iso8601 import datetime642iso8601


def datetime642ttns(time):
    r"""Converts datetime64 in ns units to epoch_tt2000
    (nanoseconds since J2000).

    Parameters
    ----------
    time : ndarray
        Times in datetime64 format.

    Returns
    -------
    time_ttns : ndarray
        Times in epoch_tt2000 format (nanoseconds since J2000).

    """

    # Convert to datetime64 in ns units
    time_iso8601 = datetime642iso8601(time)

    # Convert to ttns
    time_ttns = np.array([cdfepoch.parse(t_) for t_ in time_iso8601])

    return time_ttns
