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

"""datetime_to_tt2000.py
@author: Louis Richard
"""

import pandas as pd


def datetime_to_tt2000(time):
    """Transforms datetime to TT2000 string format.

    Parameters
    ----------
    time : datetime.datetime
        Time to convert to tt2000 string.

    Returns
    -------
    tt2000 : str
        Time in TT20000 iso_8601 format.

    """

    time_datetime = pd.Timestamp(time)

    # Convert to string
    tt2000 = "{}{:03d}".format(time_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                              time_datetime.nanosecond)

    return tt2000
