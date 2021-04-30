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

import re
import numpy as np


def iso86012timevec(time):
    r"""Convert ISO 8601 time string into time vector.

    Parameters
    ---------
    time : str
        Time in ISO 8601 format YYYY-MM-DDThh:mm:ss.mmmuuunnn.

    Returns
    -------
    time_vec : list
        Time vector.

    See Also
    --------
    pyrfu.pyrf.iso86012timevec

    """

    iso_8601 = r"(?P<years>[0-9]{4})-(?P<months>[0-9]{2})-(?P<days>[0-9]{2})" \
               r"T(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2})" \
               r":(?P<seconds>[0-9]{2}).(?P<miliseconds>[0-9]{3})" \
               r"(?P<microseconds>[0-9]{3})(?P<nanoseconds>[0-9]{3})"

    # Define parser
    fmt = re.compile(iso_8601)

    time_vec = [[int(p_) for p_ in fmt.match(t_).groups()] for t_ in time]
    time_vec = np.array(time_vec)

    return time_vec
