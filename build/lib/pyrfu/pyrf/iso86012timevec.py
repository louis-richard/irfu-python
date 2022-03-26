#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import re

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


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
