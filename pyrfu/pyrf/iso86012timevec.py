#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import re

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def iso86012timevec(time):
    r"""Convert ISO 8601 time string into time vector.

    Parameters
    ----------
    time : ndarray or list or str
        Time in ISO 8601 format YYYY-MM-DDThh:mm:ss[.f], with up to 9
        digits of fraction of second.

    Returns
    -------
    time_vec : ndarray
        Time vector, of shape (n, 9): year, month, day, hours, minutes,
        seconds, milliseconds, microseconds and nanoseconds.

    Raises
    ------
    ValueError
        If a time is not in the format above.

    See Also
    --------
    pyrfu.pyrf.timevec2iso8601

    """

    iso_8601 = (
        r"(?P<years>[0-9]{4})-(?P<months>[0-9]{2})-(?P<days>[0-9]{2})"
        r"T(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9]{2})"
        r"(?:\.(?P<fraction>[0-9]{1,9}))?$"
    )

    # Define parser
    fmt = re.compile(iso_8601)

    # Make time is a 1d array
    time = np.atleast_1d(time)

    time_vec = []

    for t_ in time:
        match = fmt.match(str(t_))

        if match is None:
            raise ValueError(f"{t_} is not in ISO 8601 format YYYY-MM-DDThh:mm:ss[.f]")

        *date_time, fraction = match.groups(default="")

        # Pad the fraction of second to nanoseconds: ms, us, ns
        fraction = fraction.ljust(9, "0")
        sub_s = [fraction[:3], fraction[3:6], fraction[6:]]

        time_vec.append([int(p_) for p_ in [*date_time, *sub_s]])

    time_vec = np.array(time_vec)

    return time_vec
