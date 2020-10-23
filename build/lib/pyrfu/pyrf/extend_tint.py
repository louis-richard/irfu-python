#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extend_tint.py

@author : Louis RICHARD
"""

from dateutil.parser import parse as date_parse
from astropy.time import Time


def extend_tint(tint=None, ext=None):
    """Extends time interval.

    Parameters
    ----------
    tint : list of str
        Reference time interval to extend.

    ext : list of float or int
        Number of seconds to extend time interval [left extend, right extend].

    Returns
    -------
    tint : list of str
        Extended time interval.

    Examples
    --------
    >>> from pyrfu import pyrf
    >>> # Time interval
    >>> tint = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]
    >>> # Spacecraft index
    >>> ic = 3
    >>> # Load spacecraft position
    >>> tint_long = pyrf.extend_tint(tint, [-100, 100])

    """

    if ext is None:
        ext = [-60, 60]

    # Convert to unix format
    start_time, end_time = [Time(date_parse(tint_bound), format="datetime").unix for tint_bound in
                            tint]

    # extend interval
    start_time, end_time = [start_time + ext[0], end_time + ext[1]]

    # back to iso format
    start_time, end_time = [Time(bound, format="unix").iso for bound in [start_time, end_time]]

    tint_long = [start_time, end_time]

    return tint_long
