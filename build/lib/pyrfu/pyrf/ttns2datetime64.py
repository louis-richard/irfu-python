#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from cdflib import cdfepoch

# Local imports
from .timevec2iso8601 import timevec2iso8601

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def ttns2datetime64(time):
    r"""Convert time in epoch_tt2000 (nanosedconds since J2000) to datetime64
    in ns units.

    Parameters
    ----------
    time : ndarray
        Time in epoch_tt2000 (nanoseconds since J2000) format.

    Returns
    -------
    time_datetime64 : ndarray
        Time in datetime64 format in ns units.

    """

    #
    time_tt2000 = cdfepoch.breakdown_tt2000(time)

    # Convert to ISO 8601 string 'YYYY-MM-DDThh:mm:ss.mmmuuunnn'
    time_iso8601 = timevec2iso8601(time_tt2000)

    #
    time_datetime64 = time_iso8601.astype("<M8[ns]")

    return time_datetime64
