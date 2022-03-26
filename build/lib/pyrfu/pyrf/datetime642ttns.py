#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

from cdflib import cdfepoch

# local imports
from .datetime642iso8601 import datetime642iso8601

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


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
