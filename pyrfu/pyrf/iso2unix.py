#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def iso2unix(time):
    r"""Converts time in iso format to unix

    Parameters
    ----------
    time : str or list of str
        Time.

    Returns
    -------
    out : float or list of float
        Time in unix format.

    """

    # Convert iso time to unix
    out = np.array(time).astype("datetime64[ns]")

    return out
