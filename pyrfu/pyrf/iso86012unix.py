#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def iso86012unix(time):
    r"""Converts time in iso format to unix

    Parameters
    ----------
    time : str or array_like of str
        Time.

    Returns
    -------
    out : float or list of float
        Time in unix format.

    """

    assert isinstance(time, (str, list, np.ndarray)), "time must be a str or array_like"

    out = np.atleast_1d(time).astype("datetime64[ns]")

    return out
