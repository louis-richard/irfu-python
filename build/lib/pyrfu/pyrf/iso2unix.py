#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
iso2unix.py

@author : Louis RICHARD
"""

import numpy as np

from astropy.time import Time


def iso2unix(t=None):
    """
    Converts time in iso format to unix

    Parameters
    ----------
    t : list of str
        Time.

    Returns
    -------
    out : list of float
        Time in unix format.

    """

    assert t is not None and isinstance(t, (list, np.ndarray))

    # Convert iso time to unix
    out = Time(t, format="iso").unix

    return out
