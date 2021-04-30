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

"""timevec2iso8601.py
@author: Louis Richard
"""

import numpy as np


def timevec2iso8601(time):
    r"""Convert time vector into ISO 8601 format YYYY-MM-DDThh:mm:ss.mmmuuunnn.

    Parameters
    ----------
    time : ndarray
        Time vector

    Returns
    -------
    time_iso8601 : ndarray
        Time in ISO 8601 format.

    See Also
    --------
    pyrfu.pyrf.iso86012timevec

    """

    time = np.atleast_2d(np.array(time))
    time = np.hstack([time, np.zeros((len(time), 9 - time.shape[1]))])

    time_iso8601 = []

    for t_ in time.astype(np.int64):
        ye_mo_da_ = f"{t_[0]:04}-{t_[1]:02}-{t_[2]:02}"  # YYYY-MM-DD
        ho_mi_se_ = f"{t_[3]:02}:{t_[4]:02}:{t_[5]:02}"  # hh:mm:ss
        ms_us_ns_ = f"{t_[6]:03}{t_[7]:03}{t_[8]:03}"    # mmmuuunnn

        # Time as ISO 8601 string 'YYYY-MM-DDThh:mm:ss.mmmuuunnn'
        time_iso8601.append(f"{ye_mo_da_}T{ho_mi_se_}.{ms_us_ns_}")

    time_iso8601 = np.array(time_iso8601)

    return time_iso8601
