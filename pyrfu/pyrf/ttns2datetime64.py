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

"""ttns2datetime64.py
@author: Louis Richard
"""


from cdflib import cdfepoch

from .timevec2iso8601 import timevec2iso8601


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
