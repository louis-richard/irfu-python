#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""start.py
@author: Louis Richard
"""


def start(inp):
    """Gives the first time of the time series.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series.

    Returns
    -------
    out : float
        Value of the first time in the desired format.

    """

    out = inp.time.data[0].astype(int) * 1e-9

    return out
