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

from astropy.time import Time


def iso2unix(t):
    """
    Converts time in iso format to unix

    Parameters
    ----------
    t : str or list of str
        Time.

    Returns
    -------
    out : float or list of float
        Time in unix format.

    """

    # Convert iso time to unix
    out = Time(t, format="iso").unix

    return out
