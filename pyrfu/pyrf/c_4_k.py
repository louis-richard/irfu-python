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

import numpy as np

from .cross import cross
from .dot import dot


def c_4_k(r_list):
    """Calculates reciprocal vectors in barycentric coordinates.

    Parameters
    ----------
    r_list : list of xarray.DataArray
        Position of the spacecrafts.

    Returns
    -------
    k_list : list of xarray.DataArray
        Reciprocal vectors in barycentric coordinates.

    Notes
    -----
    The units of reciprocal vectors are the same as [1/r].

    """

    mms_list = np.arange(4)

    k_list = [r_list[0].copy()] * 4

    for i, j, k, l in zip(mms_list, np.roll(mms_list, 1), np.roll(mms_list, 2),
                          np.roll(mms_list, 3)):
        cc = cross(r_list[k]-r_list[j], r_list[l]-r_list[j])

        dr12 = r_list[i]-r_list[j]

        k_list[i] = cc/dot(cc, dr12)

    return k_list
