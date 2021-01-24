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

from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def cross(x, y):
    """Computes cross product of two fields.

    Parameters
    ----------
    x : xarray.DataArray
        Time series of the first field X.

    y : xarray.DataArray
        Time series of the second field Y.

    Returns
    -------
    out : xarray.DataArray
        Time series of the cross product Z = XxY.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Define time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Index of the MMS spacecraft

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Compute magnitude of the magnetic field

    >>> b_mag = pyrf.norm(b_xyz)

    Compute ExB drift velocity

    >>> v_xyz_exb = pyrf.cross(e_xyz, b_xyz) / b_mag ** 2

    """

    if len(x) != len(y):
        y = resample(y, x)

    out_data = np.cross(x.data, y.data, axis=1)

    out = ts_vec_xyz(x.time.data, out_data)

    return out
