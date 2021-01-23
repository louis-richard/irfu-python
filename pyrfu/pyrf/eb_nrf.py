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
import xarray as xr

from . import resample, dot, normalize, cross


def eb_nrf(e, b, v, flag=0):
    """Find E and B in MP system given B and MP normal vector.

    Parameters
    ----------
    e : xarray.DataArray
        Time series of the electric field.

    b : xarray.DataArray
        Time series of the magnetic field.

    v : xarray.DataArray
        Normal vector.

    flag : int or ndarray
        to fill.

    Returns
    -------
    out : xarray.DataArray
        to fill.

    """

    if isinstance(flag, int):
        if flag == 1:
            flag_case = "b"
        else:
            flag_case = "a"
        
        l_direction = None

    elif isinstance(flag, np.ndarray) and np.size(flag) == 3:
        l_direction = flag
        flag_case = "c"
    else:
        raise TypeError("Invalid flag type")

    if flag_case == "a":
        be = resample(b, e).data

        nl = be / np.linalg.norm(be, axis=0)[:, None]  # along the B
        nn = np.cross(np.cross(be, v), be)  # closest to given vn vector
        nn = nn / np.linalg.norm(nn)[:, None]
        nm = np.cross(nn, nl)  # in (vn x b) direction

        # estimate e in new coordinates
        en = dot(e, nn)
        el = dot(e, nl)
        em = dot(e, nm)
        emp = np.hstack([el, em, en])

    elif flag_case == "b":
        nn = v / np.linalg.norm(v)
        nm = normalize(cross(nn, b.mean(dim="time")))
        nl = cross(nm, nn)

        # estimate e in new coordinates
        en = dot(e, nn)
        el = dot(e, nl)
        em = dot(e, nm)
        emp = np.hstack([el, em, en])

    elif flag_case == "c":
        nn = normalize(v)
        nm = normalize(np.cross(nn, l_direction))
        nl = cross(nm, nn)

        # estimate e in new coordinates
        en = dot(e, nn)
        el = dot(e, nl)
        em = dot(e, nm)

        emp = np.hstack([el, em, en])
    else:
        raise ValueError("Invalid flag_case")

    out = xr.DataArray(e.time.data, emp, e.attrs)

    return out
