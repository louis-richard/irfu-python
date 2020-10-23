#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eb_nrf.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from . import resample, dot, normalize, cross


def eb_nrf(e=None, b=None, v=None, flag=0):
    """Find E and B in MP system given B and MP normal vector.

    Parameters
    ----------
    e : xarray.DataArray
        Time series of the electric field.

    b : xarray.DataArray
        Time series of the magnetic field.

    v : xarray.DataArray
        Normal vector.

    flag : int or numpy.ndarray
        to fill.

    Returns
    -------
    out : DataArray
        to fill.

    """

    assert e is not None and isinstance(e, xr.DataArray)
    assert b is not None and isinstance(b, xr.DataArray)
    assert v is not None and isinstance(v, xr.DataArray)

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
        nm = normalize(np.cross(nn, np.mean(b)))
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
