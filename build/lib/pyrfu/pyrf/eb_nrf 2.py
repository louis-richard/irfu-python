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
    """
    Find E and B in MP system given B and MP normal vector

    Parameters :
        e : DataArray
            Time series of the electric field
        b : DataArray
            Time series of the magnetic field

        v : DataArray
            Normal vector

    Return :
        out : DataArray

    """

    if e is None:
        raise ValueError("eb_nrf requires at least 3 arguments")
    elif not isinstance(e, xr.DataArray):
        raise TypeError("e must be a DataArray")

    if b is None:
        raise ValueError("eb_nrf requires at least 3 arguments")
    elif not isinstance(b, xr.DataArray):
        raise TypeError("b must be a DataArray")

    if v is None:
        raise ValueError("eb_nrf requires at least 3 arguments")
    elif not isinstance(v, xr.DataArray):
        raise TypeError("v must be a DataArray")

    if isinstance(flag, int):
        if flag == 1:
            flag_case = "B"
        else:
            flag_case = "A"
        
        l_direction = None

    elif isinstance(flag, np.ndarray) and np.size(flag) == 3:
        l_direction = flag
        flag_case = "C"
    else:
        raise TypeError("Invalid flag type")

    if flag_case == "A":
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

    elif flag_case == "B":
        nn = v / np.linalg.norm(v)
        nm = normalize(np.cross(nn, np.mean(b)))
        nl = cross(nm, nn)

        # estimate e in new coordinates
        en = dot(e, nn)
        el = dot(e, nl)
        em = dot(e, nm)
        emp = np.hstack([el, em, en])

    elif flag_case == "C":
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
