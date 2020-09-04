#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mean.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from . import resample, cross, norm, ts_vec_xyz


def mean(inp=None, r=None, b=None, z=None):
    """
    Put inp into mean field coordinates defined by position vector r and magnetic field b
    if earth magnetic dipole axis z is given then uses another algorithm (good for auroral passages)

    Parameters :
        inp : DataArray
            Input field to put into MF coordinates
        r : DataArray
            Position of the spacecraft
        b : DataArray
            Magnetic field
        z : DataArray
            Earth magnetic dipole axis

    Returns :
        out : DataArray
            Input field in mean field coordinates

    """

    # Check if there are at least 3 arguments
    if inp is None or r is None or b is None:
        raise ValueError("mean requires at least 3 arguments")

    # Check if inp is DataArray
    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a DataArray")

    # Check if r is DataArray
    if not isinstance(r, xr.DataArray):
        raise TypeError("r must be a DataArray")

    # Check if b is DataArray
    if not isinstance(b, xr.DataArray):
        raise TypeError("b must be a DataArray")

    #
    if z is not None:
        flag_dipole = True

        if not isinstance(z, xr.DataArray):
            raise TypeError("z must be a DataArray")
        elif len(z) != len(inp):
            zz = resample(z, inp)
        else:
            zz = z
    else:
        flag_dipole = False

    if len(r) != len(inp):
        rr = resample(r, inp)
    else:
        rr = r

    if len(b) != len(inp):
        bb = resample(b, inp)
    else:
        bb = b

    zv = norm(bb)

    if not flag_dipole:
        yv = cross(zv, rr)
        yv /= np.linalg.norm(yv, axis=1)[:, None]
    else:
        ss = np.sum(b * r)
        ind = ss > 0
        ss = -1 * np.ones(ss.shape)
        ss[ind] = 1
        yv = np.cross(zz, bb) * ss[:, None]
        yv /= np.linalg.norm(yv, axis=1)[:, None]

    xv = np.cross(yv, zv)

    # in case rotation axis is used as reference uncomment next line
    # rot_axis=rr;rot_axis(:,[2 3])=0;yv=irf_norm(irf_cross(irf_cross(bb,rot_axis),bb));xv=irf_cross(yv,zv);

    outdata = np.zeros(inp.data.shape)
    outdata[:, 0] = np.sum(xv * inp, axis=1)
    outdata[:, 1] = np.sum(yv * inp, axis=1)
    outdata[:, 2] = np.sum(zv * inp, axis=1)

    out = ts_vec_xyz(inp.time.data, outdata, inp.attrs)

    return out
