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
    """Put inp into mean field coordinates defined by position vector r and magnetic field b
    if earth magnetic dipole axis z is given then uses another algorithm (good for auroral passages)

    Parameters
    ----------
    inp : xarray.DataArray
        Input field to put into MF coordinates.

    r : xarray.DataArray
        Time series of the spacecraft position.

    b : xarray.DataArray
        Time series of the background magnetic field.

    z : xarray.DataArray
        Earth magnetic dipole axis.

    Returns
    -------
    out : xarray.DataArray
        Input field in mean field coordinates.

    """

    assert inp is not None and isinstance(inp, xr.DataArray)
    assert r is not None and isinstance(r, xr.DataArray)
    assert b is not None and isinstance(b, xr.DataArray)

    #
    if z is not None:
        flag_dipole = True

        if isinstance(z, xr.DataArray):
            if len(z) != len(inp):
                zz = resample(z, inp)
            else:
                zz = z
        else:
            raise TypeError("z must be a DataArray")

    else:
        flag_dipole = False
        zz = None

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
    # rot_axis=rr
    # rot_axis(:,[2 3])=0
    # yv=irf_norm(irf_cross(irf_cross(bb,rot_axis), bb));
    # xv=irf_cross(yv,zv);

    out_data = np.zeros(inp.data.shape)
    out_data[:, 0] = np.sum(xv * inp, axis=1)
    out_data[:, 1] = np.sum(yv * inp, axis=1)
    out_data[:, 2] = np.sum(zv * inp, axis=1)

    out = ts_vec_xyz(inp.time.data, out_data, inp.attrs)

    return out
