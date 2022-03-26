#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .resample import resample
from .cross import cross
from .normalize import normalize
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.13"
__status__ = "Prototype"


def mean(inp, r_xyz, b_xyz, dipole_axis):
    r"""Put inp into mean field coordinates defined by position vector r and
    magnetic field b if earth magnetic dipole axis z is given then  uses
    another algorithm (good for auroral passages)

    Parameters
    ----------
    inp : xarray.DataArray
        Input field to put into MF coordinates.
    r_xyz : xarray.DataArray
        Time series of the spacecraft position.
    b_xyz : xarray.DataArray
        Time series of the background magnetic field.
    dipole_axis : xarray.DataArray
        Earth magnetic dipole axis.

    Returns
    -------
    out : xarray.DataArray
        Input field in mean field coordinates.

    """

    if dipole_axis is not None:
        assert isinstance(dipole_axis, xr.DataArray)
        flag_dipole = True

        if len(dipole_axis) != len(inp):
            dipole_axis = resample(dipole_axis, inp)

    else:
        flag_dipole = False

    # Make sure that spacecraft position and magnetic field sampling matches
    # input sampling
    r_xyz = resample(r_xyz, inp)
    b_xyz = resample(b_xyz, inp)

    b_hat = normalize(b_xyz)

    if not flag_dipole:
        bxr = cross(b_hat, r_xyz)
        bxr /= np.linalg.norm(bxr, axis=1)[:, None]
    else:
        fact = -1 * np.ones(len(b_xyz))
        fact[np.sum(b_xyz * r_xyz) > 0] = 1
        bxr = np.cross(dipole_axis, b_xyz) * fact[:, None]
        bxr /= np.linalg.norm(bxr, axis=1)[:, None]

    bxrxb = np.cross(bxr, b_hat)

    out_data = np.zeros(inp.data.shape)
    out_data[:, 0] = np.sum(bxrxb * inp, axis=1)
    out_data[:, 1] = np.sum(bxr * inp, axis=1)
    out_data[:, 2] = np.sum(b_hat * inp, axis=1)

    out = ts_vec_xyz(inp.time.data, out_data, inp.attrs)

    return out
