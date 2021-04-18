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

from .cotrans import cotrans


def l_shell(r_xyz):
    """Compute spacecraft position L Shell for a dipole magnetic field
    according to IRGF.

    Parameters
    ----------
    r_xyz : xarray.DataArray
        Time series of the spacecraft position. Must have a
        "COORDINATES_SYSTEM" attributes.

    Returns
    -------
    out : xarray.DataArray
        Time series of the spacecraft position L-Shell.

    """

    # Transform spacecraft coordinates to solar magnetic system
    r_sm = cotrans(r_xyz, "sm")

    # Compute Geomagnetic latitude
    lambda_ = np.arctan(r_sm[:, 2] / np.linalg.norm(r_sm[:, :2], axis=1))

    # Compute L shell
    out = np.linalg.norm(r_sm, axis=1) / np.cos(lambda_) ** 2

    return out
