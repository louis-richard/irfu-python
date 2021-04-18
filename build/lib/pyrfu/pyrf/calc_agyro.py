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

"""calc_agyro.py
@author: Louis Richard
"""

import numpy as np


def calc_agyro(p_xyz):
    """Computes agyrotropy coefficient as

    .. math::

        A\\Phi = \\frac{|P_{\\perp 1} - P_{\\perp 2}|}{P_{\\perp 1}
        + P_{\\perp 2}}


    Parameters
    ----------
    p_xyz : xarray.DataArray
        Time series of the pressure tensor

    Returns
    -------
    agyro : xarray.DataArray
        Time series of the agyrotropy coefficient of the specie.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> ic = 1

    Load magnetic field and electron pressure tensor

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, 1)
    >>> p_xyz_e = mms.get_data("pe_gse_fpi_fast_l2", tint, 1)

    Rotate electron pressure tensor to field aligned coordinates

    >>> p_fac_e_qq = mms.rotate_tensor(p_xyz_e, "fac", b_xyz, "qq")

    Compute agyrotropy coefficient

    >>> agyro_e = pyrf.calc_agyro(p_fac_e_qq)

    """

    # Parallel and perpendicular components
    p_perp_1, p_perp_2 = [p_xyz[:, 1, 1], p_xyz[:, 2, 2]]

    agyro = np.abs(p_perp_1 - p_perp_2) / (p_perp_1 + p_perp_2)

    return agyro
