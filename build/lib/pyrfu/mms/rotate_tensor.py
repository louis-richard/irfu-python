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

from ..pyrf import resample, ts_tensor_xyz


def rotate_tensor(*args):
    """Rotates pressure or temperature tensor into another coordinate system.

    Parameters
    ----------
    PeIJ/Peall : xarray.DataArray
        Time series of either separated terms of the tensor or the complete tensor.
        If columns (PeXX,PeXY,PeXZ,PeYY,PeYZ,PeZZ)

    flag : str
        Flag of the target coordinates system :
            * "fac" : 	Field-aligned coordinates, requires background magnetic field Bback,
                        optional flag "pp" :math:`\\mathbf{P}_{\\perp 1} = \\mathbf{P}_{\\perp
                        2}` or "qq" :math:`\\mathbf{P}_{\\perp 1}` and
                        :math:`\\mathbf{P}_{\\perp 2}` are most unequal, sets P23 to zero.

            * "rot" :	Arbitrary coordinate system, requires new x-direction xnew, new y and z
                        directions ynew, znew (if not included y and z directions are orthogonal
                        to xnew and closest to the original y and z directions)

            * "gse" : GSE coordinates, requires MMS spacecraft number 1--4 MMSnum

    Returns
    -------
    Pe : xarray.DataArray
        Time series of the pressure or temperature tensor in field-aligned, user-defined,
        or GSE coordinates. For "fac" Pe = [Ppar P12 P13; P12 Pperp1 P23; P13 P23 Pperp2].
        For "rot" and "gse" Pe = [Pxx Pxy Pxz; Pxy Pyy Pyz; Pxz Pyz Pzz]

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> # Time interval
    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]
    >>> # Spacecraft index
    >>> mms_id = 1
    >>> # Load magnetic field and ion temperature tensor
    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> t_xyz_i = mms.get_data("Ti_gse_fpi_fast_l2", tint, mms_id)
    >>> # Compute ion temperature in field aligned coordinates
    >>> t_xyzfac_i = mms.rotate_tensor(t_xyz_i, "fac", b_xyz, "pp")

    TODO : change input, add check that vectors are orthogonal L145
    """

    nargin = len(args)

    # Check input and load pressure/temperature terms
    if isinstance(args[1], str):
        rot_flag = args[1]
        rot_flag_pos = 1
        p_all = args[0]
        p_times = p_all.time.data

        if p_all.data.ndim == 3:
            p_tensor = p_all
        else:
            p_tensor = np.reshape(p_all.data, (p_all.shape[0], 3, 3))
            p_tensor = ts_tensor_xyz(p_times, p_tensor)
    else:
        raise SystemError("critical','Something is wrong with the input.")

    ppeq, qqeq = [0, 0]

    rot_mat = np.zeros((len(p_times), 3, 3))

    if rot_flag[0] == "f":
        print("notice : Transforming tensor into field-aligned coordinates.")

        if nargin == rot_flag_pos:
            raise ValueError("B TSeries is missing.")

        b_back = args[rot_flag_pos + 1]
        b_back = resample(b_back, p_tensor)

        if nargin == 4:
            if isinstance(args[3], str) and args[3][0] == "p":
                ppeq = 1
            elif isinstance(args[3], str) and args[3][0] == "q":
                qqeq = 1
            else:
                raise ValueError("Flag not recognized no additional rotations applied.")

        if nargin == 9:
            if isinstance(args[8], str) and args[8][0] == "p":
                ppeq = 1
            elif isinstance(args[8], str) and args[8][0] == "q":
                qqeq = 1
            else:
                raise ValueError("Flag not recognized no additional rotations applied.")

        b_vec = b_back / np.linalg.norm(b_back, axis=1, keepdims=True)

        r_x = b_vec.data
        r_y = np.array([1, 0, 0])
        r_z = np.cross(r_x, r_y)
        r_z /= np.linalg.norm(r_z, axis=1, keepdims=True)
        r_y = np.cross(r_z, r_x)
        r_y /= np.linalg.norm(r_y, axis=1, keepdims=True)

        rot_mat[:, 0, :], rot_mat[:, 1, :], rot_mat[:, 2, :] = [r_x, r_y, r_z]

    elif rot_flag[0] == "r":
        print("notice : Transforming tensor into user defined coordinate system.")

        if nargin == rot_flag_pos:
            raise ValueError("Vector(s) is(are) missing.")

        vectors = list(args[rot_flag_pos + 1:])

        if len(vectors) == 1:
            r_x = vectors[0]

            if len(r_x) != 3:
                raise TypeError("Vector format not recognized.")

            r_x /= np.linalg.norm(r_x, keepdims=True)
            r_y = np.array([0, 1, 0])
            r_z = np.cross(r_x, r_y)
            r_z /= np.linalg.norm(r_z, keepdims=True)
            r_y = np.cross(r_z, r_x)
            r_y /= np.linalg.norm(r_y, keepdims=True)

        elif len(vectors) == 3:
            r_x, r_y, r_z = [r / np.linalg.norm(r, keepdims=True) for r in vectors]

        else:
            raise TypeError("Vector format not recognized.")

        rot_mat[:, 0, :] = np.ones((len(p_times), 1)) * r_x
        rot_mat[:, 1, :] = np.ones((len(p_times), 1)) * r_y
        rot_mat[:, 2, :] = np.ones((len(p_times), 1)) * r_z

    else:
        raise ValueError("Flag is not recognized.")

    p_tensor_p = np.zeros((len(p_times), 3, 3))

    for i in range(len(p_times)):
        rot_temp = np.squeeze(rot_mat[i, :, :])

        p_tensor_p[i, :, :] = np.matmul(np.matmul(rot_temp, np.squeeze(p_tensor.data[i, :, :])),
                                        np.transpose(rot_temp))

    if ppeq:
        print("notice : Rotating tensor so perpendicular diagonal components are equal.")
        thetas = 0.5 * np.arctan(
            (p_tensor_p[:, 2, 2] - p_tensor_p[:, 1, 1]) / (2 * p_tensor_p[:, 1, 2]))

        for i, theta in enumerate(thetas):
            if np.isnan(theta):
                theta = 0

            rot_temp = np.array(
                [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]])

            p_tensor_p[i, :, :] = np.matmul(
                np.matmul(rot_temp, np.squeeze(p_tensor_p[i, :, :])), np.transpose(rot_temp))

    if qqeq:
        print("notice : Rotating tensor so perpendicular diagonal components are most unequal.")
        thetas = 0.5 * np.arctan(
            (2 * p_tensor_p[:, 1, 2]) / (p_tensor_p[:, 2, 2] - p_tensor_p[:, 1, 1]))

        for i, theta in enumerate(thetas):
            rot_temp = np.array(
                [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

            p_tensor_p[i, :, :] = np.matmul(
                np.matmul(rot_temp, np.squeeze(p_tensor_p[i, :, :])), np.transpose(rot_temp))

    # Construct output
    p_new = ts_tensor_xyz(p_times, p_tensor_p)

    try:
        p_new.attrs["units"] = args[0].attrs["units"]
    except KeyError:
        pass

    return p_new
