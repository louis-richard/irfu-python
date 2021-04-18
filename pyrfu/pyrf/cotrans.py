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

"""geocentric_coordinate_transformation.py
@author: Louis Richard
"""

import os
import yaml
import numpy as np
import xarray as xr

from astropy.time import Time
from ..models import igrf

from .ts_vec_xyz import ts_vec_xyz


def triang(angle, axis):
    """angle in degrees ax=1 > X, ax=2 > y,ax=3 > z)

    Parameters
    ----------
    angle : ndarray
        Angles in degres.

    axis : int
        Index of the rotation axis 0, 1, 2

    Returns
    -------
    transf_mat : ndarray
        Matrix of rotation of the angle around the axis.

    """

    cos_angle = np.cos(np.deg2rad(angle))
    sin_angle = np.sin(np.deg2rad(angle))

    axes = list(filter(lambda x: x != axis, np.arange(3)))

    transf_mat = np.zeros((len(angle), 3, 3))
    transf_mat[:, axes[0], axes[0]] = cos_angle
    transf_mat[:, axes[1], axes[1]] = cos_angle
    transf_mat[:, axis, axis] = 1
    transf_mat[:, axes[0], axes[1]] = sin_angle
    transf_mat[:, axes[1], axes[0]] = -sin_angle

    return transf_mat


def dipole_direction_gse(time, flag: str = "dipole"):
    """Compute IGRF model dipole in GSE coordinates.

    Parameters
    ----------
    time : ndarray
        Times in unix format.

    flag : str
        Default is "dipole"


    Returns
    -------
    dipole_direction_gse_ : ndarray
        Direction of the IGRF dipole model in GSE coordinates. [t, x, y, z]
    """

    lambda_, phi = igrf(time, flag)

    cos_phi = np.cos(np.deg2rad(phi))
    dipole_direction_geo_ = np.stack([cos_phi * np.cos(np.deg2rad(lambda_)),
                                      cos_phi * np.sin(np.deg2rad(lambda_)),
                                      np.sin(np.deg2rad(phi))]).T

    dipole_direction_gse_ = geocentric_coordinate_transformation(np.hstack([
        time[:, None], dipole_direction_geo_]), "geo>gse")

    return dipole_direction_gse_


def transformation_matrix(t, tind, hapgood, *args):
    """Compute the transformation matrix associated to the selected
    coordinates system change.

    Parameters
    ----------
    t : ndarray
        Time in unix format.

    tind : list
        Transformation indices.

    hapgood : bool
        Flag to use Hapgoog

    args : tuple
        to fill.

    Returns
    -------
    transf_mat_out : ndarray
        Transformation matrix


    """
    t_zero, ut, d0_j2000, d_j2000, h_j2000, t_j2000 = args

    transf_mat_out = np.zeros((len(t), 3, 3))
    transf_mat_out[:, 0, 0] = np.ones(len(t))
    transf_mat_out[:, 1, 1] = np.ones(len(t))
    transf_mat_out[:, 2, 2] = np.ones(len(t))

    for j, t_num in enumerate(tind[::-1]):
        if t_num == 1 or t_num == -1:
            if hapgood:
                theta = 100.461 + 36000.770 * t_zero + 15.04107 * ut

            else:
                # Source: United States Naval Observatory, Astronomical
                # Applications Dept. http: // aa.usno.navy.mil / faq / docs
                # / GAST.php Last modified: 2011/06/14T14:04
                gmst = 6.697374558
                gmst += 0.06570982441908 * d0_j2000
                gmst += 1.00273790935 * h_j2000
                gmst += 0.000026 * t_j2000 ** 2

                gmst = gmst % 24  # Interval 0->24 hours
                theta = (360 / 24) * gmst  # Convert to degree.

            # invert if tInd = -1
            transf_mat = triang(theta * np.sign(t_num), 2)

        elif t_num == 2 or t_num == -2:
            if hapgood:
                eps = 23.439 - 0.013 * t_zero
                # Suns mean anomaly
                m_anom = 357.528 + 35999.050 * t_zero + 0.04107 * ut
                # Suns mean longitude
                m_long = 280.460 + 36000.772 * t_zero + 0.04107 * ut
                l_sun = m_long
                l_sun += (1.915 - 0.0048 * t_zero) * np.sin(np.deg2rad(m_anom))
                l_sun += 0.020 * np.sin(np.deg2rad(2 * m_anom))
            else:
                # Source: United States Naval Observatory, Astronomical
                # Applications Dept.
                # http://aa.usno.navy.mil/faq/docsSunApprox.php.
                # Last modified: 2012/11/06T14:12
                eps = 23.439 - 0.00000036 * d_j2000
                m_anom = 357.529 + 0.98560028 * d_j2000
                m_long = 280.459 + 0.98564736 * d_j2000
                l_sun = m_long
                l_sun += 1.915 * np.sin(np.deg2rad(m_anom))
                l_sun += 0.020 * np.sin(np.deg2rad(2 * m_anom))

            transf_mat = np.matmul(triang(l_sun, 2), triang(eps, 0))
            if t_num == -2:
                transf_mat = np.transpose(transf_mat, [0, 2, 1])

        elif t_num == 3 or t_num == -3:
            dipole_direction_gse_ = dipole_direction_gse(t, "dipole")
            y_e = dipole_direction_gse_[:, 2]  # 1st col is time
            z_e = dipole_direction_gse_[:, 3]
            psi = np.rad2deg(np.arctan(y_e / z_e))

            transf_mat = triang(-psi * np.sign(t_num), 0)  # inverse if -3

        elif t_num == 4 or t_num == -4:
            dipole_direction_gse_ = dipole_direction_gse(t, "dipole")

            mu = np.arctan(dipole_direction_gse_[:, 1]
                           / np.sqrt(np.sum(dipole_direction_gse_[:, 2:] ** 2,
                                            axis=1)))
            mu = np.rad2deg(mu)

            transf_mat = triang(-mu * np.sign(t_num), 1)

        elif t_num == 5 or t_num == -5:
            lambda_, phi = igrf(t, "dipole")

            transf_mat = np.matmul(triang(phi - 90, 1), triang(lambda_, 2))
            if t_num == -5:
                transf_mat = np.transpose(transf_mat, [0, 2, 1])

        else:
            raise ValueError

        if j == len(tind):
            transf_mat_out = transf_mat
        else:
            transf_mat_out = np.matmul(transf_mat, transf_mat_out)

    return transf_mat_out


def geocentric_coordinate_transformation(inp, flag, hapgood: bool = True):
    """IRF.GEOCENTRIC_COORDINATE_TRANSFORMATION coordinate transformation
    GE0/GEI/GSE/GSM/SM/MAG

    Parameters
    ----------
    inp : xarray.DataArray or ndarray
        Time series of the input field.

    flag : str
        Coordinates transformation "{coord1}>{coord2}", where coord1 and
        coord2 can be geo/gei/gse/gsm/sm/mag.

    hapgood : bool
        Indicator if original Hapgood sources should be used for angle
        computations or if updated USNO-AA sources should be used.
        Default = true, meaning original Hapgood sources.


    Examples
    --------
    >>> from pyrfu.mms import get_data
    >>> from pyrfu.pyrf import geocentric_coordinate_transformation

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field in GSE coordinates

    >>> b_gse = get_data("b_gse_fgm_srvy_l2", tint, mms_id)

    Transform to GSM assuming that the original coordinates system is part
    of the inp metadata

    >>> b_gsm = geocentric_coordinate_transformation(b_gse, 'GSM')

    If the original coordinates is not in the meta

    >>> b_gsm = geocentric_coordinate_transformation(b_gse, 'GSE>GSM')

    Compute the dipole direction in GSE

    >>> dipole = geocentric_coordinate_transformation(b_gse.time,
    'dipoledirectiongse')


    References
    ----------
    .. [17]     Hapgood 1997 (corrected version of Hapgood 1992) Planet.Space
                Sci..Vol. 40, No. 5. pp. 71l - 717, 1992

    .. [18]     USNO - AA 2011 & 2012

    """

    if ">" in flag:
        ref_syst_in, ref_syst_out = flag.split(">")
    else:
        ref_syst_in, ref_syst_out = [None, flag.lower()]

    if isinstance(inp, xr.DataArray):
        if "COORDINATE_SYSTEM" in inp.attrs:
            ref_syst_internal = inp.attrs["COORDINATE_SYSTEM"].lower()
            ref_syst_internal = ref_syst_internal.split(">")[0]
        else:
            ref_syst_internal = None

        if ref_syst_in is None and ref_syst_internal is None:
            raise ValueError("input reference frame undefined")

        elif ref_syst_in is not None and ref_syst_internal is not None:
            message = "input ref. frame in variable and input flag differs"
            assert ref_syst_internal == ref_syst_in, message

        elif ref_syst_in is None and ref_syst_internal is not None:
            ref_syst_in = ref_syst_internal.lower()

        flag = f"{ref_syst_in}>{ref_syst_out}"

    if ref_syst_in == ref_syst_out:
        return inp

    # J2000 reference time
    j2000 = 946727930.8160001
    # j2000 = Time("J2000", format="jyear_str").unix

    if isinstance(inp, xr.DataArray):
        time = inp.time.data
        t = time.view("i8") * 1e-9

        #  Terrestial Time (seconds since J2000)
        tts = t - j2000
        inp_ts = inp
        inp = inp.data

    elif isinstance(inp, np.ndarray):
        time = Time(inp[:, 0], format="unix").datetime64
        t = inp[:, 0]
        #  Terrestial Time (seconds since J2000)
        tts = t - j2000
        inp_ts = None
        inp = inp[:, 1:]
    else:
        raise TypeError("invalid inpu")

    if hapgood:
        day_start_epoch = Time(time.astype("datetime64[D]"),
                               format="datetime64").unix
        mjd_ref_epoch = Time("2000-01-01T12:00:00", format="isot").unix

        # t_zero is time measured in Julian centuries from 2000-01-0112:00 UT
        # to the previous midnight
        t_zero = (day_start_epoch - mjd_ref_epoch)
        t_zero /= (3600 * 24 * 36525.0)

        hours = (time.astype('datetime64[h]')
                 - time.astype('datetime64[D]')).astype(float)
        minutes = (time.astype('datetime64[m]')
                   - time.astype('datetime64[h]')).astype(float)
        seconds = 1e-9 * (time.astype('datetime64[ns]')
                          - time.astype('datetime64[m]')).astype(float)
        ut = hours + minutes / 60 + seconds / 3600

        args_trans_mat = (t_zero, ut, None, None, None, None)

    else:
        # Julian date(of req.time) from J2000
        d_j2000 = tts / 86400

        # Julian date(of preceeding midnight of req.time) from J2000
        d0_j2000 = np.floor(tts / 86400) - 0.5

        # Julian centuries(of req.time) since J2000
        t_j2000 = d_j2000 / 36525

        # Hours in the of req.time(since midnight).
        h_j2000 = 24 * (d_j2000 - d0_j2000)

        args_trans_mat = (None, None, d0_j2000, d_j2000, h_j2000, t_j2000)

    if ">" in flag:
        root_path = os.path.dirname(os.path.abspath(__file__))
        file_name = "transformation_indices.yml"

        with open(os.sep.join([root_path, file_name])) as file:
            transformation_dict = yaml.load(file, Loader=yaml.FullLoader)

        tind = transformation_dict[flag]

    elif flag == "dipoledirectiongse":
        out_data = dipole_direction_gse(t)
        return ts_vec_xyz(inp.time.data, out_data)

    else:
        raise ValueError(f"Transformation {flag} is unknown!")

    transf_mat = transformation_matrix(t, tind, hapgood, *args_trans_mat)

    if inp.ndim == 2:
        out = np.einsum('kji,ki->kj', transf_mat, inp)
    elif inp.ndim == 1:
        out = transf_mat
    else:
        raise ValueError

    if inp_ts is not None:
        out_data = out
        out = inp_ts.copy()
        out.data = out_data
        out.attrs["COORDINATE_SYSTEM"] = ref_syst_out.upper()

    else:
        out = np.hstack([t[:, None], out])

    return out
