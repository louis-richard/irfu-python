#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os
from functools import lru_cache

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..models import igrf
from .ts_tensor_xyz import ts_tensor_xyz
from .ts_vec_xyz import ts_vec_xyz
from .unix2datetime64 import unix2datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


# Unix times of J2000 (2000-01-01T12:00:00 TT) and of 2000-01-01T12:00:00 UT. The
# first one gives TT seconds since J2000 after 2017 (TT - UTC = 69.184 s).
J2000_TT_UNIX = 946727930.8160001
J2000_UT_UNIX = 946728000.0


@lru_cache(maxsize=1)
def _transformation_indices():
    root_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "transformation_indices.json"

    with open(os.path.join(root_path, file_name), "r", encoding="utf-8") as file:
        return json.load(file)


def _triang(angle, axis):
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


def _dipole_direction_gse(time, flag: str = "dipole"):
    lambda_, phi = igrf(time, flag)

    cos_phi = np.cos(np.deg2rad(phi))
    dipole_direction_geo_ = np.stack(
        [
            cos_phi * np.cos(np.deg2rad(lambda_)),
            cos_phi * np.sin(np.deg2rad(lambda_)),
            np.sin(np.deg2rad(phi)),
        ],
    ).T
    dipole_direction_gse_ = cotrans(
        ts_vec_xyz(unix2datetime64(time), dipole_direction_geo_),
        "geo>gse",
    )

    return dipole_direction_gse_


def _transformation_matrix(t, tind, hapgood, *args):
    t_zero, ut, d0_j2000, d_j2000, h_j2000, t_j2000 = args

    transf_mat_out = np.zeros((len(t), 3, 3))
    transf_mat_out[:, 0, 0] = np.ones(len(t))
    transf_mat_out[:, 1, 1] = np.ones(len(t))
    transf_mat_out[:, 2, 2] = np.ones(len(t))

    for j, t_num in enumerate(tind[::-1]):
        assert abs(t_num) in list(range(1, 6)), "t_num must be +/- 1, 2, 3, 4, 5"

        if t_num in [-1, 1]:
            if hapgood:
                theta = 100.461 + 36000.770 * t_zero + 15.04107 * ut

            else:
                # Source: United States Naval Observatory, Astronomical
                # Applications Dept. http: // aa.usno.navy.mil / faq / docs
                # / GAST.php Last modified: 2011/06/14T14:04
                gmst = 6.697374558
                gmst += 0.06570982441908 * d0_j2000
                gmst += 1.00273790935 * h_j2000
                gmst += 0.000026 * t_j2000**2

                gmst = gmst % 24  # Interval 0->24 hours
                theta = (360 / 24) * gmst  # Convert to degree.

            # invert if tInd = -1
            transf_mat = _triang(theta * np.sign(t_num), 2)

        elif t_num in [-2, 2]:
            if hapgood:
                eps = 23.439 - 0.013 * t_zero
                # Suns mean anomaly
                m_anom = 357.528 + 35999.050 * t_zero + 0.04107 * ut
                # Suns mean longitude
                m_long = 280.460 + 36000.772 * t_zero + 0.04107 * ut
                l_sun = m_long
                l_sun += (1.915 - 0.0048 * t_zero) * np.sin(
                    np.deg2rad(m_anom),
                )
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

            transf_mat = np.matmul(_triang(l_sun, 2), _triang(eps, 0))
            if t_num == -2:
                transf_mat = np.transpose(transf_mat, [0, 2, 1])

        elif t_num in [-3, 3]:
            dipole_direction_gse_ = _dipole_direction_gse(t, "dipole")
            y_e = dipole_direction_gse_[:, 1]  # 1st col is time
            z_e = dipole_direction_gse_[:, 2]
            psi = np.rad2deg(np.arctan(y_e / z_e))

            transf_mat = _triang(-psi * np.sign(t_num), 0)  # inverse if -3

        elif t_num in [-4, 4]:
            dipole_direction_gse_ = _dipole_direction_gse(t, "dipole")

            mu = np.arctan(
                dipole_direction_gse_[:, 0]
                / np.sqrt(np.sum(dipole_direction_gse_[:, 1:] ** 2, axis=1)),
            )
            mu = np.rad2deg(mu)

            transf_mat = _triang(-mu * np.sign(t_num), 1)

        else:
            lambda_, phi = igrf(t, "dipole")

            transf_mat = np.matmul(_triang(phi - 90, 1), _triang(lambda_, 2))
            if t_num == -5:
                transf_mat = np.transpose(transf_mat, [0, 2, 1])

        if j == 0:
            transf_mat_out = transf_mat
        else:
            transf_mat_out = np.matmul(transf_mat, transf_mat_out)

    return transf_mat_out


def cotrans(inp, flag, hapgood: bool = True):
    r"""Coordinate transformation GE0/GEI/GSE/GSM/SM/MAG as described in [1]_

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input field.
    flag : str
        Coordinates transformation "{coord1}>{coord2}", where coord1 and
        coord2 can be geo/gei/gse/gsm/sm/mag (case-insensitive). If only
        "{coord2}" is given, coord1 is read from the COORDINATE_SYSTEM attribute
        of inp. "dipoledirectiongse" returns the dipole direction in GSE at the
        times of inp.
    hapgood : bool, Optional
        Indicator if original Hapgood sources should be used for angle
        computations or if updated USNO-AA sources should be used.
        Default = true, meaning original Hapgood sources.

    Returns
    -------
    out : xarray.DataArray
        Time series of the field in coord2. If inp is a scalar time series, the
        transformation matrices.

    Raises
    ------
    ValueError
        If the input frame is neither in flag nor in the COORDINATE_SYSTEM
        attribute of inp, if they differ, or if the transformation is unknown.

    Notes
    -----
    With hapgood=False, the Greenwich mean sidereal time is computed from UT, as
    the USNO formula requires. irfu-matlab computes it from TT, which rotates
    GEO by 0.29 deg about the z-axis, so the transformations between GEO or MAG
    and GEI, GSE, GSM or SM differ from irfu-matlab by up to 0.29 deg. The
    dipole direction (used for GSM and SM) is always computed with the Hapgood
    sidereal time.


    Examples
    --------
    >>> from pyrfu.mms import get_data
    >>> from pyrfu.pyrf import cotrans

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field in GSE coordinates

    >>> b_gse = get_data("b_gse_fgm_srvy_l2", tint, mms_id)

    Transform to GSM assuming that the original coordinates system is part
    of the inp metadata

    >>> b_gsm = cotrans(b_gse, 'GSM')

    If the original coordinates is not in the meta

    >>> b_gsm = cotrans(b_gse, 'GSE>GSM')

    Compute the dipole direction in GSE

    >>> dipole = cotrans(b_gse.time, 'dipoledirectiongse')


    References
    ----------
    .. [1]  Hapgood 1997 (corrected version of Hapgood 1992) Planet.Space Sci..Vol.
            40, No. 5. pp. 71l - 717, 1992

    """

    assert isinstance(inp, xr.DataArray), "inp must be a xarray.DataArray"
    assert inp.ndim < 3, "inp must be scalar or vector"

    flag = flag.lower()

    # Unix time in seconds (whatever the unit of the time coordinate)
    time = inp.time.data.astype("datetime64[ns]")
    t = (time.astype(np.int64) * 1e-9).astype(np.float64)

    if flag == "dipoledirectiongse":
        return _dipole_direction_gse(t)

    if ">" in flag:
        ref_syst_in, ref_syst_out = flag.split(">")
    else:
        ref_syst_in, ref_syst_out = [None, flag]

    if "COORDINATE_SYSTEM" in inp.attrs:
        ref_syst_internal = inp.attrs["COORDINATE_SYSTEM"].lower()
        ref_syst_internal = ref_syst_internal.split(">")[0]
    else:
        ref_syst_internal = None

    if ref_syst_in is None:
        if ref_syst_internal is None:
            raise ValueError(
                f"Transformation {flag} is unknown! Give the input frame in flag "
                "({coord1}>{coord2}) or in the COORDINATE_SYSTEM attribute of inp"
            )

        ref_syst_in = ref_syst_internal
    elif ref_syst_internal is not None and ref_syst_internal != ref_syst_in:
        raise ValueError(
            f"input frame in flag ({ref_syst_in}) and in the COORDINATE_SYSTEM "
            f"attribute of inp ({ref_syst_internal}) differ"
        )

    if ref_syst_in == ref_syst_out:
        return inp

    flag = f"{ref_syst_in}>{ref_syst_out}"
    transformation_indices = _transformation_indices()

    if flag not in transformation_indices:
        raise ValueError(f"Transformation {flag} is unknown!")

    tind = transformation_indices[flag]

    #  Terrestial Time (seconds since J2000)
    tts = t - J2000_TT_UNIX
    inp_ts = inp
    inp = inp.data

    if hapgood:
        day_start_epoch_dt64 = time.astype("datetime64[D]")
        day_start_epoch_dt64 = day_start_epoch_dt64.astype("datetime64[ns]")
        day_start_epoch = day_start_epoch_dt64.astype(np.int64) / 1e9

        # t_zero is time measured in Julian centuries from 2000-01-0112:00 UT
        # to the previous midnight
        t_zero = day_start_epoch - J2000_UT_UNIX
        t_zero /= 3600 * 24 * 36525.0

        hours = (time.astype("datetime64[h]") - time.astype("datetime64[D]")).astype(
            float
        )
        minutes = (time.astype("datetime64[m]") - time.astype("datetime64[h]")).astype(
            float
        )
        seconds = 1e-9 * (time - time.astype("datetime64[m]")).astype(np.float64)
        ut = hours + minutes / 60 + seconds / 3600

        args_trans_mat = (t_zero, ut, None, None, None, None)

    else:
        # Julian date (of req. time) from J2000, for the Sun's position
        d_j2000 = tts / 86400

        # The sidereal time formula takes UT days since 2000-01-01T12:00 UT. TT is
        # 69.184 s ahead, which would rotate GEO by 0.29 deg.
        d_ut = (t - J2000_UT_UNIX) / 86400

        # Julian date (of preceding midnight of req. time) from J2000
        d0_j2000 = np.floor(d_ut - 0.5) + 0.5

        # Julian centuries (of req. time) since J2000
        t_j2000 = d_ut / 36525

        # Hours in the of req. time (since midnight).
        h_j2000 = 24 * (d_ut - d0_j2000)

        args_trans_mat = (None, None, d0_j2000, d_j2000, h_j2000, t_j2000)

    transf_mat = _transformation_matrix(t, tind, hapgood, *args_trans_mat)

    if inp.ndim == 1:
        out = ts_tensor_xyz(inp_ts.time.data, transf_mat)

    else:
        out_data = np.einsum("kji,ki->kj", transf_mat, inp)
        out = inp_ts.copy()
        out.data = out_data
        out.attrs["COORDINATE_SYSTEM"] = ref_syst_out.upper()

    return out
