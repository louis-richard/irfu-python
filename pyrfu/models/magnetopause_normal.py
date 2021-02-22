#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2021 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""magnetopause_normal.py
@author: Louis Richard
"""

import warnings
import numpy as np

from scipy.optimize import fminbound


def magnetopause(theta, *args):
    """Magnetopause model to minimize.

    Parameters
    ----------
    theta : float
        to fill

    args : [float, float, float, float]
        extra arguments.

    Returns
    -------
    out : float
        to fill

    """

    r0, alpha, x0, y0 = args
    out = r0 ** 2 * (2. / (1 + np.cos(theta))) ** (2 * alpha) \
          - 2 * r0 * (2. / (1 + np.cos(theta))) ** alpha \
          * (x0 * np.cos(theta) + y0 * np.sin(theta)) + x0 ** 2 + y0 ** 2
    return out


def bow_shock(r_xy, *args):
    """Bow shock model to minimize.

    Parameters
    ----------
    r_xy : array_like
        to fill

    args : [float, float]
        extra arguments.

    Returns
    -------
    out : float
        to fill

    """
    x, y = r_xy
    x0, y0 = args
    out = (x - x0) ** 2 + (y - y0) ** 2
    return out


def magnetopause_normal(r_gsm, b_z_imf, p_sw, model: str = "mp_shue1997",
                        m_alfven: float = 4.):
    """Computes the distance and normal vector to the magnetopause for
    Shue et al., 1997 or Shue et al., 1998 model. Or bow shock for
    Farris & Russell 1994 model.

    Parameters
    ----------
    r_gsm : array_like
        GSM position in Re (if more than 3 values assumes that 1st is time).

    b_z_imf : float
        IMF Bz in nT.

    p_sw : float
        Solar wind dynamic pressure in nPa.

    model : str, optional
        Name of model :
            * 'mp_shue1997' : Shue et al., 1997 (Default)
            * 'mp_shue1998' : Shue et al., 1998
            * 'bs97'        : Bow shock, Farris & Russell 1994, based on
                                'mp_shue1997'
            * 'bs98'        : Bow shock, Farris & Russell 1994, based on
                                'mp_shue1998'

    m_alfven : float, optional
        Alfvenic Mach number, only needed if bow shock model is used.

    Returns
    -------
    mindist : float
        Minimum distance to the magnetopause, in Re. Positive value if
        spacecraft is inside the magnetopause, negative if outside the
        magnetopause.

    n_vec : ndarray
        Normal vector to the magnetopause (pointing away from Earth).


    References
    ----------
    .. [1]  J.-H. Shue, J. K. Chao, H. C. Fu, C. T. Russell, P. Song,
            K. K. Khurana, and H. J. Singer (1997), A new functional form to
            study the solar wind control of the magnetopause size and shape,
            J. Geophys. Res., 102, 9497, doi: https://doi.org/10.1029/97JA00196

    .. [2]  J.-H. Shue, P. Song, C. T. Russell, J. T. Steinberg, J. K. Chao,
            G. Zastenker, O. L. Vaisber, S. Kokubun, H. J. Singer,
            T. R. Detman and H. Kawano (1998), Magnetopause location under
            extreme solar wind conditions, J. Geophys. Res., 103(A8), 17,
            691–17,700, doi: https://doi.org/10.1029/98JA01103

    .. [3]  M. H. Farris and C. T. Russell (1994), Determining the standoff
            distance of the bow shock: Mach number dependence and use of
            models, J. Geophys. Res., 99, 17681.
            doi: https://doi.org/10.1029/94JA01020

    """

    if model in ["mp_shue1998", "bs98"]:
        alpha = (0.58 - 0.007 * b_z_imf) * (1.0 + 0.024 * np.log(p_sw))
        r0 = (10.22
              + 1.29 * np.tanh(.184 * (b_z_imf + 8.14))) * p_sw ** (-1. / 6.6)
        warnings.warn('Shue et al., 1998 model used.', UserWarning)
    else:
        alpha = (0.58 - 0.01 * b_z_imf) * (1.0 + 0.01 * p_sw)

        if b_z_imf >= 0:
            r0 = (11.4 + 0.013 * b_z_imf) * p_sw ** (-1.0 / 6.6)
        else:
            r0 = (11.4 + 0.140 * b_z_imf) * p_sw ** (-1.0 / 6.6)

        warnings.warn('Shue et al., 1997 model used.', UserWarning)

    # Spacecraft position
    r1_x, r1_y, r1_z = r_gsm
    r0_x, r0_y = [r1_x, np.sqrt(r1_y ** 2 + r1_z ** 2)]

    if model[:2].lower() == "mp":
        # Magnetopause

        theta_min, min_val, ierr, numfunc = fminbound(magnetopause,
                                                      x1=-np.pi / 2,
                                                      x2=np.pi / 2, args=(
            r0, alpha, r0_x, r1_y), full_output=True)

        min_dist = np.sqrt(min_val)

        # calculate the direction to the spacecraft normal to the magnetopause
        x_n = r0 * (2 / (1 + np.cos(theta_min))) ** alpha * np.cos(theta_min) \
              - r1_x
        phi = np.arctan2(r1_z, r1_y)
        y_n = np.cos(phi) * (r0 * (2 / (1 + np.cos(theta_min))) ** alpha *
                             np.sin(theta_min)) - r1_y
        z_n = np.sin(phi) * (r0 * (2 / (1 + np.cos(theta_min))) ** alpha
                             * np.sin(theta_min)) - r1_z

        n_vec = np.stack([x_n, y_n, z_n]) / min_dist

        # if statement to ensure normal is pointing away from Earth
        if (np.sqrt(r0_x ** 2 + r0_y ** 2)
                > r0 * (2 / (1 + np.cos(theta_min))) ** alpha):
            n_vec *= -1
            min_dist *= -1

    else:
        # Bow shock
        warnings.warn('Farris & Russell 1994 bow shock model used.',
                      UserWarning)

        gamma = 5 / 3
        mach = m_alfven

        # Bow shock standoff distance
        rbs = r0 * (1 + 1.1 * ((gamma - 1) * mach ** 2 + 2)
                    / ((gamma + 1) * (mach ** 2 - 1)))

        # y ^ 2 = 0 - Ax + Bx ^ 2
        coeffs = [0, 45.3, .04]
        x = np.linspace(rbs, -100, int((rbs + 100) * 1e3) + 1)
        y = np.sqrt(-coeffs[1] * (x - rbs) + coeffs[2] * (x - rbs) ** 2)
        x = np.hstack([np.flip(x), x])
        y = np.hstack([-np.flip(y), y])

        r_xy = np.transpose(np.stack([x, y]))
        args_bow_shock = (r0_x, r0_y)

        min_val = np.min(bow_shock(r_xy, *args_bow_shock))
        min_pos = np.argmin(bow_shock(r_xy, *args_bow_shock))

        d_vec = np.stack([x - r0_x, y - r0_y])
        d_min = d_vec[min_pos, :]

        x_n = d_min[0] / np.linalg.norm(d_min)
        min_dist = np.sqrt(min_val)

        qyz = r1_y / r1_z
        z_n = np.sign(r1_z) * np.sign(x_n) \
              * np.sqrt((1 - x_n ** 2) / (1 + qyz ** 2))
        y_n = z_n * qyz

        n_vec = np.stack([x_n, y_n, z_n])

        # if statement to ensure normal is pointing away from Earth
        if n_vec[0] < 0:
            n_vec *= -1
            min_dist *= -1

    return min_dist, n_vec
