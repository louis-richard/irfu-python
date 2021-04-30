#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 - 2021 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""magnetosphere.py
@author: Louis Richard
"""

import numpy as np


def _rzero(d_p, b_z):
    return (10.22 + 1.29 * np.tanh(0.184 * (b_z + 8.14))) * d_p ** (- 1 / 6.6)


def _alpha(d_p, b_z):
    return (0.58 - 0.007 * b_z) * (1 + 0.024 * np.log(d_p))


def magnetosphere(model: str = "mp_shue1998", d_p: float = 1.7389104,
                  b_z: float = 0.012695087):
    r"""Returns the location of magnetopause.


    Parameters
    ----------
    model : str
        Model to use. Implemented: 'mp_shue1998', 'bs'.
        Default is 'mp_shue1998'.

    d_p : float
        Dynamic pressure.

    b_z : float
        IMF Bz GSM.


    Returns
    -------
    x_ : ndarray
        X location of the magnetopause.

    y_ : ndarray
        Y location of the magnetopause.


    Examples
    --------
    >>> from pyrfu.pyrf import magnetosphere

    >>> x_mp, y_mp = magnetosphere("mp_shue1998", 10, -2)

    """

    if model == "mp_shue1998":
        theta_ = np.linspace(0, np.pi, int(np.pi / .1))
        r_zero = _rzero(d_p, b_z)
        alpha_ = _alpha(d_p, b_z)

        r_ = r_zero * (2. / (1 + np.cos(theta_))) ** alpha_
        x_ = r_ * np.cos(theta_)
        y_ = r_ * np.sin(theta_)
        y_ = y_[abs(x_) < 100]
        x_ = x_[abs(x_) < 100]

    elif model == "bs":
        x_mp, _ = magnetopause("mp_shue1998", d_p, b_z)
        gamma_ = 5 / 3
        m_ = 4

        rstandoff = x_mp[0] * (1 + 1.1 * ((gamma_ - 1) * m_ ** 2 + 2) / (
                    (gamma_ + 1) * (m_ ** 2 - 1)))
        # Smaller increments at the subsolar point
        x_ = rstandoff - np.logspace(np.log10(0.1), np.log10(100 + rstandoff),
                                     300) + 0.1
        # original F/G model adds rstandoff^2=645
        y_ = np.sqrt(0.04 * (x_ - rstandoff) ** 2 - 45.3 * (x_ - rstandoff))

    else:
        raise NotImplementedError("This model is not implemented yet!!")

    return x_, y_
