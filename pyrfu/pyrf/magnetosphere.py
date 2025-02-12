#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any, List, Optional

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.pyrf.get_omni_data import get_omni_data

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _rzero(d_p, b_z):
    return (10.22 + 1.29 * np.tanh(0.184 * (b_z + 8.14))) * d_p ** (-1 / 6.6)


def _alpha(d_p, b_z):
    return (0.58 - 0.007 * b_z) * (1 + 0.024 * np.log(d_p))


def magnetosphere(model: str = "mp_shue1998", tint: Optional[List[Any]] = None):
    r"""Returns the location of magnetopause.

    Parameters
    ----------
    model : str
        Model to use. Implemented: 'mp_shue1998', 'bs'.
        Default is 'mp_shue1998'.
    tint : list
        Time interval.

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

    if tint is None:
        d_p = 1.7389104
        b_z = 0.012695087
        m_a = 4.0
    else:
        omni_data = get_omni_data(["p", "bzgsm", "ma"], tint)
        b_z = float(omni_data.bzgsm.mean("time").data)
        d_p = float(omni_data.p.mean("time").data)
        m_a = float(omni_data.ma.mean("time").data)

    if model == "mp_shue1998":
        theta_ = np.linspace(0, np.pi, int(np.pi / 0.1))
        r_zero = _rzero(d_p, b_z)
        alpha_ = _alpha(d_p, b_z)

        with np.errstate(divide="ignore"):
            r_s = r_zero * (2.0 / (1 + np.cos(theta_))) ** alpha_

        x_s = r_s * np.cos(theta_)
        y_s = r_s * np.sin(theta_)
        y_s = y_s[abs(x_s) < 100]
        x_s = x_s[abs(x_s) < 100]

    elif model == "bs":
        x_mp, _ = magnetosphere("mp_shue1998", tint)
        gamma_ = 5 / 3

        rstandoff = x_mp[0] * (
            1 + 1.1 * ((gamma_ - 1) * m_a**2 + 2) / ((gamma_ + 1) * (m_a**2 - 1))
        )
        # Smaller increments at the subsolar point
        x_s = (
            rstandoff - np.logspace(np.log10(0.1), np.log10(100 + rstandoff), 300) + 0.1
        )
        # original F/G model adds rstandoff^2=645
        y_s = np.sqrt(
            0.04 * (x_s - rstandoff) ** 2 - 45.3 * (x_s - rstandoff),
        )

    else:
        raise NotImplementedError("This model is not implemented yet!!")

    return x_s, y_s
