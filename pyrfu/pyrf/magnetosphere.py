#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any, List, Optional

# 3rd party imports
import numpy as np

# Local imports
from pyrfu.pyrf.get_omni_data import get_omni_data

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _rzero(d_p, b_z):
    return (10.22 + 1.29 * np.tanh(0.184 * (b_z + 8.14))) * d_p ** (-1 / 6.6)


def _alpha(d_p, b_z):
    return (0.58 - 0.007 * b_z) * (1 + 0.024 * np.log(d_p))


def _shue1998(d_p, b_z):
    # Magnetopause of Shue et al. (1998), in R_E
    theta_ = np.linspace(0, np.pi, int(np.pi / 0.1))
    r_zero = _rzero(d_p, b_z)
    alpha_ = _alpha(d_p, b_z)

    with np.errstate(divide="ignore"):
        r_s = r_zero * (2.0 / (1 + np.cos(theta_))) ** alpha_

    x_s = r_s * np.cos(theta_)
    y_s = r_s * np.sin(theta_)

    return x_s[abs(x_s) < 100], y_s[abs(x_s) < 100]


def magnetosphere(model: str = "mp_shue1998", tint: Optional[List[Any]] = None):
    r"""Returns the location of the magnetopause or of the bow shock.

    Parameters
    ----------
    model : str
        Model to use: 'mp_shue1998' (magnetopause, Shue et al., 1998) or
        'bs' (bow shock, Farris and Russell, 1994, from the standoff
        distance of the magnetopause). Default is 'mp_shue1998'.
    tint : list of str, Optional
        Time interval: the solar wind dynamic pressure, IMF Bz (GSM) and
        Mach number are averaged over it from OMNI. Default uses
        Dp = 1.74 nPa, Bz = 0.013 nT and M = 4 (irfu-matlab
        irf_magnetosphere uses Dp = 2 nPa, Bz = 0 nT).

    Returns
    -------
    x_ : ndarray
        X location of the magnetopause or bow shock, in R_E.
    y_ : ndarray
        Y location (distance from the X axis), in R_E.

    Raises
    ------
    NotImplementedError
        If the model is not implemented.

    Examples
    --------
    >>> from pyrfu.pyrf import magnetosphere

    Default solar wind

    >>> x_mp, y_mp = magnetosphere("mp_shue1998")

    Solar wind of a time interval, from OMNI

    >>> tint = ["2019-09-14T07:00:00", "2019-09-14T08:00:00"]
    >>> x_bs, y_bs = magnetosphere("bs", tint)

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
        x_s, y_s = _shue1998(d_p, b_z)

    elif model == "bs":
        # Standoff distance of the magnetopause, from the same solar wind (it
        # was downloaded from OMNI a second time)
        x_mp, _ = _shue1998(d_p, b_z)
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
