#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Any, List, Optional, Tuple

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# Local imports
from pyrfu.pyrf.get_omni_data import get_omni_data

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Default solar wind: dynamic pressure (nPa), IMF Bz GSM (nT), Mach number
_DEFAULTS = {"d_p": 1.7389104, "b_z": 0.012695087, "m_a": 4.0}


def _rzero(d_p, b_z):
    return (10.22 + 1.29 * np.tanh(0.184 * (b_z + 8.14))) * d_p ** (-1 / 6.6)


def _alpha(d_p, b_z):
    return (0.58 - 0.007 * b_z) * (1 + 0.024 * np.log(d_p))


def _shue1998(d_p, b_z):
    # Magnetopause of Shue et al. (1998), in R_E, at theta = 0, 0.1, ..., 3.1
    # rad as irfu-matlab (0:0.1:pi; it was linspace(0, pi, 31))
    theta_ = np.arange(0.0, np.pi, 0.1)
    r_zero = _rzero(d_p, b_z)
    alpha_ = _alpha(d_p, b_z)

    with np.errstate(divide="ignore"):
        r_s = r_zero * (2.0 / (1 + np.cos(theta_))) ** alpha_

    x_s = r_s * np.cos(theta_)
    y_s = r_s * np.sin(theta_)

    return x_s[abs(x_s) < 100], y_s[abs(x_s) < 100]


def magnetosphere(
    model: str = "mp_shue1998",
    tint: Optional[List[Any]] = None,
    d_p: Optional[float] = None,
    b_z: Optional[float] = None,
    m_a: Optional[float] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Returns the location of the magnetopause or of the bow shock.

    Parameters
    ----------
    model : str
        Model to use: 'mp_shue1998' (magnetopause, Shue et al., 1998) or
        'bs' (bow shock, Farris and Russell, 1994, from the standoff
        distance of the magnetopause), case-insensitive. Default is
        'mp_shue1998'.
    tint : list of str, Optional
        Time interval: the solar wind dynamic pressure, IMF Bz (GSM) and
        Mach number are averaged over it from OMNI (hourly). Not with
        d_p, b_z or m_a.
    d_p : float, Optional
        Solar wind dynamic pressure in nPa, as irfu-matlab
        irf_magnetosphere(model, Dp, Bz). Default is 1.74 nPa (irfu-matlab:
        2 nPa).
    b_z : float, Optional
        IMF Bz (GSM) in nT. Default is 0.013 nT (irfu-matlab: 0 nT).
    m_a : float, Optional
        Mach number of the solar wind, for the bow shock. Default is 4, as
        irfu-matlab.

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
    ValueError
        If tint is given with d_p, b_z or m_a, if d_p is not positive or b_z
        not finite (e.g., no OMNI data in tint), or if m_a is not larger than
        1 for the bow shock.

    Notes
    -----
    The magnetopause is sampled at the angles 0, 0.1, ..., 3.1 rad from the
    X axis, as irfu-matlab, up to abs(x) = 100 R_E. With a time,
    irf_magnetosphere interpolates the 1-minute OMNI data at that time;
    here the hourly OMNI data are averaged over tint.

    Examples
    --------
    >>> from pyrfu.pyrf import magnetosphere

    Default solar wind

    >>> x_mp, y_mp = magnetosphere("mp_shue1998")

    Given solar wind, as irf_magnetosphere("mp_shue1998", 10, -2)

    >>> x_mp, y_mp = magnetosphere("mp_shue1998", d_p=10.0, b_z=-2.0)

    Solar wind of a time interval, from OMNI

    >>> tint = ["2019-09-14T07:00:00", "2019-09-14T08:00:00"]
    >>> x_bs, y_bs = magnetosphere("bs", tint)

    """

    solar_wind = {"d_p": d_p, "b_z": b_z, "m_a": m_a}

    if tint is not None:
        if any(value is not None for value in solar_wind.values()):
            raise ValueError("give either tint (OMNI) or d_p, b_z and m_a")

        omni_data = get_omni_data(["p", "bzgsm", "ma"], tint)
        b_z = float(omni_data.bzgsm.mean("time").data)
        d_p = float(omni_data.p.mean("time").data)
        m_a = float(omni_data.ma.mean("time").data)
    else:
        d_p, b_z, m_a = [
            _DEFAULTS[key] if value is None else float(value)
            for key, value in solar_wind.items()
        ]

    # NaN, e.g. without OMNI data in tint
    if not np.isfinite(d_p) or not np.isfinite(b_z) or d_p <= 0:
        raise ValueError(
            f"the dynamic pressure must be positive and the IMF Bz finite, not "
            f"{d_p} nPa and {b_z} nT"
        )

    model = model.lower()

    if model == "mp_shue1998":
        x_s, y_s = _shue1998(d_p, b_z)

    elif model == "bs":
        if not np.isfinite(m_a) or m_a <= 1:
            raise ValueError(f"the Mach number must be larger than 1, not {m_a}")

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
