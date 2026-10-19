#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants, optimize

# Local imports
from ..pyrf.resample import resample
from ..pyrf.trace import trace
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.ts_tensor_xyz import ts_tensor_xyz

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _f_one_pop(x, *args):
    # Misfit of one photoelectron population
    i_e, i_aspoc, sc_pot_r = args
    return np.nansum(np.abs(i_e + i_aspoc - x[0] * np.exp(-sc_pot_r / x[1])))


def _f_two_pop(x, *args):
    # Misfit of two photoelectron populations, the sum of both as in the
    # reconstruction of the density
    i_e, i_aspoc, sc_pot_r = args
    i_ph = x[0] * np.exp(-sc_pot_r / x[1]) + x[2] * np.exp(-sc_pot_r / x[3])
    return np.nansum(np.abs(i_e + i_aspoc - i_ph))


def scpot2ne(sc_pot, n_e, t_e, i_aspoc: xr.DataArray = None):
    r"""Compute number density from spacecraft potential. Function uses number
    density and electron temperature to compute electron thermal current. A
    best fit of the photoelectron current to the thermal current is used to
    determine the photoelectron currents and temperatures. These are then used
    to construct a new number density from the spacecraft potential.

    Parameters
    ----------
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential.
    n_e : xarray.DataArray
        Time series of the electron number density.
    t_e : xarray.DataArray
        Time series of the electron temperature. Function accepts scalar
        temperature or tensor (scalar temperature is used in either case).
    i_aspoc : xarray.DataArray, Optional
        Time series of the ASPOC current in :math:`\\mu` A.


    Returns
    -------
    n_esc : xarray.DataArray
        Time series of the number density estimated from SCpot, at the same
        resolution as ``sc_pot``.
    i_ph0 : float
        Value of the photoelectron currents ( :math:`\\mu` A) of the first
        population.
    t_ph0 : float
        Value of the temperature (eV) of the first population.
    i_ph1 : float
        Value of the photoelectron currents ( :math:`\\mu` A) of the second
        population.
    t_ph1 : float
        Value of the temperature (eV) of the second population.

    Raises
    ------
    ValueError
        If `t_e` is neither a scalar nor a tensor time series.

    Notes
    -----
    As irfu-matlab mms.scpot2ne.
    Usual assumptions are made for thermal and photoelectron current, vis.,
    planar geometry for photoelectrons and spherical geometry for thermal
    electrons.
    Currently the calculation neglects the ion thermal current, secondary
    electrons, and other sources of current.
    ASPOC on does not work very well.
    The currents are computed in double precision (FPI moments are single
    precision): the two-population fit is loosely constrained, and single
    precision moves its result, e.g. the photoelectron current by 50 %.

    """

    if i_aspoc is not None:
        i_aspoc = resample(i_aspoc, n_e).data.astype(np.float64)
    else:
        i_aspoc = np.zeros(len(n_e))

    # Scalar electron temperature, from the trace of a tensor
    t_e = t_e.astype(np.float64)

    if t_e.ndim == 3 and t_e.shape[1:] == (3, 3):
        t_e = trace(ts_tensor_xyz(t_e.time.data, t_e.data)) / 3
    elif t_e.ndim != 1:
        raise ValueError(
            "t_e must be a time series of the temperature (scalar or tensor)"
        )

    # Define constants
    m_e = constants.electron_mass
    q_e = constants.elementary_charge
    s_surf = 34

    v_eth = np.sqrt(2 * q_e * t_e.data / m_e)

    sc_pot_r = resample(sc_pot, n_e).data.astype(np.float64)

    # Thermal current in muA
    i_e = n_e.data.astype(np.float64) * v_eth * (1 + sc_pot_r / t_e.data)
    i_e *= 1e12 * q_e * s_surf / (2 * np.sqrt(np.pi))

    # First a simple fit of Iph to Ie using 1 photoelectron population
    opt_p1 = optimize.fmin(
        _f_one_pop,
        x0=[500.0, 3.0],
        args=(i_e, i_aspoc, sc_pot_r),
        maxfun=5000,
        disp=False,
    )

    # Fit of Iph to Ie for two photoelectron populations
    opt_p2 = optimize.fmin(
        _f_two_pop,
        x0=[opt_p1[0], opt_p1[1], 10.0, 10.0],
        args=(i_e, i_aspoc, sc_pot_r),
        maxfun=5000,
        disp=False,
    )
    i_ph0, t_ph0, i_ph1, t_ph1 = opt_p2

    t_e = resample(t_e, sc_pot).data
    v_sc = sc_pot.data.astype(np.float64)
    v_eth = np.sqrt(2 * q_e * t_e / m_e)
    n_esc = i_ph0 * np.exp(-v_sc / t_ph0) + i_ph1 * np.exp(-v_sc / t_ph1)
    n_esc /= s_surf * q_e * v_eth * (1 + v_sc / t_e)
    n_esc *= 2 * np.sqrt(np.pi) * 1e-12
    n_esc = ts_scalar(sc_pot.time.data, n_esc)

    return n_esc, i_ph0, t_ph0, i_ph1, t_ph1
