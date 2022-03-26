#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import constants, optimize

# Local imports
from ..pyrf import trace, ts_tensor_xyz, ts_scalar, resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _f_one_pop(x, *args):
    i_e, i_aspoc, sc_pot_r = args
    return np.nansum(np.abs(i_e.data + i_aspoc.data
                            - (x[0] * np.exp(-sc_pot_r.data / x[1]))))


def _f_two_pop(x, *args):
    i_e, i_aspoc, sc_pot_r = args
    return np.nansum(np.abs(i_e.data + i_aspoc.data
                            - (x[0] * np.exp(-sc_pot_r.data / x[1]))
                            + (x[2] * np.exp(-sc_pot_r.data / x[3]))))


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

    Notes
    -----
    Usual assumptions are made for thermal and photoelectron current, vis.,
    planar geometry for photoelectrons and spherical geometry for thermal
    electrons.
    Currently the calculation neglects the ion thermal current, secondary
    electrons, and other sources of current.
    ASPOC on does not work very well.

    """

    if i_aspoc is not None:
        i_aspoc = resample(i_aspoc, n_e)
    else:
        i_aspoc = ts_scalar(n_e.time.data, np.zeros(len(n_e)))

    # Check format of electron temperature
    if t_e.ndim == 3 and t_e.shape[1:] == (3, 3):
        t_e = trace(ts_tensor_xyz(t_e.time.data, t_e.data)) / 3
    else:
        raise IndexError("Te format not recognized")

    # Define constants
    m_e = constants.electron_mass
    q_e = constants.elementary_charge
    s_surf = 34

    v_eth = np.sqrt(2 * q_e * t_e.data / m_e)

    sc_pot_r = resample(sc_pot, n_e)

    # Thermal current in muA
    i_e = n_e.data * v_eth * (1 + sc_pot_r.data / t_e.data)
    i_e *= 1e12 * q_e * s_surf / (2 * np.sqrt(np.pi))

    # First a simple fit of Iph to Ie using 1 photoelectron population
    opt_p1 = optimize.fmin(_f_one_pop, x0=[500., 3.],
                           args=(i_e, i_aspoc, sc_pot_r), maxfun=5000)

    # Fit of Iph to Ie for two photoelectron populations
    opt_p2 = optimize.fmin(_f_two_pop, x0=[opt_p1[0], opt_p1[1], 10., 10.],
                           args=(i_e, i_aspoc, sc_pot_r), maxfun=5000)
    i_ph0, t_ph0, i_ph1, t_ph1 = opt_p2

    v_eth = np.sqrt(2 * q_e * resample(t_e, sc_pot).data / m_e)
    n_esc = i_ph0 * np.exp(-sc_pot.data / t_ph0) + i_ph1 * np.exp(
        -sc_pot.data / t_ph1)
    n_esc /= s_surf * q_e * v_eth * (
                1 + sc_pot.data / resample(t_e, sc_pot).data)
    n_esc *= 2 * np.sqrt(np.pi) * 1e-12
    n_esc = ts_scalar(sc_pot.time.data, n_esc)

    return n_esc, i_ph0, t_ph0, i_ph1, t_ph1
