#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fk_power_spectrum.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def fk_power_spectrum(pc=-1, tint=None, v6=None, b_xyz=None, z_phase=None, cav=128, field=True,
                      num_k=101, num_f=200, df=None):
    """Calculates the frequency-wave number power spectrum using MMS's four spin plane probes.
    Follows the same procedure as c_wk_powerspec. Function uses L2 scpot probe potential data to
    construct electric fields aligned with B separated by 60 m. Wavelet based cross-spectral
    analysis is used to calculate the phase difference between and the fields, and hence the wave
    number. The power is then binned according to frequency and wave number
    (Graham et al., JGR, 2016).

    Parameters
    ----------
    pc : int
        Probe combination to use :
        * 1 : ``b_xyz`` aligned with probes 1 and 2.
        * 3 : ``b_xyz`` aligned with probes 3 and 4.
        * 5 : ``b_xyz`` aligned with probes 5 and 6.

    tint : list of str
        Time interval over which the power spectrum is calculated. ``b_xyz`` should be closely
        aligned with one probe pair over this time.

    v6 : xarray.DataArray
        L2 probe potentials.

    b_xyz : xarray.DataArray
        Time series of the magnetic field in DMPA coordinates.

    z_phase : xarray.DataArray
        Spacecraft phase (zphase). Obtained from ancillary_defatt. Needed only if ``pc`` = 1 or 3.

    cav : int, optional
        Number of points in time series used to estimate phase. Default is ``cav`` = 128.

    field : bool, optional
        Set to True to use electric fields calculated from opposing probes and sc_pot. Set to
        False to use only opposing probe potentials. Default is True.

    num_k : int, optional
		Set number of wave numbers used in spectrogram. Default ``num_k`` = 101.

	df : float, optional
		Linearly spaced frequencies. Default is logarithmic spacing.

	num_f : int, optional
		Set number of frequencies used in spectrogram. Default ``num_f`` = 200.

	Returns
	-------
	fk_power : xarray.DataArray
	    Powers as a function of frequency and wavenumber. Power is normalized to the maximum value.

    See also
    --------
    pyrfu.mms.probe_align_times : Returns times when f-a electrostatic waves can be characterized.

    Notes
    -----
    Timing corrections are applied to ``v6`` in this function. Do not apply them before running
    this function. Directions and speeds are consistent with expectations based on time delays.
    Work still in progress. Time corrections for the probes need to be revised.

    Examples
    --------
    >>> from pyrfu import mms
    >>> fk_power = mms.fk_power_spectrum(pc, tint, v6, b_xyz, z_phase, cav=256, field=False)
    >>> fk_power = mms.fk_power_spectrum(pc, tint, v6, b_xyz, z_phase, cav=256, df=50, num_k=500)

    """

    # Check input
    assert isinstance(pc, int) and pc in [1, 3, 5]
    assert isinstance(tint, list) and isinstance(tint[0], str) and isinstance(tint[1], str)
    assert v6 is not None and isinstance(v6, xr.DataArray)
    assert b_xyz is not None and isinstance(b_xyz, xr.DataArray)
    assert z_phase is not None and isinstance(z_phase, xr.DataArray)
    assert isinstance(cav, int)
    assert isinstance(field, bool)
    assert isinstance(num_k, int)
    assert isinstance(num_f, int)

    return
