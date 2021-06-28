#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf import ts_vec_xyz, normalize, resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _calc_angle(look_, vec):
    vec_hat = normalize(vec)
    theta_ = np.rad2deg(np.pi - np.arccos(np.sum(vec_hat.data * look_.data,
                                                 axis=1)))
    return theta_


def eis_pad(inp_allt, vec: xr.DataArray = None, energy: list = None,
            pa_width: int = 15):
    r"""Calculates Pitch Angle Distributions (PADs) using data from the MMS
    Energetic Ion Spectrometer (EIS)

    Parameters
    ----------
    inp_allt : xarray.Dataset
        Energy spectrum for all telescopes.
    vec : xarray.DataArray, Optional
        Axis with to compute pitch-angle. Default is X-GSE.
    energy : array_like, Optional
        Energy range to include in the calculation. Default is [55, 800].
    pa_width : int, Optional
        Size of the pitch angle bins, in degrees. Default is 15.

    Returns
    -------
    pa_flux : xarray.DataArray
        Pitch-angle angle distribution for every energy channels in the
        `energy` range.

    See Also
    --------
    pyrfu.mms.get_eis_allt

    """

    time_ = inp_allt.time.data

    if vec is None:
        vec = ts_vec_xyz(time_, np.tile(np.eye(1, 3), (len(time_), 1)))
    elif isinstance(vec, list):
        assert len(vec) == 3
        vec = ts_vec_xyz(time_, np.tile(vec, (len(time_), 1)))
    elif isinstance(vec, xr.DataArray):
        vec = resample(vec, inp_allt.time)

    if energy is None:
        energy = [55, 800]

    # set up the number of pa bins to create
    n_pabins = int(180. / pa_width)
    pa_label = list(180. * np.arange(0, n_pabins) / n_pabins + pa_width / 2.)

    # Account for angular response (finite field of view) of instruments
    pa_hangw = 10.0  # deg
    delta_pa = pa_width / 2.

    scopes = list(filter(lambda x: x.startswith("t"), inp_allt))

    pa_file = np.zeros([len(time_), len(scopes)])

    e_minu = inp_allt.energy.data + inp_allt.energy_dminus.data
    e_plus = inp_allt.energy.data + inp_allt.energy_dplus.data

    cond_low = np.logical_and(e_minu >= energy[0], e_minu <= energy[1])
    cond_hig = np.logical_and(e_plus >= energy[0], e_plus <= energy[1])
    ener_idx = np.where(np.logical_or(cond_low, cond_hig))[0]

    msg = "Energy range selected is not covered by the detector"
    assert np.sum(cond_low) != 0 and np.sum(cond_hig) != 0, msg

    flux_file = np.zeros([len(time_), len(scopes), len(ener_idx)])
    shape_ = [len(time_), n_pabins, len(ener_idx)]
    pa_flux, pa_num_in_bin = [np.zeros(shape_) for _ in range(2)]

    pa_flux[...] = np.nan

    for t, scope in enumerate(scopes):
        # get pa from each detector
        pa_file[:, t] = _calc_angle(inp_allt[f"look_{scope}"], vec)

        # get energy range of interest
        flux_file[:, t, :] = inp_allt[scope][:, ener_idx]

    flux_file[flux_file == 0] = np.nan

    for (i, t_), (j, pa_lbl) in itertools.product(enumerate(time_),
                                                  enumerate(pa_label)):
        cond_ = np.logical_and(pa_file[i, :] + pa_hangw >= pa_lbl - delta_pa,
                               pa_file[i, :] - pa_hangw < pa_lbl + delta_pa)
        ind = np.where(cond_)[0]
        if ind.size != 0:
            pa_flux[i, j, :] = np.nanmean(flux_file[i, ind, :], axis=0)

    pa_flux = xr.DataArray(pa_flux,
                           coords=[time_, pa_label,
                                   inp_allt.energy.data[ener_idx]],
                           dims=["time", "theta", "energy"])

    return pa_flux
