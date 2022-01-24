#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _idx_closest(lst0, lst1):
    return [(np.abs(np.asarray(lst0) - k)).argmin() for k in lst1]


def eis_spec_combine_sc(omni_vars, method: str = "mean"):
    r"""Combines omni-directional energy spectrogram variable from EIS on
    multiple MMS spacecraft.

    Parameters
    ----------
    omni_vars : list of xarray.DataArray
        Omni-directional energy spectrograms of all spacecraft.
    method : str, Optional
        Method to combine spectra, "mean" or "sum"

    Returns
    -------
    omni_spec : xarray.DataArray
        Combined omni-directional energy spectrogram.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_omni, pyrfu.mms.eis_pad_combine_sc

    Examples
    --------
    >>> from pyrfu.mms import get_eis_allt, eis_omni, eis_spec_combine_sc

    Define time interval

    >>> tint = ["2017-07-23T16:10:00", "2017-07-23T18:10:00"]

    Load EIS ExTOF flux spectrograms for all 6 telescopes for all spacecraft

    >>> extof_allt_mms = []
    >>> for ic in range(2, 5):
    ...     extof_allt_mms.append(get_eis_allt("flux_extof_proton_srvy_l2",
    ...                             tint, ic))

    Compute the omni-direction flux spectrogram for all spacecraft

    >>> extof_omni_mms = []
    >>> for extof_allt in extof_allt_mms:
    ...     extof_omni_mms.append(eis_omni(extof_allt))

    Combine spectrograms of all spacecraft

    >>> extof_omni_mmsx = eis_spec_combine_sc(extof_omni_mms)

    """
    
    assert method.lower() in ["mean", "sum"]

    reftime_sc_loc = np.argmin([len(x_.time.data) for x_ in omni_vars])
    refener_sc_loc = np.argmin([len(x_.energy.data) for x_ in omni_vars])

    time_refprobe = omni_vars[reftime_sc_loc]
    ener_refprobe = omni_vars[refener_sc_loc]

    nt_ref, ne_ref = [time_refprobe.shape[0], ener_refprobe.shape[1]]

    # time x energy x spacecraft
    omni_spec_data = np.empty([nt_ref, ne_ref, len(omni_vars)])
    omni_spec_data[:] = np.nan
    # time x energy
    omni_spec = np.empty([nt_ref, ne_ref])
    omni_spec[:] = np.nan

    # energy_data = np.zeros([ne_ref, len(omni_vars)])

    # Average omni flux over all spacecraft and define common energy grid
    for pp, flux_ in enumerate(omni_vars):
        idx_closest = _idx_closest(flux_.energy.data,
                                   ener_refprobe.energy.data)
        # energy_data[:, pp] = flux_.energy.data[0:ne_ref]
        omni_spec_data[0:nt_ref, :, pp] = flux_.data[0:nt_ref, idx_closest]

    # Average omni flux over all spacecraft
    for tt, ee in itertools.product(range(nt_ref), range(ne_ref)):
        if method.lower() == "mean":
            omni_spec[tt, ee] = np.nanmean(omni_spec_data[tt, ee, :], axis=0)
        else:
            omni_spec[tt, ee] = np.nansum(omni_spec_data[tt, ee, :], axis=0)

    omni_spec = xr.DataArray(omni_spec,
                             coords=[time_refprobe.time.data,
                                     ener_refprobe.energy.data],
                             dims=["time", "energy"])
    omni_spec.attrs["energy_dplus"] = ener_refprobe.attrs["energy_dplus"]
    omni_spec.attrs["energy_dminus"] = ener_refprobe.attrs["energy_dminus"]

    return omni_spec
