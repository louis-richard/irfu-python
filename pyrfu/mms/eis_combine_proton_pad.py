#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .eis_pad import eis_pad
from .eis_omni import eis_omni
from .eis_combine_proton_spec import eis_combine_proton_spec

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _despin(inp, spin_nums):
    spin_starts = np.where(spin_nums[1:] > spin_nums[:-1])[0]
    time_rec = inp.time.data[spin_starts]

    pad_ds = np.zeros([len(spin_starts), len(inp.theta), len(inp.energy)])

    c_strt = 0

    for i, spin_strt in enumerate(spin_starts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pad_ds[i, :, :] = np.nanmean(inp.data[c_strt:spin_strt + 1, :, :],
                                         axis=0)
        c_strt = spin_strt + 1

    out = xr.DataArray(pad_ds,
                       coords=[time_rec, inp.theta.data, inp.energy.data],
                       dims=["time", "theta", "energy"])

    return out


def eis_combine_proton_pad(phxtof_allt, extof_allt, vec: xr.DataArray = None,
                           energy: list = None, pa_width: int = 15,
                           despin: bool = False):
    r"""Combines EPD-EIS Energy by Time Of Flight (ExTOF) and Pulse Height by
    Time Of Flight (PHxTOF) proton Pitch-Angle Distributions (PADs).

    Parameters
    ----------
    phxtof_allt : xarray.Dataset
        Dataset containing the energy spectrum of the 6 telescopes of the
        Energy Ion Spectrometer (EIS) for MCP Pulse Height by Time Of Flight
        (PHxTOF).
    extof_allt : xarray.Dataset
        Dataset containing the energy spectrum of the 6 telescopes of the
        Energy Ion Spectrometer (EIS) for Energy by Time Of Flight (ExTOF).
    vec : xarray.DataArray, Optional
        Axis with to compute pitch-angle. Default is X-GSE.
    energy : array_like, Optional
        Energy range to include in the calculation. Default is [55, 800].
    pa_width : int, Optional
        Size of the pitch angle bins, in degrees. Default is 15.
    despin : bool, Optional
        Remove spintone. Default is False.

    Returns
    -------
    out : xarray.DataArray
        Combined PHxTOF and ExTOF proton PADs.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad, pyrfu.mms.eis_omni,
    pyrfu.mms.eis_combine_proton_spec


    """

    if energy is None:
        energy = [55, 800]

    # set up the number of pa bins to create
    n_pabins = int(180. / pa_width)
    pa_label = 180. * np.arange(n_pabins) / n_pabins + pa_width / 2.

    # Compute Pulse-Height x Time Of Flight (PHxTOF) Pitch-Angle
    # Distribution (PAD)
    phxtof_pad = eis_pad(phxtof_allt, vec, energy, pa_width)

    # Compute Energy x Time Of Flight (ExTOF) Pitch-Angle Distribution (PAD)
    extof_pad = eis_pad(extof_allt, vec, energy, pa_width)

    # Compute combined PHxTOF and ExTOF omni-directional energy spectrum.
    proton_combined_spec = eis_omni(eis_combine_proton_spec(phxtof_allt,
                                                            extof_allt))

    data_size = [len(phxtof_pad), len(extof_pad)]

    if data_size[0] == data_size[1]:
        time_data = phxtof_pad.time.data
        phxtof_pad_data = phxtof_pad.data
        extof_pad_data = extof_pad.data
    elif data_size[0] > data_size[1]:
        time_data = extof_pad.time.data
        phxtof_pad_data = phxtof_pad.data[:data_size[1], ...]
        extof_pad_data = extof_pad.data
    elif data_size[0] < data_size[1]:
        time_data = phxtof_pad.time.data
        phxtof_pad_data = phxtof_pad.data
        extof_pad_data = extof_pad.data[:data_size[0], ...]
    else:
        raise ValueError

    cond_ = np.logical_and(proton_combined_spec.energy.data > energy[0],
                           proton_combined_spec.energy.data < energy[1])
    energy_size = np.where(cond_)[0]

    proton_pad = np.zeros((len(time_data), n_pabins, len(energy_size)))

    cond_phxtof = np.logical_and(phxtof_pad.energy.data > 14,
                                 phxtof_pad.energy.data < 52,
                                 phxtof_pad.energy.data > energy[0])
    cond_extof = np.logical_and(extof_pad.energy.data > 82,
                                extof_pad.energy.data < energy[1])

    phxtof_taren = np.where(cond_phxtof)[0]
    phxtof_taren_cross = np.where(phxtof_pad.energy.data > 52)[0]
    extof_taren_cross = np.where(extof_pad.energy.data < 82)[0]
    extof_taren = np.where(cond_extof)[0]

    n_pe, n_pce, n_ece, n_ee = [phxtof_taren.size, phxtof_taren_cross.size,
                                extof_taren_cross.size, extof_taren.size]

    proton_pad[..., :n_pe] = phxtof_pad_data[..., phxtof_taren]

    for (i, phxtof_en), extof_en in zip(enumerate(phxtof_taren_cross),
                                        extof_taren_cross):
        temp_ = np.stack([phxtof_pad_data[..., phxtof_en],
                          extof_pad_data[..., extof_en]])
        r_, l_ = [n_pe, n_pe + i + 1]
        proton_pad[..., r_:l_] = np.nanmean(temp_, axis=0)[:, :, None]

    r_ = len(phxtof_pad.energy.data)
    l_ = len(phxtof_pad.energy.data) + len(extof_taren)
    proton_pad[..., r_:l_] = extof_pad_data[..., extof_taren]

    """
    energy = np.hstack([phxtof_pad.energy.data[phxtof_taren],
                        (phxtof_pad.energy.data[phxtof_taren_cross]
                         + extof_pad.energy.data[extof_taren_cross]) / 2,
                        extof_pad.energy.data[extof_taren]])
    """

    energy = proton_combined_spec.energy.data[cond_]

    out = xr.DataArray(proton_pad, coords=[time_data, pa_label, energy],
                       dims=["time", "theta", "energy"])

    if despin:
        out = _despin(out, phxtof_allt.spin.data)

    return out
