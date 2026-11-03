#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ._eis_units import _check_same_units
from .eis_spec_combine_sc import _idx_closest, _nearest

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def eis_pad_combine_sc(pads):
    r"""Generate composite Pitch Angle Distributions (PAD) from the EIS
    sensors across the MMS spacecraft.

    Parameters
    ----------
    pads : list of xarray.DataArray
        Pitch-angle distribution for all spacecrafts, on the same pitch angle
        bins.

    Returns
    -------
    allmms_pad_avg : xarray.DataArray
        Composite pitch angle distribution, on the times of the spacecraft with
        the fewest samples and the energies of the one with the fewest channels
        (the others at their closest channels).

    Raises
    ------
    ValueError
        If the pitch angle bins or the ``UNITS`` differ.

    Notes
    -----
    The PADs are averaged at the nearest time of each spacecraft, within half a
    time step (a spacecraft without a sample there is left out), and at the
    closest energy channel. IDL SPEDAS ``mms_eis_pad_combine_sc`` pairs the
    samples and the channels by index; here they were too, and spacecraft with
    different numbers of channels raised ValueError. The energies were the
    means of the channels of the same index.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad, pyrfu.mms.eis_spec_combine_sc

    """
    # Spacecraft with the fewest time steps (reference times) and with the
    # fewest energy channels (reference energies)
    ref_probe = pads[np.argmin([len(probe.time.data) for probe in pads])]
    ener_probe = pads[np.argmin([len(probe.energy.data) for probe in pads])]

    # The units of the reference spacecraft were taken
    _check_same_units(pads, "PADs")

    for pad_ in pads:
        if not np.array_equal(pad_.theta.data, ref_probe.theta.data):
            raise ValueError("The PADs must have the same pitch angle bins")

    t_ref = ref_probe.time.data.astype("datetime64[ns]").astype(np.int64)
    d_t = np.median(np.diff(t_ref)) if len(t_ref) > 1 else np.inf

    allmms_pad = np.full(
        (len(t_ref), len(ref_probe.theta), len(ener_probe.energy), len(pads)),
        np.nan,
    )

    # Nearest times and closest channels (they were paired by index)
    for i_pad, pad_ in enumerate(pads):
        t_pad = pad_.time.data.astype("datetime64[ns]").astype(np.int64)
        idx_time = _nearest(t_ref, t_pad, d_t / 2)
        idx_ener = _idx_closest(pad_.energy.data, ener_probe.energy.data)
        paired = idx_time >= 0
        allmms_pad[paired, ..., i_pad] = pad_.data[idx_time[paired]][..., idx_ener]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        allmms_pad_avg = np.nanmean(allmms_pad, axis=3)

    allmms_pad_avg = xr.DataArray(
        allmms_pad_avg,
        coords=[ref_probe.time.data, ref_probe.theta.data, ener_probe.energy.data],
        dims=["time", "theta", "energy"],
        attrs={**ref_probe.attrs},
    )

    return allmms_pad_avg
