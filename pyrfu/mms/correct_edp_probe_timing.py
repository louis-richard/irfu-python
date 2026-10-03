#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def correct_edp_probe_timing(sc_pot):
    r"""Corrects for the channel delays not accounted for in the MMS EDP
    processing. As described in the MMS EDP data products guide.

    Parameters
    ----------
    sc_pot : xarray.DataArray
        Time series created from L2 sc_pot files, from the variable
        "mms#_edp_dcv_brst_l2" containing individual probe potentials.

    Returns
    -------
    v_corrected : xarray.DataArray
        Time series where the channel delay for each probe have been
        accounted and corrected for.

    Notes
    -----
    This function is only useful for Burst mode data. For the other
    telemetry modes (i.e. slow and fast) the channel delays are
    completely negligible and the interpolation and resampling applied
    here will have no effect other than possibly introduce numerical
    noise.

    """

    time, pot = [sc_pot.time.data, sc_pot.data]
    ref = ts_scalar(time, pot[:, 0])

    def _shifted(data, shift_ns):
        # Time series with the time tags shifted, resampled to those of V1
        out = ts_scalar(time + np.timedelta64(shift_ns, "ns"), data)
        return resample(out, ref).data

    # Correct the time tags of V3, V5 and of E12, E34, E56 as computed in MMS
    # processing, and resample all data to time tags of V1 (i.e. timeOrig).
    v_3 = _shifted(pot[:, 2], 7629)
    v_5 = _shifted(pot[:, 4], 15259)
    e12 = _shifted((pot[:, 0] - pot[:, 1]) / 0.120, 26703)
    e34 = _shifted((pot[:, 2] - pot[:, 3]) / 0.120, 30518)
    e56 = _shifted((pot[:, 4] - pot[:, 5]) / 0.0292, 34332)

    # Recompute individual even probe potentials 2, 4, 6
    sc_pot_corrected = np.column_stack(
        [
            pot[:, 0],
            pot[:, 0] - e12 * 0.120,
            v_3,
            v_3 - e34 * 0.120,
            v_5,
            v_5 - e56 * 0.0292,
        ]
    )

    # Create the new time series with the corrected values
    sc_pot_corrected = xr.DataArray(
        sc_pot_corrected,
        coords=[time, np.arange(1, 7)],
        dims=["time", "probe"],
        attrs=dict(sc_pot.attrs),
    )

    return sc_pot_corrected
