#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf import ts_skymap

from .dpf2psd import dpf2psd

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def eis_skymap(inp_ang_ang, to_psd: bool = True):
    r"""Construct skymap distribution from angle-angle distribution.

    Parameters
    ----------
    inp_ang_ang : xarray.DataArray
        EIS angle-angle distribution.
    to_psd : bool, Optional
        Flag to convert differential particle flux to phase space density.

    Returns
    -------
    out : xarray.Dataset
        EIS skymap distribution.

    See Also
    --------
    pyrfu.mms.eis_ang_ang

    """

    time_data = inp_ang_ang.time.data
    enr_ = np.tile(inp_ang_ang.energy.data, (len(time_data), 1))
    phi_ = np.tile(inp_ang_ang.phi.data, (len(time_data), 1))
    the_ = inp_ang_ang.theta.data

    out = ts_skymap(time_data, inp_ang_ang.data, enr_, phi_, the_)
    out.attrs["species"] = "ions"
    out.attrs["UNITS"] = "1/(cm^2 s sr keV)"

    out.energy.data *= 1e3

    if to_psd:
        out = dpf2psd(out)

    out.attrs["energy_dminus"] = inp_ang_ang.attrs["energy_dminus"]
    out.attrs["energy_dplus"] = inp_ang_ang.attrs["energy_dplus"]

    return out
