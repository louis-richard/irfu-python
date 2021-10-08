#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from .eis_combine_proton_spec import eis_combine_proton_spec
from .eis_ang_ang import eis_ang_ang
from .eis_skymap import eis_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def eis_combine_proton_skymap(phxtof_allt, extof_allt, en_chan: list = None,
                              to_psd: bool = True):
    r"""Combines ExTOF and PHxTOF proton energy spectra and generate proton
    skymap distribution.

    Parameters
    ----------
    phxtof_allt : xarray.Dataset
        Dataset containing the PHxTOF energy spectrum of the 6 telescopes.
    extof_allt : xarray.Dataset
        Dataset containing the ExTOF energy spectrum of the 6 telescopes.
    en_chan : array_like, Optional
        Energy channels to use. Default use all energy channels.
    to_psd : bool, Optional
        Flag to convert differential particle flux to phase space density.

    Returns
    -------
    eis_skymap : xarray.Dataset
        EIS skymap distribution

    """

    # Combine Pulse-Height x Time Of Flight and Energy x Time Of Flight spectra
    eis_allt = eis_combine_proton_spec(phxtof_allt, extof_allt)

    # Compute EIS angle-angle (azimuth-elevation) distribution.
    eis_ang_ = eis_ang_ang(eis_allt, en_chan=en_chan)

    # Convert to skymap format
    eis_skymap_ = eis_skymap(eis_ang_, to_psd=to_psd)

    return eis_skymap_
