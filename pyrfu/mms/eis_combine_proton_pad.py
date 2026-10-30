#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

# Local imports
from .eis_combine_proton_spec import eis_combine_proton_spec
from .eis_pad import eis_pad
from .eis_pad_spinavg import eis_pad_spinavg

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def eis_combine_proton_pad(
    phxtof_allt,
    extof_allt,
    vec: xr.DataArray = None,
    energy: list = None,
    pa_width: int = 15,
    despin: bool = False,
):
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
    vec : xarray.DataArray or array_like
        Magnetic field in GSE (see :func:`pyrfu.mms.eis_pad`).
    energy : array_like, Optional
        Energy range to include in the calculation (see
        :func:`pyrfu.mms.eis_pad`). Default is all the combined channels.
    pa_width : int, Optional
        Size of the pitch angle bins, in degrees. Default is 15.
    despin : bool, Optional
        Spin average (:func:`pyrfu.mms.eis_pad_spinavg`). Default is False.

    Returns
    -------
    out : xarray.DataArray
        Combined PHxTOF and ExTOF proton PADs.

    Raises
    ------
    ValueError
        As :func:`pyrfu.mms.eis_combine_proton_spec`,
        :func:`pyrfu.mms.eis_pad` and :func:`pyrfu.mms.eis_pad_spinavg`.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad, pyrfu.mms.eis_omni,
    pyrfu.mms.eis_combine_proton_spec

    Notes
    -----
    The PADs are those of the combined spectra,
    ``eis_pad(eis_combine_proton_spec(phxtof_allt, extof_allt), ...)``: one
    channel table and its energies as labels (the columns were stitched from
    three channel selections: they could be mislabelled, and the default
    arguments could raise). The channels are those of
    :func:`pyrfu.mms.eis_combine_proton_spec`, as IDL SPEDAS
    ``mms_eis_combine_proton_spec``: the lowest ExTOF channel is left out and
    the PHxTOF and ExTOF channels are kept separately. IDL SPEDAS
    ``mms_eis_combine_proton_pad`` instead averages PHxTOF and ExTOF channels
    around 50 to 80 keV, selected with fixed 10, 52 and 82 keV limits: it
    re-uses the lowest ExTOF channel and labels the means with the arithmetic
    mean of the two energies (e.g., 75.4 keV for 70.6 and 80.2 keV). The other
    channels are the same.

    """

    # Combined PHxTOF and ExTOF spectra, with the ExTOF look directions and
    # spin numbers
    proton_allt = eis_combine_proton_spec(phxtof_allt, extof_allt)

    out = eis_pad(proton_allt, vec, energy, pa_width)

    if despin:
        out = eis_pad_spinavg(out, proton_allt.spin)

    return out
