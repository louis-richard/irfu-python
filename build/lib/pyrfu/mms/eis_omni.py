#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def eis_omni(eis_allt, method: str = "mean"):
    r"""Calculates the omni-directional flux for all 6 telescopes.

    Parameters
    ----------
    eis_allt : xarray.Dataset
        Dataset of the fluxes of all 6 telescopes.

    Returns
    -------
    flux_omni : xarray.DataArray
        Omni-directional flux for all 6 telescopes

    See Also
    --------
    pyrfu.mms.get_eis_allt

    Examples
    --------
    >>> from pyrfu import mms

    Define spacecraft index and time interval

    >>> tint = ["2017-07-23T16:10:00", "2017-07-23T18:10:00"]
    >>> ic = 2

    Get EIS ExTOF all 6 telescopes fluxes

    >>> extof_allt = mms.get_eis_allt("flux_extof_proton_srvy_l2", tint, ic)

    Compute the omni-directional flux for all 6 telescopes

    >>> extof_omni = mms.eis_omni(extof_allt)

    """

    assert method.lower() in ["mean", "sum"]

    scopes = list(filter(lambda x: x[0] == "t", eis_allt))

    flux_omni = None

    for scope in scopes:
        try:
            flux_omni += eis_allt[scope].copy()
        except TypeError:
            flux_omni = eis_allt[scope].copy()

    if method.lower() == "mean":
        flux_omni.data /= len(scopes)

    flux_omni.name = "flux_omni"
    flux_omni.attrs["energy_dplus"] = eis_allt.energy_dplus.data
    flux_omni.attrs["energy_dminus"] = eis_allt.energy_dminus.data

    return flux_omni
