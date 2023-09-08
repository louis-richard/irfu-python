#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
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

    flux_omni = np.zeros_like(eis_allt[scopes[0]].data)

    for scope in scopes:
        flux_omni += eis_allt[scope].data.copy()

        # Why??
        # try:
        #     flux_omni += eis_allt[scope].data.copy()
        # except TypeError:
        #     flux_omni = eis_allt[scope].data.copy()

    if method.lower() == "mean":
        flux_omni /= len(scopes)

    # Get dimensions, coordinates and attributes based on first telescope
    dims = eis_allt[scopes[0]].dims
    coords = [eis_allt[scopes[0]][k] for k in dims]
    attrs = eis_allt[scopes[0]].attrs

    flux_omni = xr.DataArray(flux_omni, coords=coords, dims=dims, attrs=attrs)

    return flux_omni
