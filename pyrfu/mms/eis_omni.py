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
    method : {"mean", "sum"}, Optional
        Method over the telescopes. Use "sum" for counts (the total counts,
        for their statistics) and "mean" for everything else, as
        :func:`pyrfu.mms.vdf_omni`. Default is "mean".

    Returns
    -------
    flux_omni : xarray.DataArray
        Omni-directional flux for all 6 telescopes, with the spin numbers and
        sectors of the samples (``spin`` and ``sector`` coordinates, if
        `eis_allt` has them on the same times) and the energy widths and
        species of `eis_allt` in its attributes (``delta_energy_plus``,
        ``delta_energy_minus``, ``species``).

    Raises
    ------
    ValueError
        If method is not "mean" or "sum".

    Notes
    -----
    The omni-directional flux is the plain mean (or sum) of the telescopes, as
    IDL SPEDAS ``mms_eis_omni`` and pySPEDAS: a NaN in one telescope gives NaN,
    and zeros count.

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

    if method.lower() not in ["mean", "sum"]:
        raise ValueError(f"method must be mean or sum, not {method}")

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
    attrs = {**eis_allt[scopes[0]].attrs}

    # Energy widths and species, attributes of the Dataset (they were left
    # out, on which eis_spec_combine_sc raised KeyError)
    for key in ["delta_energy_plus", "delta_energy_minus", "species"]:
        if key in eis_allt.attrs:
            attrs[key] = eis_allt.attrs[key]

    flux_omni = xr.DataArray(flux_omni, coords=coords, dims=dims, attrs=attrs)

    # Spin numbers and sectors of the samples, to pair the spacecraft in
    # eis_spec_combine_sc (the sector gives the look direction)
    for key in ["spin", "sector"]:
        if key in eis_allt and np.array_equal(
            eis_allt[key].time.data, flux_omni.time.data
        ):
            flux_omni = flux_omni.assign_coords({key: ("time", eis_allt[key].data)})

    return flux_omni
