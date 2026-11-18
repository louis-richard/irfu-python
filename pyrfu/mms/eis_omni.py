#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import re

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.mms._internal.epd import _common_attrs


def _omni_attrs(telescope_attrs, energy_dim, method):
    r"""Attributes of the omni-directional flux: those with the same value in
    all the telescopes (the attributes of the telescope 0 labelled it: name,
    pointers to its support variables, display maximum), and for the omni
    flux its name (``_t0`` → ``_omni``, as SPEDAS), DEPEND_1 the name of its
    energy coordinate, and SCALEMAX the telescopes' maxima combined as the
    data (an upper bound of the omni maximum of the file)."""
    attrs = _common_attrs(telescope_attrs)
    first = telescope_attrs[0]

    if isinstance(first.get("FIELDNAM"), str) and re.search(
        r"_t\d$", first["FIELDNAM"]
    ):
        attrs["FIELDNAM"] = re.sub(r"_t\d$", "_omni", first["FIELDNAM"])

    if "DEPEND_1" in first:
        attrs["DEPEND_1"] = energy_dim

    try:
        scale_max = np.stack(
            [
                np.asarray(attrs_["SCALEMAX"], dtype=np.float64)
                for attrs_ in telescope_attrs
            ]
        )
    except (KeyError, TypeError, ValueError):
        pass
    else:
        if method == "sum":
            attrs["SCALEMAX"] = scale_max.sum(axis=0)
        else:
            attrs["SCALEMAX"] = scale_max.mean(axis=0)

    return attrs


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
        ``delta_energy_minus``, ``species``). Its attributes are those common
        to the telescopes, with FIELDNAM the name of the omni flux (``_omni``
        instead of ``_t0``), DEPEND_1 the name of its energy coordinate and
        SCALEMAX the telescopes' SCALEMAX combined as the data (an upper
        bound of the maximum of the omni flux over the file).

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

    if method.lower() == "mean":
        flux_omni /= len(scopes)

    # Dimensions and coordinates of the first telescope (the energies are the
    # same for all, one coordinate in eis_allt), the attributes common to the
    # telescopes
    dims = eis_allt[scopes[0]].dims
    coords = [eis_allt[scopes[0]][k] for k in dims]
    attrs = _omni_attrs(
        [eis_allt[scope].attrs for scope in scopes], dims[1], method.lower()
    )

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
