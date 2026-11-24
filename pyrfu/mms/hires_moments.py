#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from .psd_moments import psd_moments


def hires_moments(vdf_hires, sc_pot, energy_range=(1.0, 4e4), **kwargs):
    r"""Moments of the parts of the distributions of `vdf_hires`, each
    multiplied by the number of parts.

    Parameters
    ----------
    vdf_hires : xarray.Dataset
        Skymap of the parts of the distributions, from `vdf_hires`.
    sc_pot : xarray.DataArray
        Time series of the spacecraft potential, as for `psd_moments`.
    energy_range : array_like, Optional
        Energy range (eV) of the channels integrated over. Default is 1 eV to
        40 keV.
    **kwargs
        Other options of `psd_moments`.

    Returns
    -------
    out : tuple
        The moments of `psd_moments`, at the times of the parts.

    Raises
    ------
    ValueError
        If `vdf_hires` has no attribute ``n_parts`` (not from `vdf_hires`).

    Notes
    -----
    Each part has the bins measured in its fraction 1 / n of the
    accumulation, the others are NaN (no data, they add nothing). Its
    moments are those of these bins times n: the measured directions are
    taken to hold 1 / n of the particles of each energy. This holds when
    each part samples 1 / n of the solid angle evenly, which is the case of
    the FPI burst distributions checked (see `vdf_hires`), apart from the
    highest DES channel; the moments of a part then differ from those of the
    whole distribution by the anisotropy at the scale of the sampling as
    well as by the time variation. For DES burst distributions of 2019-01-01
    00:18:30 (MMS1, 5 cm^-3), the densities of the four quarters are 0.97 to
    1.05 times that of the whole distribution, and their mean equals it.

    See Also
    --------
    pyrfu.mms.vdf_hires, pyrfu.mms.psd_moments

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2019-01-01T00:18:30.000", "2019-01-01T00:18:31.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> off_e = mms.get_data("pdtoffe_fpi_brst_l2", tint, 1)
    >>> sc_pot = mms.get_data("v_edp_brst_l2", tint, 1)
    >>> moments = mms.hires_moments(mms.vdf_hires(vdf_e, off_e), sc_pot)

    """

    if "n_parts" not in vdf_hires.attrs:
        raise ValueError("vdf_hires must come from vdf_hires (attribute n_parts)")

    vdf = vdf_hires.copy()
    vdf["data"] = vdf_hires.data.copy(
        data=vdf_hires.data.data * vdf_hires.attrs["n_parts"]
    )

    return psd_moments(vdf, sc_pot, energy_range=energy_range, **kwargs)
