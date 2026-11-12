#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ._vdf_bins import _charge_sign
from .vdf_convert import _DEF, vdf_convert


def remove_idist_background(vdf, def_bg):
    r"""Remove the mode background population due to penetrating radiation
    `def_bg` from the ion velocity distribution function `vdf` using the
    method from [1]_.

    Parameters
    ----------
    vdf : xarray.Dataset
        Ion velocity distribution function, in any units of `vdf_convert`
        (``UNITS``, s^3/cm^6 if missing).
    def_bg : xarray.DataArray
        Omni-directional ion differential energy flux of the background, at
        the times of `vdf` (e.g. ``defbgi_fpi_brst_l2``), in
        keV/(cm^2 s sr keV) (``UNITS``, the default) or eV/(cm^2 s sr eV).

    Returns
    -------
    vdf_new : xarray.Dataset
        Ion velocity distribution function cleaned, in the units of `vdf`.

    Raises
    ------
    ValueError
        If `vdf` is an electron distribution (see `remove_edist_background`),
        `def_bg` is not at the times of `vdf` or is not a differential energy
        flux, or the units of `vdf` are not supported.

    Notes
    -----
    The background flux is taken in all the directions and converted to the
    units of `vdf` at its energies with the mass of its species
    (`vdf_convert`): it was converted to s^3/cm^6 with the proton mass
    whatever the units of `vdf` (nothing was removed from a distribution in
    s^3/m^6), and used by position. As irfu-matlab
    mms.remove_ion_penetrating_radiation_bg, the values that become negative
    are set to 0.

    References
    ----------
    .. [1]  Gershman, D. J., Dorelli, J. C., Avanov,L. A., Gliese, U.,
            Barrie, A., Schiff, C.,et al. (2019). Systematic uncertainties
            in plasma parameters reported by the fast plasma investigation on
            NASA's magnetospheric multiscale mission. Journal of Geophysical
            Research: Space Physics, 124, https://doi.org/10.1029/2019JA026980

    """
    if _charge_sign(vdf) < 0:
        raise ValueError(
            "vdf must be an ion distribution (electrons: remove_edist_background)"
        )

    if not np.array_equal(def_bg.time.data, vdf.time.data):
        raise ValueError("def_bg must be at the times of vdf")

    bg_units = str(def_bg.attrs.get("UNITS", "keV/(cm^2 s sr keV)"))

    if bg_units.lower() not in _DEF:
        raise ValueError(f"def_bg must be a differential energy flux, not {bg_units}")

    units = str(vdf.data.attrs.get("UNITS", "s^3/cm^6"))

    # The background flux in all the directions, in the units of vdf
    shape = (-1,) + (1,) * (vdf.data.ndim - 1)
    flux = np.asarray(def_bg.data, dtype=np.float64).reshape(shape)
    vdf_bg = vdf.copy()
    vdf_bg["data"] = vdf.data.copy(data=np.broadcast_to(flux, vdf.data.shape))
    vdf_bg.data.attrs = {"UNITS": bg_units}

    with np.errstate(divide="ignore"):  # channels at 0 eV
        vdf_bg = vdf_convert(vdf_bg, units)

    vdf_new = vdf.copy(deep=True)  # deep: the caller's VDF is unchanged
    data = vdf_new.data.data - vdf_bg.data.data
    vdf_new.data.data[...] = np.where(data < 0, 0.0, data)

    return vdf_new
