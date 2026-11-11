#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np

from ..pyrf.ts_skymap import ts_skymap

# Local imports
from ._eis_attrs import _common_attrs
from .eis_spec_combine_sc import _idx_closest, _nearest

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


# Per-time or per-channel attributes, set from the combined skymap
_SKIPPED_ATTRS = (
    "delta_energy_plus",
    "delta_energy_minus",
    "esteptable",
    "energy0",
    "energy1",
)


def eis_skymap_combine_sc(skymaps, method: str = "mean"):
    r"""Generate composite skymap from the EIS sensors across the MMS
    spacecraft.

    Parameters
    ----------
    skymaps : list of xarray.Dataset
        Skymap distribution for all spacecraft (from
        :func:`pyrfu.mms.eis_skymap`), with the same units and species.
    method : str, Optional
        Method to combine spectra, "mean" or "sum"

    Returns
    -------
    out : xarray.Dataset
        Composite skymap distribution, on the times of the spacecraft with the
        fewest skymaps and the energies (and widths) of the one with the fewest
        channels (the others at their closest channels).

    Raises
    ------
    ValueError
        If method is not "mean" or "sum", or if the skymaps have different
        units or species.

    Notes
    -----
    The skymaps (one per spin) are combined at the nearest time of each
    spacecraft, within half a spin (they were combined by index: one missing
    spin shifted all the next ones), and NaN where no spacecraft has data
    (the sum was 0). The energy widths are per channel (they were repeated
    for every time in a flat array).

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_pad,
    pyrfu.mms.eis_spec_combine_sc, pyrfu.mms.eis_spec_combine_sc

    """
    if method.lower() not in ["mean", "sum"]:
        raise ValueError(f"method must be mean or sum, not {method}")

    for key in ["UNITS", "species"]:
        values = {
            str(skymap.data.attrs.get(key, skymap.attrs.get(key))) for skymap in skymaps
        }

        if len(values) > 1:
            raise ValueError(f"The skymaps have different {key}: {sorted(values)}")

    # Spacecraft with the fewest skymaps (reference times) and with the fewest
    # energy channels (reference energies)
    ref_probe = skymaps[np.argmin([len(probe.time.data) for probe in skymaps])]
    ener_probe = skymaps[np.argmin([probe.energy.shape[1] for probe in skymaps])]
    ref_energy = ener_probe.energy.data[0, :]

    # Use azimuthal and elevation angle from reference spacecraft (in
    # practice they are the same for all spacecraft)
    phi = ref_probe.phi.data
    theta = ref_probe.theta.data

    t_ref = ref_probe.time.data.astype("datetime64[ns]").astype(np.int64)
    d_t = np.median(np.diff(t_ref)) if len(t_ref) > 1 else np.inf

    allmms_skymap = np.full(
        [len(t_ref), len(ref_energy), phi.shape[1], len(theta), len(skymaps)],
        np.nan,
    )

    # Nearest skymaps and closest channels (they were paired by index)
    for i_s, skymap in enumerate(skymaps):
        t_sky = skymap.time.data.astype("datetime64[ns]").astype(np.int64)
        idx_time = _nearest(t_ref, t_sky, d_t / 2)
        idx_en = _idx_closest(skymap.energy.data[0, :], ref_energy)
        paired = idx_time >= 0
        allmms_skymap[paired, ..., i_s] = skymap.data.data[idx_time[paired]][
            :, idx_en, ...
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method.lower() == "mean":
            # Average the four spacecraft
            allmms_skymap_avg = np.nanmean(allmms_skymap, axis=-1)
        else:
            # Sum the four spacecraft, NaN without data (it was 0)
            allmms_skymap_avg = np.nansum(allmms_skymap, axis=-1)
            allmms_skymap_avg[np.all(np.isnan(allmms_skymap), axis=-1)] = np.nan

    # Combine global attributes, with the energy widths of the reference
    # energies
    glob_attrs = _common_attrs([s.attrs for s in skymaps], _SKIPPED_ATTRS)
    glob_attrs = {
        "delta_energy_plus": np.asarray(ener_probe.attrs["delta_energy_plus"]),
        "delta_energy_minus": np.asarray(ener_probe.attrs["delta_energy_minus"]),
        **glob_attrs,
    }

    # Combine coordinates attributes
    coords_attrs = {}
    for coor in ["time", "energy", "phi", "theta"]:
        coords_attrs[coor] = _common_attrs(
            [skymap[coor].attrs for skymap in skymaps], _SKIPPED_ATTRS
        )

    # Combine data attributes
    attrs = _common_attrs([skymap.data.attrs for skymap in skymaps], _SKIPPED_ATTRS)

    common_energy = np.tile(ref_energy, (len(t_ref), 1))

    # Create combined skymap
    out = ts_skymap(
        ref_probe.time.data,
        allmms_skymap_avg,
        common_energy,
        phi,
        theta,
        energy0=ref_energy,
        energy1=ref_energy,
        esteptable=np.zeros(len(t_ref)),
        attrs=attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )

    return out
