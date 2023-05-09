#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.3.26"
__status__ = "Prototype"


def ts_skymap(time, data, energy, phi, theta, **kwargs):
    r"""Creates a skymap of the distribution function.

    Parameters
    ----------
    time : array_like
        List of times.
    data : array_like
        Values of the distribution function.
    energy : array_like
        Energy levels.
    phi : array_like
        Azimuthal angles.
    theta : array_like
        Elevation angles.

    Other Parameters
    ---------------
    **kwargs
        Hash table of keyword arguments with :
            * energy0 : array_like
                Energy table 0 (odd time indices).
            * energy1 : array_like
                Energy table 1 (even time indices).
            * esteptable : array_like
                Time series of the stepping table between energies (burst).

    Returns
    -------
    out : xarray.Dataset
        Skymap of the distribution function.

    """

    # Check if even (odd) time step energy channels energy1 (energy0), and
    # energy step table are provided.
    energy0 = kwargs.get("energy0", None)
    energy1 = kwargs.get("energy1", None)
    esteptable = kwargs.get("esteptable", None)
    attrs = kwargs.get("attrs", {})
    coords_attrs = kwargs.get("coords_attrs", {})
    glob_attrs = kwargs.get("glob_attrs", {})

    if energy is None:
        assert energy0 is not None and energy1 is not None and esteptable is not None

        energy = np.tile(energy0, (len(esteptable), 1))

        energy[esteptable == 1] = np.tile(
            energy1,
            (int(np.sum(esteptable)), 1),
        )

    if phi.ndim == 1:
        phi = np.tile(phi, (len(time), 1))

    out_dict = {
        "data": (["time", "idx0", "idx1", "idx2"], data),
        "phi": (["time", "idx1"], phi),
        "theta": (["idx2"], theta),
        "energy": (["time", "idx0"], energy),
        "time": time,
        "idx0": np.arange(energy.shape[1]),
        "idx1": np.arange(phi.shape[1]),
        "idx2": np.arange(len(theta)),
    }

    # Construct global attributes and sort them
    # remove energy0, energy1, and esteptable from global attrs to overwrite
    overwrite_keys = ["energy0", "energy1", "esteptable"]
    glob_attrs = {k: glob_attrs[k] for k in glob_attrs if k not in overwrite_keys}
    glob_attrs = {
        "energy0": energy0,
        "energy1": energy1,
        "esteptable": esteptable,
        **glob_attrs,
    }

    glob_attrs = {k: glob_attrs[k] for k in sorted(glob_attrs)}

    # Create Dataset
    out = xr.Dataset(out_dict, attrs=glob_attrs)

    # Sort and fill coordinates attributes
    for k in coords_attrs:
        out[k].attrs = {m: coords_attrs[k][m] for m in sorted(coords_attrs[k])}

    # Sort and fill data attributes
    out.data.attrs = {k: attrs[k] for k in sorted(attrs)}

    return out
