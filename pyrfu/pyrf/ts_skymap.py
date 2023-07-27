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

    # Check input type
    assert isinstance(time, np.ndarray), "time must be numpy.ndarray"
    assert isinstance(data, np.ndarray), "data must be numpy.ndarray"
    assert isinstance(energy, np.ndarray), "energy must be numpy.ndarray"
    assert isinstance(phi, np.ndarray), "phi must be numpy.ndarray"
    assert isinstance(theta, np.ndarray), "theta must be numpy.ndarray"

    # Check if even (odd) time step energy channels energy1 (energy0), and
    # energy step table are provided.
    energy0 = kwargs.get("energy0", energy[0, :])
    energy1 = kwargs.get("energy1", energy[1, :])
    esteptable = kwargs.get("esteptable", np.zeros(len(time)))

    # Check that energy0 and energy1
    assert isinstance(energy0, np.ndarray), "energy0 must be 1D numpy.ndarray"
    assert energy0.ndim == 1, "energy0 must be 1D numpy.ndarray"
    assert energy0.shape[0] == energy.shape[1], "energy0 is not consistent with time"
    assert isinstance(energy1, np.ndarray), "energy1 must be 1D numpy.ndarray"
    assert energy1.ndim == 1, "energy1 must be 1D numpy.ndarray"
    assert energy1.shape[0] == energy.shape[1], "energy1 is not consistent with time"

    # Check esteptable
    assert isinstance(esteptable, np.ndarray), "esteptable must be 1D numpy.ndarray"
    assert esteptable.ndim == 1, "esteptable must be 1D numpy.ndarray"
    assert esteptable.shape[0] == len(time), "esteptable is not consistent with time"

    attrs = kwargs.get("attrs", {})
    coords_attrs = kwargs.get("coords_attrs", {})
    glob_attrs = kwargs.get("glob_attrs", {})

    # Check attributes are dictionaries
    assert isinstance(attrs, dict)

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
