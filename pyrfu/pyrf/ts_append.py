#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import List, Mapping, Optional

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def _append_attrs(attrs0: dict, attrs1: dict, n_0: int, n_1: int) -> dict:
    # The time-dependent attributes (arrays with the times along the first
    # dimension, as in time_clip) of both inputs are appended; the others are
    # those of the first input
    out = {}

    for k, attr in attrs0.items():
        other = attrs1.get(k)

        if (
            isinstance(attr, np.ndarray)
            and isinstance(other, np.ndarray)
            and attr.ndim > 0
            and attr.shape[0] == n_0
            and other.shape == (n_1, *attr.shape[1:])
        ):
            out[k] = np.concatenate([attr, other])
        else:
            out[k] = attr

    return out


def ts_append(
    inp0: Optional[DataArray] = None, inp1: Optional[DataArray] = None
) -> DataArray:
    r"""Concatenate two time series along the time axis.

    Parameters
    ----------
    inp0 : DataArray, Optional
        Time series of the first input (early times).
    inp1 : DataArray, Optional
        Time series of the second input (late times).

    Returns
    -------
    DataArray
        Concatenated time series.

    Notes
    -----
    The time series must be in the correct time order. The other coordinates
    are those of `inp0`, except if it has no time samples: an empty input is
    skipped. The array attributes with the times along the first dimension (in
    both inputs) are appended; the other attributes are those of `inp0`.

    """

    # An input without time samples (e.g. from a file without records) is
    # skipped, so that the coordinates of the other one are kept
    if inp0 is None or (inp1 is not None and inp0.shape[0] == 0):
        out = inp1
    elif inp1 is None or inp1.shape[0] == 0:
        out = inp0
    elif inp0 is not None and inp1 is not None:
        out_data = {}

        if inp0.data.ndim != 1:
            out_data["data"] = np.vstack([inp0.data, inp1.data])

        else:
            out_data["data"] = np.hstack([inp0.data, inp1.data])

        n_0, n_1 = inp0.shape[0], inp1.shape[0]
        out_data["attrs"] = _append_attrs(inp0.attrs, inp1.attrs, n_0, n_1)

        depends: List[Mapping[str, object]] = [{} for _ in range(len(inp0.dims))]

        for i, dim in enumerate(inp0.dims):
            if i == 0 or dim == "time":
                depends[i]["data"] = np.hstack([inp0[dim].data, inp1[dim].data])

                # add attributes
                depends[i]["attrs"] = _append_attrs(
                    inp0[dim].attrs, inp1[dim].attrs, n_0, n_1
                )

            else:
                # Use values of other coordinates of inp0 assuming equal to inp1
                depends[i]["data"] = inp0[dim].data

                # add attributes
                depends[i]["attrs"] = {}

                for k in inp0[dim].attrs:
                    depends[i]["attrs"][k] = inp0[dim].attrs[k]

        # Create DataArray
        out = xr.DataArray(
            out_data["data"],
            coords=[depend["data"] for depend in depends],
            dims=inp0.dims,
            attrs=out_data["attrs"],
        )

        for i, dim in enumerate(out.dims):
            out[dim].attrs = depends[i]["attrs"]

    else:
        raise ValueError("At least one input must be provided")

    return out
