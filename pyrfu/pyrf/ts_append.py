#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, cast

# 3rd party imports
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

# Local imports
from pyrfu._core.time_attrs import _is_time_attr

# Built-in imports


def _append_attrs(attrs0: dict, attrs1: dict, n_0: int, n_1: int) -> dict:
    # The time-dependent attributes of both inputs (the rule of time_clip: the
    # energy tables of 32 channels with 32 times were appended) are appended;
    # the others are those of the first input
    out = {}

    for k, attr in attrs0.items():
        other = attrs1.get(k)

        if (
            _is_time_attr(k, attr, n_0)
            and isinstance(other, np.ndarray)
            and other.shape == (n_1, *attr.shape[1:])
        ):
            out[k] = np.concatenate([attr, other])
        else:
            out[k] = attr

    return out


def _n_times(inp: DataArray | Dataset) -> int:
    # Number of times: the first dimension of a DataArray, "time" of a Dataset
    return inp.sizes["time"] if isinstance(inp, Dataset) else inp.shape[0]


def _spectrum(inp: DataArray | Dataset):
    r"""The parts of a spectrum: a 2-D DataArray (time, axis) or a Dataset of
    ``ts_spectr`` (the data (time, idx0) and an axis coordinate (time, idx0)):
    data, axis table (time, n), axis name, axis attributes, data attributes."""
    if isinstance(inp, Dataset):
        name = next(k for k, v in inp.coords.items() if v.dims == ("time", "idx0"))
        return (
            inp["data"].data,
            inp[name].data,
            name,
            dict(inp[name].attrs),
            dict(inp["data"].attrs),
        )

    name = inp.dims[1]
    table = np.broadcast_to(inp[name].data, inp.shape)
    return inp.data, table, name, dict(inp[name].attrs), dict(inp.attrs)


def _is_spectra(inp0: DataArray | Dataset, inp1: DataArray | Dataset) -> bool:
    r"""Spectra whose axes differ: a ``ts_spectr`` Dataset, or 2-D numeric
    axes not equal (the axis of the first input labelled both)."""
    if isinstance(inp0, Dataset) or isinstance(inp1, Dataset):
        return True

    if inp0.ndim != 2 or inp1.ndim != 2 or inp0.dims != inp1.dims:
        return False

    axis0, axis1 = inp0[inp0.dims[1]].data, inp1[inp1.dims[1]].data

    if axis0.dtype.kind not in "iuf" or axis1.dtype.kind not in "iuf":
        return False

    return bool(axis0.shape != axis1.shape or not np.array_equal(axis0, axis1))


def _append_spectra(inp0: DataArray | Dataset, inp1: DataArray | Dataset):
    r"""Spectra appended with the axis table of each time: a DataArray if the
    axis is the same at all times, else a Dataset as ``ts_spectr``."""
    data0, table0, name, axis_attrs, attrs0 = _spectrum(inp0)
    data1, table1, _, _, attrs1 = _spectrum(inp1)
    n_0, n_1 = len(data0), len(data1)

    time = np.hstack([inp0.time.data, inp1.time.data])
    time_attrs = _append_attrs(inp0.time.attrs, inp1.time.attrs, n_0, n_1)
    data = np.vstack([data0, data1])
    table = np.vstack([table0, table1])
    attrs = _append_attrs(attrs0, attrs1, n_0, n_1)

    out: DataArray | Dataset

    if np.all(table == table[0]):
        out = xr.DataArray(data, coords=[time, table[0]], dims=["time", name])
        out.attrs = attrs
        out[name].attrs = axis_attrs
    else:
        out = xr.Dataset(
            {"data": (["time", "idx0"], data, attrs)},
            coords={
                "time": time,
                "idx0": np.arange(table.shape[1]),
                name: (["time", "idx0"], table, axis_attrs),
            },
        )

    out.time.attrs = time_attrs
    return out


def ts_append(
    inp0: DataArray | Dataset | None = None,
    inp1: DataArray | Dataset | None = None,
) -> DataArray | Dataset | None:
    r"""Concatenate two time series along the time axis.

    Parameters
    ----------
    inp0 : DataArray or Dataset, Optional
        Time series of the first input (early times), or a spectrum whose
        axis changes with time (a Dataset of `ts_spectr`).
    inp1 : DataArray or Dataset, Optional
        Time series of the second input (late times), as `inp0`.

    Returns
    -------
    DataArray, Dataset or None
        Concatenated time series: the other input if one is None, None if both
        are (so that a loop can start from None, ``out = ts_append(out, ts)``).
        Spectra (2-D) whose axes differ, or a `ts_spectr` Dataset, keep the
        axis of each time: a DataArray if it is the same at all times, else a
        Dataset as `ts_spectr`.

    Notes
    -----
    The time series must be in the correct time order. The other coordinates
    are those of `inp0`, except if it has no time samples: an empty input is
    skipped. The array attributes with the times along the first dimension (in
    both inputs) are appended; the other attributes are those of `inp0`.

    """

    # An input without time samples (e.g. from a file without records) is
    # skipped, so that the coordinates of the other one are kept
    if inp0 is None or (inp1 is not None and _n_times(inp0) == 0):
        out = inp1
    elif inp1 is None or _n_times(inp1) == 0:
        out = inp0
    elif _is_spectra(inp0, inp1):
        out = _append_spectra(inp0, inp1)
    else:
        # Both with time samples, DataArrays (_is_spectra for any Dataset)
        inp0, inp1 = cast(DataArray, inp0), cast(DataArray, inp1)
        out_data: dict[str, Any] = {}

        if inp0.data.ndim != 1:
            out_data["data"] = np.vstack([inp0.data, inp1.data])

        else:
            out_data["data"] = np.hstack([inp0.data, inp1.data])

        n_0, n_1 = inp0.shape[0], inp1.shape[0]
        out_data["attrs"] = _append_attrs(inp0.attrs, inp1.attrs, n_0, n_1)

        depends: list[dict[str, Any]] = [{} for _ in range(len(inp0.dims))]

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

    return out
