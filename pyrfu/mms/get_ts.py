#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import re
import warnings
from typing import Any

# 3rd party imports
import numpy as np
import xarray as xr
from pycdfpp import DataType, load, to_datetime64
from xarray import DataArray, Dataset

# Local imports
from pyrfu._core.time import _timedelta

from ..pyrf.time_clip import _time_interval, time_clip
from .get_variable import _pycdfpp_attributes_to_dict

logger = logging.getLogger(__name__)

# Time types of the epochs converted to datetime64 (CDF_EPOCH and CDF_EPOCH16
# epochs were left as structured arrays, and raised KeyError)
_EPOCH_TYPES = (DataType.CDF_TIME_TT2000, DataType.CDF_EPOCH, DataType.CDF_EPOCH16)


def _shift_epochs(file, epoch):
    r"""Shift times for particles."""

    epoch_shifted = epoch["data"].copy()

    try:
        delta_minus_key = epoch["attrs"]["DELTA_MINUS_VAR"]
        delta_minus_var = {
            "data": file[delta_minus_key].values,
            "attrs": _pycdfpp_attributes_to_dict(file[delta_minus_key].attributes),
        }
        delta_plus_key = epoch["attrs"]["DELTA_PLUS_VAR"]
        delta_plus_var = {
            "data": file[delta_plus_key].values,
            "attrs": _pycdfpp_attributes_to_dict(file[delta_plus_key].attributes),
        }

        delta_vars = [delta_minus_var, delta_plus_var]
        flags_vars = [1e3, 1e3]  # Time scaling conversion flags

        delta_keys = [delta_minus_key, delta_plus_key]

        # The variable and its units in the warnings (" units are not clear")
        for i, (delta_key, delta_var) in enumerate(zip(delta_keys, delta_vars)):
            if isinstance(delta_var["attrs"], dict) and "UNITS" in delta_var["attrs"]:
                units = delta_var["attrs"]["UNITS"]
                if units.lower() == "s":
                    flags_vars[i] = 1e3
                elif units.lower() == "ms":
                    flags_vars[i] = 1e0
                else:
                    warnings.warn(f"{delta_key}: units {units!r} unclear, assumed s")
            else:
                warnings.warn(f"{delta_key}: no units, assumed s")

        flag_minus, flag_plus = flags_vars

        # Shift to the centre of [epoch - delta_minus, epoch + delta_plus] and half
        # width of the accumulation, as mms.variable2ts in irfu-matlab
        t_offset, t_diff = [
            _timedelta(np.round(delta, 1) / 2e3)
            for delta in [
                delta_plus_var["data"] * flag_plus
                - delta_minus_var["data"] * flag_minus,
                delta_plus_var["data"] * flag_plus
                + delta_minus_var["data"] * flag_minus,
            ]
        ]

        # Half width of the accumulation
        half_width = np.mean(t_diff)

        # If the accumulation does not match the sampling period, assume that the
        # epochs are start times and use half the sampling period (needs at least
        # two records, otherwise keep the delta variables)
        if len(epoch["data"]) > 1:
            t_diff_data = np.median(np.diff(epoch["data"])) / 2

            if not np.isnat(t_diff_data) and t_diff_data != np.mean(t_diff):
                mismatch = np.abs(t_diff_data - np.mean(t_diff)) / t_diff_data
                if mismatch > 0.01:
                    logger.warning(
                        "Epoch delta variables (half width %s) do not match the "
                        "sampling time (half %s), assume the latter",
                        np.mean(t_diff),
                        t_diff_data,
                    )

                t_offset = t_diff_data
                half_width = t_diff_data

        epoch_shifted += t_offset

        return {
            "data": epoch_shifted,
            "attrs": epoch["attrs"],
            "half_width": half_width / np.timedelta64(1, "s"),
        }

    except KeyError:
        return {"data": epoch_shifted, "attrs": epoch["attrs"]}


def _get_epochs(file, cdf_name):
    r"""Get epochs form cdf and shift if needed."""

    depend0_key = file[cdf_name].attributes["DEPEND_0"][0]

    out = {
        "data": file[depend0_key].values,
        "attrs": _pycdfpp_attributes_to_dict(file[depend0_key].attributes),
    }

    if file[depend0_key].type not in _EPOCH_TYPES:
        raise TypeError(
            f"{depend0_key} is not a CDF time (TT2000, EPOCH or EPOCH16): "
            f"{file[depend0_key].type}"
        )

    out["data"] = to_datetime64(out["data"])

    # Shift times if particle data (FPI or HPCA)
    if re.search("^mms[1-4]_(d[ei]s|hpca)_", cdf_name):
        out = _shift_epochs(file, out)

    return out


def _decode_labels(labels):
    r"""Labels as stripped str (CDF character variables are padded bytes)."""
    labels = np.asarray(labels)

    if labels.dtype.kind == "S":
        labels = np.char.decode(labels, "utf-8")

    if labels.dtype.kind == "U":
        labels = np.char.strip(labels)

    return labels


def _last_label_is_magnitude(file, cdf_name):
    r"""True if the last label of LABL_PTR_1 is a magnitude (e.g. MEC B fields)."""
    if "LABL_PTR_1" not in file[cdf_name].attributes:
        return False

    labels = _decode_labels(file[file[cdf_name].attributes["LABL_PTR_1"][0]].values)
    labels = labels.ravel()

    return len(labels) == 4 and str(labels[-1]).lower().endswith("mag")


def _get_depend_attributes(file, depend_key):
    attributes = _pycdfpp_attributes_to_dict(file[depend_key].attributes)

    # Remove spaces in label
    try:
        attributes["LABLAXIS"] = attributes["LABLAXIS"].replace(" ", "_")

        if attributes["LABLAXIS"] == "Diffential_energy_channels":
            attributes["LABLAXIS"] = "Differential_energy_channels"

    except (KeyError, AttributeError):
        attributes["LABLAXIS"] = "comp"

    return attributes


def _get_depend(file, cdf_name, dep_num=1):
    out = {}

    if f"DEPEND_{dep_num:d}" in file[cdf_name].attributes:
        depend_key = file[cdf_name].attributes[f"DEPEND_{dep_num:d}"][0]
    elif f"REPRESENTATION_{dep_num:d}" in file[cdf_name].attributes:
        depend_key = file[cdf_name].attributes[f"REPRESENTATION_{dep_num:d}"][0]
    else:
        raise KeyError(f"no DEPEND_{dep_num:d}/REPRESENTATION_{dep_num:d} attributes")

    # Components, of a vector or of a vector and its magnitude (HPCA vperp,
    # vparallel and B: "x,y,z,total", which raised KeyError): the magnitude is
    # removed from the data, as for FGM
    if depend_key in ["x,y,z", "x,y,z,total"]:
        out["data"] = np.array(["x", "y", "z"])

        out["attrs"] = {"LABLAXIS": "comp"}
    else:
        out["data"] = _decode_labels(file[depend_key].values)

        if len(out["data"]) == 1:
            out["data"] = out["data"][0]

        # Vector and magnitude (e.g. FGM): the magnitude is removed from the data
        if (
            out["data"].ndim == 1
            and len(out["data"]) == 4
            and all(out["data"].astype(str) == ["x", "y", "z", "r"])
        ):
            out["data"] = out["data"].astype(str)[:-1]

        elif out["data"].ndim == 2:
            if len(out["data"].flatten()) == 3:
                out["data"] = out["data"].flatten()
            elif out["data"].shape[0] == 0:
                # Record varying table of a file without records: NaN values
                out["data"] = np.full(out["data"].shape[1], np.nan)
            else:
                # Record varying table (e.g. FPI burst energies): the table of
                # each record, the first one as the 1-D coordinate
                out["table"] = out["data"]
                out["data"] = out["data"][0, :]

        out["attrs"] = _get_depend_attributes(file, depend_key)

    return out


def _mask_fill(data, fill_val):
    r"""Floating-point data with the values equal to FILLVAL set to NaN, as
    irfu-matlab mms.variable2ts and pySPEDAS (the moments of the HPCA species
    without data were -1e31); integer data (flags, counts) unchanged."""
    if fill_val is None or not np.issubdtype(data.dtype, np.floating):
        return data

    # A NaN fill value matches no value
    is_fill = data == np.ravel(np.asarray(fill_val, dtype=np.float64))[0]

    if not np.any(is_fill):
        return data

    out = np.array(data, copy=True)
    out[is_fill] = np.nan
    return out


def _record_varying_axis(out, dim, table, depend_attrs):
    r"""Time series of a 2-D variable whose axis `dim` (e.g. the FPI burst
    energies) is a table per record, `table` (time, n), clipped as `out`: a
    DataArray with that axis if it is the same at all times, else a Dataset
    as ``pyrf.ts_spectr`` (the data (time, idx0) and the coordinate `dim`
    (time, idx0); the first table labelled every record before)."""
    if len(table) == 0 or np.all(table == table[0]):
        if len(table):
            out = out.assign_coords({dim: table[0]})
            out[dim].attrs = depend_attrs

        return out

    time_attrs = out.time.attrs
    out = xr.Dataset(
        {"data": (["time", "idx0"], out.data, out.attrs)},
        coords={
            "time": out.time.data,
            "idx0": np.arange(table.shape[1]),
            dim: (["time", "idx0"], table, depend_attrs),
        },
    )
    out.time.attrs = time_attrs
    return out


def _unique_labels(depends):
    r"""Dimension names of the depends made distinct: a label used by more
    than one depend gets the number of its DEPEND (the blank labels of
    DEPEND_1 and DEPEND_3 of the HPCA azimuths gave the duplicate dimensions
    ('time', '_', 'comp', '_'), on which xarray selects both or raises)."""
    labels = [dep["attrs"]["LABLAXIS"] if dep else None for dep in depends]

    for i, (dep, label) in enumerate(zip(depends, labels), start=1):
        if dep and labels.count(label) > 1:
            dep["attrs"] = {**dep["attrs"], "LABLAXIS": f"{label}{i:d}"}


def get_ts(
    file_path: str | bytes, cdf_name: str, tint: list | None = None
) -> DataArray | Dataset:
    r"""Reads field named cdf_name in file and convert to time series.

    Parameters
    ----------
    file_path : str
        Path of the cdf file.
    cdf_name : str
        Name of the target variable in the cdf file.
    tint : array_like or xarray.DataArray, Optional
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`. Default is all the times of the file.

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the target variable in the selected time interval. A
        2-D variable whose axis is a table per record that changes within the
        interval (e.g. the FPI burst energy spectra of the early mission,
        with two alternating energy tables) is a Dataset as
        `pyrfu.pyrf.ts_spectr`: the variable data (time, idx0) and the
        coordinate of the axis (time, idx0). A vector and its magnitude (FGM
        "x, y, z, r", HPCA "x,y,z,total") is the vector (time, comp): the
        magnitude is removed. The fill values (``FILLVAL``) of floating-point
        variables are NaN, as irfu-matlab and pySPEDAS; integer variables
        (flags, counts) keep them.

    Raises
    ------
    ValueError
        If the variable is not a time series (no epoch DEPEND_0).
    NotImplementedError
        If the fourth dimension of the variable is of components ("x,y,z").

    """

    # Time interval, in the forms of time_clip (a tuple, datetime.datetime
    # and a time series raised TypeError), checked before reading the file
    if tint is None:
        tint = ["1995-10-06T18:50:00.000000000", "2200-10-06T18:50:00.000000000"]

    tint = _time_interval(tint)

    out_dict: dict[str, Any] = {}
    time, depend_1, depend_2, depend_3 = [{}, {}, {}, {}]

    # Load CDF file
    file = load(file_path)

    var_attrs = _pycdfpp_attributes_to_dict(file[cdf_name].attributes)
    glb_attrs = _pycdfpp_attributes_to_dict(file.attributes)
    out_dict["attrs"] = {"GLOBAL": glb_attrs, **var_attrs}
    out_dict["attrs"] = {k: out_dict["attrs"][k] for k in sorted(out_dict["attrs"])}

    if "DEPEND_0" not in var_attrs or "epoch" not in var_attrs["DEPEND_0"].lower():
        raise ValueError(f"{cdf_name} is not a time series (no epoch DEPEND_0)")

    time = _get_epochs(file, cdf_name)

    # Dimensions of the variable, with the records: a DEPEND of a dimension
    # the variable does not have labels nothing (HPCA data_quality, 1-D with
    # the DEPEND_1 of the anodes, raised ValueError)
    n_dims = len(file[cdf_name].shape)

    if "DEPEND_1" in var_attrs or "REPRESENTATION_1" in var_attrs:
        if n_dims > 1:
            depend_1 = _get_depend(file, cdf_name, 1)

    elif "afg" in cdf_name or "dfg" in cdf_name:
        depend_1 = {
            "data": ["x", "y", "z"],
            "attrs": {"LABLAXIS": "comp"},
        }

    if n_dims > 2 and ("DEPEND_2" in var_attrs or "REPRESENTATION_2" in var_attrs):
        depend_2 = _get_depend(file, cdf_name, 2)

        if depend_2["attrs"]["LABLAXIS"] == depend_1["attrs"]["LABLAXIS"]:
            depend_1["attrs"]["LABLAXIS"] = "rcomp"
            depend_2["attrs"]["LABLAXIS"] = "ccomp"

    if n_dims > 3 and ("DEPEND_3" in var_attrs or "REPRESENTATION_3" in var_attrs):
        if out_dict["attrs"].get("REPRESENTATION_3") == "x,y,z":
            raise NotImplementedError(
                f"{cdf_name}: a fourth dimension of components is not supported"
            )

        depend_3 = _get_depend(file, cdf_name, 3)

        if depend_3["attrs"]["LABLAXIS"] == depend_2["attrs"]["LABLAXIS"]:
            depend_2["attrs"]["LABLAXIS"] = "rcomp"
            depend_3["attrs"]["LABLAXIS"] = "ccomp"

    if "sector_mask" in cdf_name:
        # FEEPS sector mask: one value per spin sector
        depend_1["data"] = np.arange(file[cdf_name].shape[-1])
        depend_1["attrs"] = {"LABLAXIS": "sectornum"}

    if "edp_dce_sensor" in cdf_name:
        depend_1["data"] = ["x", "y", "z"]
        depend_1["attrs"] = {"LABLAXIS": "comp"}

    # Fill values to NaN (floating-point data), as irfu-matlab
    out_dict["data"] = _mask_fill(file[cdf_name].values, var_attrs.get("FILLVAL"))

    # Non-record-varying variables (e.g. FEEPS sector masks): same values at
    # all times, without copy until the time clip
    is_nrv = file[cdf_name].is_nrv and len(out_dict["data"]) != len(time["data"])

    if is_nrv:
        shape = (len(time["data"]), *out_dict["data"].shape[1:])
        out_dict["data"] = np.broadcast_to(out_dict["data"][0], shape)

    # Remove the magnitude of (x, y, z, magnitude) variables only (not e.g. the
    # scalar part of quaternions)
    if out_dict["data"].ndim == 2 and out_dict["data"].shape[1] == 4:
        if depend_1:
            is_magnitude = len(depend_1["data"]) == 3
        else:
            is_magnitude = _last_label_is_magnitude(file, cdf_name)

        if is_magnitude:
            out_dict["data"] = out_dict["data"][:, :-1]

    if out_dict["data"].ndim == 2 and not depend_1:
        depend_1["data"] = np.arange(out_dict["data"].shape[1])
        depend_1["attrs"] = {"LABLAXIS": "idx"}

    _unique_labels([depend_1, depend_2, depend_3])

    # The dimensions of the depends present, in order
    depends = [dep for dep in [depend_1, depend_2, depend_3] if dep]
    dims = ["time", *[dep["attrs"]["LABLAXIS"] for dep in depends]]
    coords_data = [time["data"], *[dep["data"] for dep in depends]]
    coords_attrs = [time["attrs"], *[dep["attrs"] for dep in depends]]

    out = xr.DataArray(
        out_dict["data"],
        coords=coords_data,
        dims=dims,
        attrs=out_dict["attrs"],
    )

    for dim, coord_attrs in zip(dims, coords_attrs):
        # Sort attributes and set to coordinates attribute
        out[dim].attrs = {k: coord_attrs[k] for k in sorted(coord_attrs)}

    # Axis table per record (time, n) of a 2-D variable, clipped with it
    table = depend_1.get("table")
    has_table = (
        table is not None and out.ndim == 2 and table.shape == out.shape and not is_nrv
    )

    if has_table:
        out = out.assign_coords(_table=(dims, table))

    for dep in [depend_1, depend_2, depend_3]:
        if dep.get("table") is not None and not has_table:
            if not np.all(dep["table"] == dep["table"][0]):
                logger.warning(
                    "%s: the axis %s changes with time; the first record is used",
                    cdf_name,
                    dep["attrs"]["LABLAXIS"],
                )

    # Time clip to original time interval
    out = time_clip(out, tint)

    if has_table:
        table = out["_table"].data
        out = out.drop_vars("_table")
        out = _record_varying_axis(out, dims[1], table, coords_attrs[1])

    if is_nrv:
        # writable copy of the broadcast (read-only) data
        out = out.copy()

    return out
