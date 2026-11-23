#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import interpolate

# Local imports
from pyrfu._core.time import _ns, _seconds
from pyrfu._core.time_attrs import _is_time_attr

# Interpolation methods: the kinds of scipy.interpolate.interp1d
_METHODS = (
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
    "pchip",
)

# Methods whose interpolant depends on all the samples, where a NaN would
# spread to the whole output: each component is interpolated through its
# finite samples, as MATLAB interp1 ("spline", "pchip")
_GLOBAL_METHODS = ("quadratic", "cubic", "pchip")

# Statistics of the averaging windows, as the irf_resamp flags; the NaN-aware
# ones for the points kept by thresh
_AVERAGES = {"mean": np.mean, "median": np.median, "max": np.max}
_NAN_AVERAGES = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax}


def _interpolant(inp_time, inp_data, method, fill_value):
    r"""Interpolant of inp_data (along the first axis): scipy interp1d, or the
    piecewise cubic Hermite interpolant for "pchip" (MATLAB interp1 "pchip"),
    extrapolated with its end polynomials or NaN outside."""

    if method == "pchip":
        return interpolate.PchipInterpolator(
            inp_time, inp_data, axis=0, extrapolate=fill_value == "extrapolate"
        )

    return interpolate.interp1d(
        inp_time,
        inp_data,
        kind=method,
        axis=0,
        bounds_error=False,
        fill_value=fill_value,
    )


def _guess_sampling_frequency(ref_time):
    r"""Sampling frequency of the time line (in seconds), from the median time
    step (robust to an irregular first interval)."""

    d_t = np.median(np.diff(ref_time))

    if not np.isfinite(d_t) or d_t <= 0:
        raise RuntimeError("Cannot guess the sampling frequency of the reference")

    return 1 / d_t


def _average(inp_time, inp_data, ref_time, thresh, dt2, average="mean"):
    r"""Average (mean, median or max) inp_data in the windows (ref_time - dt2,
    ref_time + dt2] (as in irf_resamp), with inp_time, ref_time and dt2 in the
    same units. A window without samples, or with a NaN, gives NaN. With
    thresh, the points farther than thresh * std from the mean of the window
    are disregarded.
    """

    inp_data = np.asarray(inp_data, dtype=np.float64)
    shape = inp_data.shape[1:]
    data = inp_data.reshape(len(inp_data), -1)

    # Samples of each window: inp_time[idx_l:idx_r]
    idx_l = np.searchsorted(inp_time, ref_time - dt2, side="right")
    idx_r = np.searchsorted(inp_time, ref_time + dt2, side="right")
    n_pts = idx_r - idx_l

    if not thresh and average == "mean":
        # Sums of the windows from cumulative sums (offset by the mean to limit
        # round-off), with NaNs counted separately so they only affect their
        # own windows
        is_nan = np.isnan(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            offset = np.nan_to_num(np.nanmean(data, axis=0))
        values = np.where(is_nan, 0.0, data - offset)
        zeros = np.zeros((1, data.shape[1]))
        cum_sum = np.concatenate([zeros, np.cumsum(values, axis=0)])
        cum_nan = np.concatenate([zeros, np.cumsum(is_nan, axis=0)])

        with np.errstate(invalid="ignore", divide="ignore"):
            out_data = (cum_sum[idx_r] - cum_sum[idx_l]) / n_pts[:, None] + offset

        out_data[(n_pts == 0) | np.any(cum_nan[idx_r] - cum_nan[idx_l] > 0, axis=1)] = (
            np.nan
        )
    else:
        out_data = np.full((len(ref_time), data.shape[1]), np.nan)

        for i, (i_l, i_r) in enumerate(zip(idx_l, idx_r)):
            if i_r == i_l:
                continue

            window = data[i_l:i_r]

            if not thresh:
                # A NaN gives NaN (np.median and np.max propagate it)
                out_data[i] = _AVERAGES[average](window, axis=0)
                continue

            # Standard deviation as MATLAB's std (N - 1; 0 for one sample); NaN
            # if the window contains a NaN
            std_ = np.std(window, axis=0, ddof=1) if len(window) > 1 else 0 * window[0]
            mean_ = np.mean(window, axis=0)
            keep = np.abs(window - mean_) <= thresh * std_

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out_data[i] = _NAN_AVERAGES[average](
                    np.where(keep, window, np.nan), axis=0
                )

    return out_data.reshape(len(ref_time), *shape)


def _interp_finite(inp_time, inp_data, ref_time, method, fill_value):
    r"""Interpolation of each component through its finite samples (NaN for
    a component without any), as MATLAB interp1 with "spline" or "pchip"
    (Interp1DStripNaN); a NaN would spread to all the times otherwise."""

    data = np.asarray(inp_data, dtype=np.float64).reshape(len(inp_time), -1)
    out_data = np.full((len(ref_time), data.shape[1]), np.nan)

    for k in range(data.shape[1]):
        finite = np.isfinite(data[:, k])

        if np.any(finite):
            out_data[:, k] = _interpolant(
                inp_time[finite], data[finite, k], method, fill_value
            )(ref_time)

    return out_data.reshape(len(ref_time), *np.shape(inp_data)[1:])


def _resampled_like(inp, out_data, time):
    r"""DataArray of out_data at the times time, with the dimensions, the
    dimension coordinates and the attributes of inp. A dimension without
    coordinate (e.g. dims ["time", "comp"], coords time only) is
    allowed."""

    coords = {inp.dims[0]: time}
    coords.update({k: inp.coords[k].data for k in inp.dims[1:] if k in inp.coords})

    return xr.DataArray(out_data, coords=coords, dims=inp.dims, attrs=inp.attrs)


def _resample_dataarray(
    inp, ref, method, f_s, window, thresh, extrapolate, average="mean"
):
    r"""Resample for time series (xarray.DataArray)"""

    flag_do = "check"

    if method:
        flag_do = "interpolation"

    # The median and max average whatever the sampling, as irf_resamp
    if average != "mean":
        flag_do = "average"

    if f_s is not None:
        sfy = f_s
    elif window is not None:
        sfy = 1 / window
    else:
        sfy = None

    # Both datetime64 (checked in resample) or both numeric
    if np.issubdtype(inp.time.dtype, np.datetime64):
        # Integer ns for the averaging windows, float seconds otherwise
        inp_time_ns = _ns(inp.time.data)
        ref_time_ns = _ns(ref.time.data)
        inp_time = _seconds(inp.time.data, inp.time.data[0])
        ref_time = _seconds(ref.time.data, inp.time.data[0])
    elif np.issubdtype(inp.time.dtype, np.number) and np.issubdtype(
        ref.time.dtype, np.number
    ):
        # Numeric times are in seconds
        inp_time = inp_time_ns = np.asarray(inp.time.data, dtype=np.float64)
        ref_time = ref_time_ns = np.asarray(ref.time.data, dtype=np.float64)
    else:
        raise TypeError("inp and ref must both have datetime64 or numeric times")

    if flag_do == "average" and not sfy:
        sfy = _guess_sampling_frequency(ref_time)

    if flag_do == "check":
        if len(ref_time) > 1:
            if not sfy:
                sfy = _guess_sampling_frequency(ref_time)

            if len(inp_time) / (inp_time[-1] - inp_time[0]) > 2 * sfy:
                flag_do = "average"
            else:
                flag_do = "interpolation"
        else:
            flag_do = "interpolation"

    # Internal: averages are chosen without method, and sfy is set then
    assert flag_do in ["average", "interpolation"]

    if flag_do == "average":
        assert not method, "cannot mix interpolation and averaging flags"

        if np.issubdtype(inp.time.dtype, np.datetime64):
            dt2 = int(np.round(0.5e9 / sfy))  # half window in ns
        else:
            dt2 = 0.5 / sfy

        out_data = _average(inp_time_ns, inp.data, ref_time_ns, thresh, dt2, average)

    else:
        if not method:
            method = "linear"

        # If time series agree, no interpolation is necessary.
        if len(inp_time_ns) == len(ref_time_ns) and np.array_equal(
            inp_time_ns, ref_time_ns
        ):
            return _resampled_like(inp, inp.data.copy(), ref.coords["time"].data)

        # Extrapolation outside the time range of inp with the interpolant,
        # as irf_resamp (MATLAB interp1 with 'extrap'), or NaN there
        if extrapolate:
            fill_value = "extrapolate"
        else:
            fill_value = np.nan

        if method in _GLOBAL_METHODS and not np.all(np.isfinite(inp.data)):
            out_data = _interp_finite(inp_time, inp.data, ref_time, method, fill_value)
        else:
            out_data = _interpolant(inp_time, inp.data, method, fill_value)(ref_time)

    return _resampled_like(inp, out_data, ref.coords["time"])


def _resample_dataset(inp, ref, **kwargs):
    r"""Resample for VDFs (xarray.Dataset)"""

    # Find time dependent zVariables and resample
    # The median or max of the data only, the mean of the energies, angles
    # and per-time attributes (e.g. the alternating energy tables of FPI)
    tdepnd_zvars = list(filter(lambda x: "time" in inp[x].dims, inp))
    mean_kwargs = {**kwargs, "average": "mean"}
    out_dict = {
        k: _resample_dataarray(inp[k], ref, **(kwargs if k == "data" else mean_kwargs))
        for k in tdepnd_zvars
    }

    # Complete the dictionary with non-time dependent zVaraiables
    ndepnd_zvars = list(filter(lambda x: x not in tdepnd_zvars, inp))
    out_dict = {**out_dict, **{k: inp[k] for k in ndepnd_zvars}}

    # The per-time attributes are resampled, the others kept, with the rule of
    # time_clip (_is_time_attr); the quality flags are left out (bits, neither
    # interpolated nor averaged)
    out_attrs = {}

    for k in sorted(inp.attrs):
        attr = inp.attrs[k]

        if k == "errorflags":
            continue

        if _is_time_attr(k, attr, len(inp.time.data)):
            coords = [np.arange(attr.shape[i + 1]) for i in range(attr.ndim - 1)]
            dims = [f"idx{i:d}" for i in range(attr.ndim - 1)]
            attr_ts = xr.DataArray(
                attr,
                coords=[inp.time.data, *coords],
                dims=["time", *dims],
            )
            out_attrs[k] = _resample_dataarray(attr_ts, ref, **mean_kwargs).data
        else:
            out_attrs[k] = attr

    # Make output Dataset
    out = xr.Dataset(out_dict, attrs=out_attrs)

    return out


def resample(
    inp,
    ref,
    method: str = "",
    f_s: float | None = None,
    window: int | None = None,
    thresh: float = 0,
    *,
    extrapolate: bool = True,
    average: str = "mean",
):
    r"""Resample inp to the time line of ref. If sampling of X is more than two
    times higher than Y, we average X, otherwise we interpolate X.

    Parameters
    ----------
    inp : xarray.DataArray or xarray.Dataset
        Time series to resample.
    ref : xarray.DataArray
        Reference time line.
    method : str, Optional
        Method of interpolation, a kind of ``scipy.interpolate.interp1d``:
        "linear", "nearest", "nearest-up", "zero", "slinear", "quadratic",
        "cubic", "previous" or "next", or "pchip" (the shape-preserving
        piecewise cubic Hermite interpolant, scipy PchipInterpolator, MATLAB
        interp1 "pchip"). If given, interpolate whatever the sampling.
        Default is "linear" when interpolating, as irf_resamp. The names are
        those of scipy, not of MATLAB: the MATLAB "spline" (the not-a-knot
        cubic spline) is "cubic" here, and the MATLAB "cubic" is "pchip".
    f_s : float, Optional
        Sampling frequency of the Y signal, 1/window. Default is guessed from
        the median time step of ref.
    window : float, Optional
        Length of the averaging window in seconds, 1/fsample.
    thresh : float, Optional
        Points farther than thresh * STD from the mean of an averaging window are
        disregarded (per component). Default is 0 (all points are used).
    extrapolate : bool, Optional
        Extrapolate the interpolation outside the time range of inp with the
        interpolant, as irf_resamp. False gives NaN at the reference times
        outside the time range of the samples (of the finite samples of each
        component with "quadratic", "cubic" or "pchip"). Default is True. Averages
        are NaN where a window has no samples in both cases.
    average : {"mean", "median", "max"}, Optional
        Statistic of the averaging windows, as the irf_resamp flags
        "median" and "max": "median" and "max" average whatever the sampling
        (no interpolation method then), after the thresh rejection as the
        mean. A NaN in a window gives NaN for the three (MATLAB max skips
        it). For a distribution, the data only; its energies, angles and
        per-time attributes are averaged with the mean. Default is "mean".

    Returns
    -------
    out : xarray.DataArray
        Resampled input to the reference time line using the selected method.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray or a xarray.Dataset, or if the times
        of inp and ref are not both datetime64 or both numeric (seconds).
    ValueError
        If method or average is not one of the values above, or average is
        "median" or "max" with a method.

    Notes
    -----
    As in irf_resamp:

    * Averaging uses the windows (t - 1/(2 f_s), t + 1/(2 f_s)] around each
      reference time t, so that each sample belongs to one window. A window
      without samples gives NaN, and a NaN in a window gives NaN. For
      datetime64 times the half window is rounded to the nanosecond, so a
      sample within 1 ns of an edge can fall in the other window than with
      irf_resamp's floating-point times (e.g., a relative difference of 3e-5
      of the wave numbers of `pyrfu.mms.fk_power_spectrum` on a 1 s burst
      interval).
    * Interpolation extrapolates outside the time range of inp with the
      interpolant (linearly with "linear"); ``extrapolate=False`` gives NaN
      there instead.
    * With "quadratic", "cubic" or "pchip", whose interpolant depends on all
      the samples, each component is interpolated through its finite
      samples, as MATLAB interp1 ("spline", "pchip"): NaN samples are
      skipped. With "linear" or "nearest", a NaN gives NaN at the times next
      to it, as MATLAB.

    The per-time attributes of a distribution (xarray.Dataset) are resampled
    as the data, except the FPI quality flags (``errorflags``), bits which
    are left out.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("e_gse_edp_fast_l2", tint, mms_id)

    Resample magnetic field to electric field sampling

    >>> b_xyz = pyrf.resample(b_xyz, e_xyz)

    """

    if not isinstance(inp, (xr.DataArray, xr.Dataset)):
        raise TypeError("inp must be a xarray.DataArray or xarray.Dataset")

    if method and method not in _METHODS:
        if method == "spline":
            hint = ' (MATLAB "spline" is "cubic")'
        else:
            hint = ""

        raise ValueError(
            f"method must be one of {', '.join(_METHODS)}, not {method!r}{hint}"
        )

    if average not in _AVERAGES:
        raise ValueError(
            f"average must be one of {', '.join(_AVERAGES)}, not {average!r}"
        )

    if average != "mean" and method:
        raise ValueError("cannot mix interpolation (method) and averaging (average)")

    # Both datetime64 or both numeric, checked before the times are cast to
    # datetime64[ns] (numeric times would be read as 1970 epochs)
    is_datetime = [np.issubdtype(x.time.dtype, np.datetime64) for x in [inp, ref]]

    if is_datetime[0] != is_datetime[1]:
        raise TypeError("inp and ref must both have datetime64 or numeric times")

    # The times in the same precision
    if is_datetime[0]:
        inp = inp.assign_coords(time=inp.time.astype("datetime64[ns]"))
        ref = ref.assign_coords(time=ref.time.astype("datetime64[ns]"))

    # Define options for resampling
    options = {
        "method": method,
        "f_s": f_s,
        "window": window,
        "thresh": thresh,
        "extrapolate": extrapolate,
        "average": average,
    }

    if isinstance(inp, xr.DataArray):
        out = _resample_dataarray(inp, ref, **options)
    else:
        out = _resample_dataset(inp, ref, **options)

    return out
