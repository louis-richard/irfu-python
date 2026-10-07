#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import warnings

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import interpolate

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def _guess_sampling_frequency(ref_time):
    r"""Sampling frequency of the time line (in seconds), from the median time
    step (robust to an irregular first interval)."""
    d_t = np.median(np.diff(ref_time))

    if not np.isfinite(d_t) or d_t <= 0:
        raise RuntimeError("Cannot guess the sampling frequency of the reference")

    return 1 / d_t


def _average(inp_time, inp_data, ref_time, thresh, dt2):
    r"""Average inp_data in the windows (ref_time - dt2, ref_time + dt2] (as in
    irf_resamp), with inp_time, ref_time and dt2 in the same units. A window
    without samples, or with a NaN, gives NaN. With thresh, the points farther
    than thresh * std from the mean of the window are disregarded.
    """
    inp_data = np.asarray(inp_data, dtype=np.float64)
    shape = inp_data.shape[1:]
    data = inp_data.reshape(len(inp_data), -1)

    # Samples of each window: inp_time[idx_l:idx_r]
    idx_l = np.searchsorted(inp_time, ref_time - dt2, side="right")
    idx_r = np.searchsorted(inp_time, ref_time + dt2, side="right")
    n_pts = idx_r - idx_l

    if not thresh:
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

            # Standard deviation as MATLAB's std (N - 1; 0 for one sample); NaN
            # if the window contains a NaN
            std_ = np.std(window, axis=0, ddof=1) if len(window) > 1 else 0 * window[0]
            mean_ = np.mean(window, axis=0)
            keep = np.abs(window - mean_) <= thresh * std_

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out_data[i] = np.nanmean(np.where(keep, window, np.nan), axis=0)

    return out_data.reshape(len(ref_time), *shape)


def _resample_dataarray(inp, ref, method, f_s, window, thresh, verbose=False):
    r"""Resample for time series (xarray.DataArray)"""

    flag_do = "check"

    if method:
        flag_do = "interpolation"

    if f_s is not None:
        sfy = f_s
    elif window is not None:
        sfy = 1 / window
    else:
        sfy = None

    if np.issubdtype(inp.time.dtype, np.datetime64):
        if not np.issubdtype(ref.time.dtype, np.datetime64):
            raise TypeError("inp and ref must both have datetime64 or numeric times")

        # Integer nanoseconds for the averaging windows, float seconds otherwise
        inp_time_ns = inp.time.data.astype("datetime64[ns]").astype(np.int64)
        ref_time_ns = ref.time.data.astype("datetime64[ns]").astype(np.int64)
        inp_time = (inp_time_ns - inp_time_ns[0]) * 1e-9
        ref_time = (ref_time_ns - inp_time_ns[0]) * 1e-9
    elif np.issubdtype(inp.time.dtype, np.number) and np.issubdtype(
        ref.time.dtype, np.number
    ):
        # Numeric times are in seconds
        inp_time = inp_time_ns = np.asarray(inp.time.data, dtype=np.float64)
        ref_time = ref_time_ns = np.asarray(ref.time.data, dtype=np.float64)
    else:
        raise TypeError("inp and ref must both have datetime64 or numeric times")

    if flag_do == "check":
        if len(ref_time) > 1:
            if not sfy:
                sfy = _guess_sampling_frequency(ref_time)

            if len(inp_time) / (inp_time[-1] - inp_time[0]) > 2 * sfy:
                flag_do = "average"
                if verbose:
                    logger.info("Using averages in resample")
            else:
                flag_do = "interpolation"
        else:
            flag_do = "interpolation"

    assert flag_do in ["average", "interpolation"]

    if flag_do == "average":
        assert not method, "cannot mix interpolation and averaging flags"

        if not sfy:
            sfy = _guess_sampling_frequency(ref_time)

        if np.issubdtype(inp.time.dtype, np.datetime64):
            dt2 = int(np.round(0.5e9 / sfy))  # half window in ns
        else:
            dt2 = 0.5 / sfy

        out_data = _average(inp_time_ns, inp.data, ref_time_ns, thresh, dt2)

    else:
        if not method:
            method = "linear"

        # If time series agree, no interpolation is necessary.
        if len(inp_time_ns) == len(ref_time_ns) and np.array_equal(
            inp_time_ns, ref_time_ns
        ):
            out_data = inp.data.copy()
            coord = [ref.coords["time"].data]

            if len(inp.coords) > 1:
                for k in list(inp.dims)[1:]:
                    coord.append(inp.coords[k].data)

            out = xr.DataArray(
                out_data,
                coords=coord,
                dims=inp.dims,
                attrs=inp.attrs,
            )

            return out

        # Linear extrapolation outside the time range of inp, as irf_resamp
        tck = interpolate.interp1d(
            inp_time,
            inp.data,
            kind=method,
            axis=0,
            fill_value="extrapolate",
        )
        out_data = tck(ref_time)

    coord = [ref.coords["time"]]

    if len(inp.coords) > 1:
        for k in list(inp.dims)[1:]:
            coord.append(inp.coords[k].data)

    out = xr.DataArray(out_data, coords=coord, dims=inp.dims, attrs=inp.attrs)

    return out


def _resample_dataset(inp, ref, **kwargs):
    r"""Resample for VDFs (xarray.Dataset)"""

    # Find time dependent zVariables and resample
    tdepnd_zvars = list(filter(lambda x: "time" in inp[x].dims, inp))
    out_dict = {k: _resample_dataarray(inp[k], ref, **kwargs) for k in tdepnd_zvars}

    # Complete the dictionary with non-time dependent zVaraiables
    ndepnd_zvars = list(filter(lambda x: x not in tdepnd_zvars, inp))
    out_dict = {**out_dict, **{k: inp[k] for k in ndepnd_zvars}}

    # Find array_like attributes
    arr_attrs = filter(
        lambda x: isinstance(inp.attrs[x], np.ndarray),
        inp.attrs,
    )
    arr_attrs = list(arr_attrs)

    # Initialize attributes dictionary with non array_like attributes
    gen_attrs = filter(lambda x: x not in arr_attrs, inp.attrs)
    out_attrs = {k: inp.attrs[k] for k in list(gen_attrs)}

    for k in arr_attrs:
        attr = inp.attrs[k]

        # If array_like attributes have one dimension equal to time length
        # assume time dependent. One option would be move the time dependent
        # array_like attributes to time series to zVaraibles to avoid
        # confusion
        if attr.shape[0] == len(inp.time.data):
            coords = [np.arange(attr.shape[i + 1]) for i in range(attr.ndim - 1)]
            dims = [f"idx{i:d}" for i in range(attr.ndim - 1)]
            attr_ts = xr.DataArray(
                attr,
                coords=[inp.time.data, *coords],
                dims=["time", *dims],
            )
            out_attrs[k] = _resample_dataarray(attr_ts, ref, **kwargs).data
        else:
            out_attrs[k] = attr

    out_attrs = {k: out_attrs[k] for k in sorted(out_attrs)}

    # Make output Dataset
    out = xr.Dataset(out_dict, attrs=out_attrs)

    return out


def resample(
    inp,
    ref,
    method: str = "",
    f_s: float = None,
    window: int = None,
    thresh: float = 0,
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
        Method of interpolation "spline", "linear" etc.
        (default "linear") if method is given then interpolate
        independent of sampling.
    f_s : float, Optional
        Sampling frequency of the Y signal, 1/window. Default is guessed from
        the median time step of ref.
    window : float, Optional
        Length of the averaging window in seconds, 1/fsample.
    thresh : float, Optional
        Points farther than thresh * STD from the mean of an averaging window are
        disregarded (per component). Default is 0 (all points are used).

    Returns
    -------
    out : xarray.DataArray
        Resampled input to the reference time line using the selected method.

    Raises
    ------
    TypeError
        If the times of inp and ref are not both datetime64 or both numeric
        (seconds).

    Notes
    -----
    As in irf_resamp:

    * Averaging uses the windows (t - 1/(2 f_s), t + 1/(2 f_s)] around each
      reference time t, so that each sample belongs to one window. A window
      without samples gives NaN, and a NaN in a window gives NaN.
    * Interpolation extrapolates linearly outside the time range of inp: clip
      ref to the time range of inp (e.g., with time_clip) if this is not wanted.

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

    message = "Invalid input type. Input must be xarray.DataArary or xarray.Dataset"
    assert isinstance(inp, (xr.DataArray, xr.Dataset)), message

    # Fix make sure that the time are in the same precision format

    if np.issubdtype(inp.time.dtype, np.datetime64):

        inp = inp.assign_coords(time=inp.time.astype("datetime64[ns]"))
        ref = ref.assign_coords(time=ref.time.astype("datetime64[ns]"))

    # Define options for resampling
    options = {"method": method, "f_s": f_s, "window": window, "thresh": thresh}

    if isinstance(inp, xr.DataArray):
        out = _resample_dataarray(inp, ref, **options)
    else:
        out = _resample_dataset(inp, ref, **options)

    return out
