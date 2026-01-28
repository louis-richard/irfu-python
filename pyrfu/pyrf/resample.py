#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import logging

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import interpolate

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def _guess_sampling_frequency(ref_time):
    r"""Compute sampling frequency of the time line."""

    n_data = len(ref_time)

    sfy1 = 1 / (ref_time[1] - ref_time[0])
    sfy = None
    not_found = True

    if n_data == 2:
        sfy = sfy1
        not_found = False

    cur, max_try = [2, 10]

    while not_found and cur < n_data and cur - 3 < max_try:
        sfy = 1 / (ref_time[cur] - ref_time[cur - 1])

        if np.absolute(sfy - sfy1) < sfy * 0.001:
            not_found = False

            sfy = (sfy + sfy1) / 2
            break

        sfy = sfy1
        cur += 1

    if not_found:
        raise RuntimeError(f"Cannot guess sampling frequency. Tried {max_try:d} times")

    return sfy


def _average(inp_time, inp_data, ref_time, thresh, dt2):
    r"""Resample inp_data to timeline of ref_time, using half-window of dt2.
    Points above std*tresh are excluded. thresh=0 turns off this option.
    """

    try:
        out_data = np.zeros([len(ref_time), *inp_data.shape[1:]])
    except IndexError:
        inp_data = inp_data[:, None]
        out_data = np.zeros((len(ref_time), inp_data.shape[1]))

    for i, ref_t in enumerate(ref_time):
        idx_l = bisect.bisect_left(inp_time, ref_t - dt2)
        idx_r = bisect.bisect_right(inp_time, ref_t + dt2)

        idx = np.arange(idx_l, idx_r)

        if idx.size == 0:
            out_data[i, ...] = np.nan
        else:
            if thresh:
                std_ = np.std(inp_data[idx, ...], axis=0)
                mean_ = np.mean(inp_data[idx, ...], axis=0)

                assert any(np.isnan(std_))

                for j, stdd in enumerate(std_):
                    if not np.isnan(stdd):
                        idx_r = bisect.bisect_right(
                            inp_data[idx, j + 1] - mean_[j],
                            thresh * stdd,
                        )
                        if idx_r:
                            out_data[i, j + 1] = np.mean(
                                inp_data[idx[idx_r], j + 1],
                                axis=0,
                            )
                        else:
                            out_data[i, j + 1] = np.nan
                    else:
                        out_data[i, ...] = np.nan

            else:
                out_data[i, ...] = np.mean(inp_data[idx, ...], axis=0)

    if out_data.ndim > 1 and out_data.shape[1] == 1:
        out_data = out_data[:, 0]

    return out_data


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

    inp_time = inp.time.data.view("i8") * 1e-9
    ref_time = ref.time.data.view("i8") * 1e-9

    if flag_do == "check":
        if len(ref_time) > 1:
            if not sfy:
                sfy = _guess_sampling_frequency(ref_time)

            if len(inp_time) / (inp_time[-1] - inp_time[0]) > 2 * sfy:
                flag_do = "average"
                if verbose:
                    logging.info("Using averages in resample")
            else:
                flag_do = "interpolation"
        else:
            flag_do = "interpolation"

    assert flag_do in ["average", "interpolation"]

    if flag_do == "average":
        assert not method, "cannot mix interpolation and averaging flags"

        if not sfy:
            sfy = _guess_sampling_frequency(ref_time)

        out_data = _average(inp_time, inp.data, ref_time, thresh, 0.5 / sfy)

    else:
        if not method:
            method = "linear"

        # If time series agree, no interpolation is necessary.
        if len(inp_time) == len(ref_time) and all(inp_time == ref_time):
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
        Sampling frequency of the Y signal, 1/window.
    window : int or float or ndarray, Optional
        Length of the averaging window, 1/fsample.
    thresh : float, Optional
        Points above STD*THRESH are disregarded for averaging

    Returns
    -------
    out : xarray.DataArray
        Resampled input to the reference time line using the selected method.


    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field and electric field

    >>> b_xyz = mms.get_data("e_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("e_gse_edp_fast_l2", tint, mms_id)

    Resample magnetic field to electric field sampling

    >>> b_xyz = pyrf.resample(b_xyz, e_xyz)

    """

    message = "Invalid input type. Input must be xarray.DataArary or xarray.Dataset"
    assert isinstance(inp, (xr.DataArray, xr.Dataset)), message

    # Fix make sure that the time are in the same precision format
    inp = inp.assign_coords(time=inp.time.astype("datetime64[ns]"))
    ref = ref.assign_coords(time=ref.time.astype("datetime64[ns]"))

    # Define options for resampling
    options = {"method": method, "f_s": f_s, "window": window, "thresh": thresh}

    if isinstance(inp, xr.DataArray):
        out = _resample_dataarray(inp, ref, **options)
    else:
        out = _resample_dataset(inp, ref, **options)

    return out
