#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import pycdfpp
import xarray as xr

# CDF time types, decoded to datetime64 with decode_epochs
_EPOCH_TYPES = (
    pycdfpp.DataType.CDF_TIME_TT2000,
    pycdfpp.DataType.CDF_EPOCH,
    pycdfpp.DataType.CDF_EPOCH16,
)


def read_cdf(path: str, *, decode_epochs: bool = False, squeeze: bool = True) -> dict:
    r"""Reads a .cdf file and returns a dictionary with the fields contained in
    the file.

    Parameters
    ----------
    path : str
        Path to the .cdf file.
    decode_epochs : bool, Optional
        Decode the CDF times (TT2000, EPOCH and EPOCH16 variables) to
        datetime64[ns], as `pyrfu.mms.get_ts`. Default is False, the times as
        stored.
    squeeze : bool, Optional
        Remove the dimensions of size 1 (``numpy.squeeze``). Default is True;
        False keeps the shape as stored (records first, e.g. (n, 1) for a
        column, (1, n) for a non-record-varying variable).

    Returns
    -------
    dict
        Hash table with fields contained in the .cdf file, with their
        attributes (``UNITS``, ...).

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be read as a CDF file.

    Notes
    -----
    The variables are returned with lower-case names, by default as stored:
    the epochs not decoded (TT2000 nanoseconds, see ``decode_epochs``), the
    dimensions of size 1 removed (see ``squeeze``), and the dimensions
    numbered (``dim_0``, ``dim_1``, ...) without the CDF attributes.

    """

    # Local import: pyrfu.mms imports pyrfu.pyrf
    # pylint: disable-next=import-outside-toplevel
    from ..mms.get_variable import _pycdfpp_attributes_to_dict

    # Initialize output dictionary
    out_dict = {}

    # Load file: pycdfpp returns None for a missing or unreadable file, or
    # raises (FileNotFoundError, or ValueError for a file that is not a CDF)
    # in recent versions (0.18)
    try:
        file = pycdfpp.load(path)
    except (FileNotFoundError, ValueError):
        file = None

    if file is None:
        raise FileNotFoundError(f"cannot read the CDF file {path}")

    # Get keys (a.k.a zvariables) from file
    keys = list(map(lambda x: x[0], file.items()))

    for key in keys:
        # Get data (times decoded, size-1 dimensions removed, if asked) and
        # coordinates
        data = file[key].values

        if decode_epochs and file[key].type in _EPOCH_TYPES:
            data = pycdfpp.to_datetime64(data)

        if squeeze:
            data = np.squeeze(data)

        coords = [np.arange(dim_size) for dim_size in data.shape]

        # Construct xarray DataArray, with the attributes of the variable
        out_dict[key.lower()] = xr.DataArray(
            data, coords=coords, attrs=_pycdfpp_attributes_to_dict(file[key].attributes)
        )

    return out_dict
