#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import pycdfpp
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def read_cdf(path: str) -> dict:
    r"""Reads a .cdf file and returns a dictionary with the fields contained in
    the file.

    Parameters
    ----------
    path : str
        Path to the .cdf file.

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
    The variables are returned as stored, with lower-case names: the epochs
    are not decoded (TT2000 nanoseconds, see ``pycdfpp.to_datetime64``), the
    dimensions of size 1 are removed (``numpy.squeeze``), and the dimensions
    are numbered (``dim_0``, ``dim_1``, ...) without the CDF attributes.

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
        # Get data and coordinates
        data = np.squeeze(file[key].values)
        coords = [np.arange(dim_size) for dim_size in data.shape]

        # Construct xarray DataArray, with the attributes of the variable (they
        # were not read)
        out_dict[key.lower()] = xr.DataArray(
            data, coords=coords, attrs=_pycdfpp_attributes_to_dict(file[key].attributes)
        )

    return out_dict
