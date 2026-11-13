#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from pycdfpp import load

# Local imports
from ..pyrf.time_clip import _time_interval, time_clip
from ..pyrf.ts_skymap import ts_skymap
from .get_ts import _get_epochs
from .get_variable import _pycdfpp_attributes_to_dict


def _fpi_name(cdf_name, quantity):
    # Name of the variable of the quantity of the FPI distribution cdf_name,
    # e.g. mms1_dis_energy0_brst for mms1_dis_dist_brst
    parts = cdf_name.split("_")
    return "_".join([parts[0], parts[1], quantity, parts[-1]])


def _first_table(energy, step_table, parity):
    # Energy table of the first record of the stepping table parity, or of the
    # first record if no record has it (e.g. a file of a single record)
    idx = np.flatnonzero(step_table == parity)
    return energy[idx[0] if idx.size else 0, :].copy()


def get_dist(file_path, cdf_name, tint: list = None):
    r"""Read field named cdf_name in file and convert to velocity distribution
    function.

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
    out : xarray.Dataset
        Time series of the velocity distribution function if the target
        specie in the selected time interval. The times are the centres of
        the accumulations, whose half widths (s) are the attributes
        ``delta_time_minus`` and ``delta_time_plus`` (time,), as the
        irfu-matlab ``dt_minus`` and ``dt_plus``. The energy widths of the
        channels are ``delta_energy_minus`` and ``delta_energy_plus``, when
        the file has them.

    """

    tmmode = cdf_name.split("_")[-1]

    if "_dis_" in cdf_name:
        specie = "ions"
    elif "_des_" in cdf_name:
        specie = "electrons"
    else:
        raise AttributeError(
            "Couldn't get the particle species from file name!!",
        )

    # Time interval, in the forms of time_clip (a tuple, datetime.datetime
    # and a time series raised TypeError), checked before reading the file
    if tint is None:
        tint = ["1995-10-06T18:50:00.000000000", "2200-10-06T18:50:00.000000000"]

    tint = _time_interval(tint)

    # Load CDF file
    file = load(file_path)

    # Get the relevant CDF file information (zVariables)
    z_vars = [z_var[0] for z_var in file.items()]

    # Get the global attributes
    glob_attrs = _pycdfpp_attributes_to_dict(file.attributes)
    glob_attrs = {**glob_attrs, **{"tmmode": tmmode, "species": specie}}

    # Get VDF zVariable attributes
    dist_attrs = _pycdfpp_attributes_to_dict(file[cdf_name].attributes)

    # Get CDF keys to Epoch, energy, azimuthal and elevation angle
    # zVariables
    depends_keys = [dist_attrs[f"DEPEND_{i:d}"] for i in range(4)]

    # Get coordinates attributes
    coords_attrs = {}

    for n, k in zip(["time", "phi", "theta", "energy"], depends_keys):
        coords_attrs[n] = _pycdfpp_attributes_to_dict(file[k].attributes)

    epochs = _get_epochs(file, cdf_name)
    times = epochs["data"]

    # (time, energy, phi, theta)
    dist = np.transpose(file[cdf_name].values, [0, 3, 1, 2])
    _, n_energy, n_phi, n_theta = dist.shape

    # Tables of one record (non-record-varying) or of all records; reshaped
    # rather than squeezed, which made those of a single record 1-D
    phi = file[depends_keys[1]].values.reshape(-1, n_phi)
    theta = file[depends_keys[2]].values.reshape(n_theta)
    energy = file[depends_keys[3]].values.reshape(-1, n_energy)

    if times.size == 0:
        # File without records: empty skymap, without energy tables
        phi = np.zeros((0, n_phi))
        energy = np.zeros((0, n_energy))
        energy0 = np.full(n_energy, np.nan)
        energy1 = np.full(n_energy, np.nan)
        step_table = np.zeros(0)

    elif tmmode == "brst":
        e_step_table_name = _fpi_name(cdf_name, "steptable_parity")
        en0_name = _fpi_name(cdf_name, "energy0")
        en1_name = _fpi_name(cdf_name, "energy1")

        step_table = file[e_step_table_name].values

        if en0_name in z_vars:
            energy0 = file[en0_name].values.reshape(n_energy)
            energy1 = file[en1_name].values.reshape(n_energy)
        else:
            energy = np.broadcast_to(energy, (len(times), n_energy))
            energy0 = _first_table(energy, step_table, 0)
            energy1 = _first_table(energy, step_table, 1)

        # Overwrite energy to make sure that energy0 and energy1
        # are used instead
        energy = np.tile(energy0, (len(step_table), 1))
        energy[step_table == 1] = np.tile(energy1, (int(np.sum(step_table)), 1))

    elif tmmode == "fast":
        # No stepping: a single energy table (the energies of the first
        # record; the records of the files have the same)
        energy = np.broadcast_to(energy, (len(times), n_energy)).copy()
        energy0 = energy[0, :].copy()
        energy1 = energy[0, :].copy()
        step_table = np.zeros(len(times))

    else:
        raise ValueError("Invalid sampling mode!!")

    if phi.shape[0] == 1:
        phi = phi[0, :]

    d_en_name = _fpi_name(cdf_name, "energy_delta")

    # Energy widths, left out of the files without them (they were None, which
    # the routines that check for the attribute, e.g. reduce, took as given)
    if d_en_name in z_vars:
        glob_attrs["delta_energy_plus"] = file[d_en_name].values
        glob_attrs["delta_energy_minus"] = file[d_en_name].values

    # Half widths of the accumulation (s) around the (centred) times, as
    # irfu-matlab dt_minus and dt_plus
    if "half_width" in epochs:
        half_width = np.full(len(times), epochs["half_width"])
        glob_attrs["delta_time_minus"] = half_width
        glob_attrs["delta_time_plus"] = half_width.copy()

    out = ts_skymap(
        times,
        dist,
        energy,
        phi,
        theta,
        energy0=energy0,
        energy1=energy1,
        esteptable=step_table,
        attrs=dist_attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )

    out = time_clip(out, tint)

    return out
