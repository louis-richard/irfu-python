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


def _energy_parity(energy):
    # Stepping table parity of each record from its energies, for the files
    # without the parity variable (the DES distaux files): 0 for the table of
    # the first record, 1 for the other one
    other = np.any(energy != energy[0], axis=1)

    if np.any(energy[other] != energy[other][:1]):
        raise ValueError("more than two energy tables without a stepping parity")

    return other.astype(np.uint8)


def _species(cdf_name):
    # Particle species of the FPI variable cdf_name (dis or des)
    if "_dis_" in cdf_name:
        species = "ions"
    elif "_des_" in cdf_name:
        species = "electrons"
    else:
        raise AttributeError(
            "Couldn't get the particle species from file name!!",
        )

    return species


def _burst_tables(file, cdf_name, z_vars, energy, n_times):
    # Energies of each record, energy tables and stepping table parity of a
    # burst file: the parity of the file or, without it (the DES distaux
    # files), of the energies; the tables of the file (energy0, energy1) or
    # of the first record of each parity
    n_energy = energy.shape[1]
    step_table_name = _fpi_name(cdf_name, "steptable_parity")
    en0_name = _fpi_name(cdf_name, "energy0")
    en1_name = _fpi_name(cdf_name, "energy1")

    if step_table_name in z_vars:
        step_table = file[step_table_name].values
    else:
        step_table = _energy_parity(np.broadcast_to(energy, (n_times, n_energy)))

    if en0_name in z_vars:
        energy0 = file[en0_name].values.reshape(n_energy)
        energy1 = file[en1_name].values.reshape(n_energy)
    else:
        energy = np.broadcast_to(energy, (n_times, n_energy))
        energy0 = _first_table(energy, step_table, 0)
        energy1 = _first_table(energy, step_table, 1)

    # Overwrite energy to make sure that energy0 and energy1 are used instead
    energy = np.tile(energy0, (len(step_table), 1))
    energy[step_table == 1] = np.tile(energy1, (int(np.sum(step_table)), 1))

    return energy, energy0, energy1, step_table


def _tables(file, cdf_name, z_vars, tmmode, phi, energy, n_times):
    # Azimuths, energies of each record, energy tables and stepping table
    # parity of the file, from the tables of one record (non-record-varying)
    # or of all records
    n_phi, n_energy = phi.shape[1], energy.shape[1]

    if n_times == 0:
        # File without records: empty skymap, without energy tables
        phi = np.zeros((0, n_phi))
        energy = np.zeros((0, n_energy))
        energy0 = np.full(n_energy, np.nan)
        energy1 = np.full(n_energy, np.nan)
        step_table = np.zeros(0)
    elif tmmode == "brst":
        energy, energy0, energy1, step_table = _burst_tables(
            file, cdf_name, z_vars, energy, n_times
        )
    elif tmmode == "fast":
        # No stepping: a single energy table (the energies of the first
        # record; the records of the files have the same)
        energy = np.broadcast_to(energy, (n_times, n_energy)).copy()
        energy0 = energy[0, :].copy()
        energy1 = energy[0, :].copy()
        step_table = np.zeros(n_times)
    else:
        raise ValueError("Invalid sampling mode!!")

    if phi.shape[0] == 1:
        phi = phi[0, :]

    return phi, energy, energy0, energy1, step_table


def _per_time(file, name, z_vars, n_times, per_file=False):
    # Values of the variable name, one per record, or None if the file has
    # none or another number of values; with per_file, one value
    # (non-record-varying) is that of all the records
    if name not in z_vars or not n_times:
        return None

    values = np.asarray(file[name].values).ravel()

    if per_file and values.size == 1:
        values = np.full(n_times, values[0], dtype=values.dtype)

    if values.size != n_times:
        return None

    return values


def _file_attrs(file, cdf_name, z_vars, n_times):
    # Energy widths, compression indicator and quality flags of the file,
    # those it has
    attrs = {}
    d_en_name = _fpi_name(cdf_name, "energy_delta")

    # Energy widths, left out (not None) for the files without them, as the
    # routines, e.g. reduce, check for the attribute
    if d_en_name in z_vars:
        attrs["delta_energy_plus"] = file[d_en_name].values
        attrs["delta_energy_minus"] = file[d_en_name].values

    # Compression indicator of each distribution: 0 lossless, 1 lossy; one
    # per record in burst, one per file (non-record-varying, "at survey-start
    # time") in fast survey, then for all its distributions. Quality flags of
    # each distribution (32 bits, 0 without quality errors)
    for key, per_file in [("compressionloss", True), ("errorflags", False)]:
        name = _fpi_name(cdf_name, key)
        values = _per_time(file, name, z_vars, n_times, per_file)

        if values is not None:
            attrs[key] = values

    return attrs


def get_dist(file_path, cdf_name, tint: list | None = None):
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
        the file has them. The attribute ``compressionloss`` (time,) is the
        compression indicator of each distribution of the file: 0 lossless, 1
        lossy, when the file has it: one per distribution in burst mode, the
        indicator of the file for all its distributions in fast survey
        (`pyrfu.pyrf.resample` interpolates it as the other per-time
        attributes, to a fraction between 0 and 1). The attribute
        ``errorflags`` (time,) holds the quality flags of each distribution,
        when the file has them: 0 without quality errors, else bit 0 a
        manually flagged interval, bit 1 overcounting or saturation likely,
        bit 2 a compression pipeline error (``VAR_NOTES`` of the files;
        4294967295 is the fill value).

    Raises
    ------
    ValueError
        If a burst file without the stepping table parity (the DES
        ``distaux`` files) has more than two energy tables.

    Notes
    -----
    In burst mode, the energy table of each distribution is given by the
    stepping table parity of the file (``steptable_parity``). The files
    without it (the DES ``distaux`` files of the modelled photoelectron
    distribution, ``disterr_photo``) take it from the energies of each
    distribution: 0 for the table of the first distribution, 1 for the
    other one.

    The FPI distributions are compressed on board (CCSDS 122.0 wavelet
    compression, Barrie et al., 2017 [1]_). The fast survey distributions, the
    sums of 4.5 s of burst distributions, are compressed 16.7 times and are
    always lossy (``compressionloss`` 1): their counts are not Poisson where
    they are low, and do not go below a floor of about 2 counts per bin
    ("noise floor" of Barrie et al., 2017; artifacts of 2 to 3 counts, da
    Silva et al., 2020 [2]_). On 2017-07-11 (MMS3 electrons, plasma sheet), the
    fast survey omni-directional spectrum is 1.7 to 2.8 times the burst one
    at 21 and 27.5 keV, where the true counts are below 2 per bin and 4.5 s.
    In five plasma-sheet intervals of 2017 (MMS3 electrons), the fast survey
    density is within 1 % of that of the burst distributions averaged over
    the same 4.5 s, the temperature within about 1 % (6.5 % in a tenuous, hot
    interval), the heat flux 7 to 23 % higher (median) and the bulk velocity
    has a spin tone of 20 to 120 km/s.
    Most burst distributions are lossless since late 2016 (about 95 % of
    them, da Silva et al., 2020). The counts, errors and one-count levels of
    the low-count bins (`vdf_counts`, `vdf_error`, `vdf_one_count`,
    `psd_moments_err`) are then those of the burst distributions.

    References
    ----------
    .. [1] Barrie, A. C., et al. (2017), Performance of a space-based wavelet
       compressor for plasma count data on the MMS Fast Plasma Investigation,
       J. Geophys. Res. Space Physics, 122, 765-779,
       doi:10.1002/2016JA022645.
    .. [2] da Silva, D., et al. (2020), Neural network repair of lossy
       compression artifacts in the September 2015 to March 2016 duration of
       the MMS/FPI data set, J. Geophys. Res. Space Physics, 125,
       e2019JA027181, doi:10.1029/2019JA027181.

    """

    tmmode = cdf_name.split("_")[-1]
    specie = _species(cdf_name)

    # Time interval (forms of time_clip), checked before reading the file
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

    # Tables of one record (non-record-varying) or of all records, reshaped
    # (squeeze would make those of a single record 1-D)
    phi = file[depends_keys[1]].values.reshape(-1, n_phi)
    theta = file[depends_keys[2]].values.reshape(n_theta)
    energy = file[depends_keys[3]].values.reshape(-1, n_energy)

    phi, energy, energy0, energy1, step_table = _tables(
        file, cdf_name, z_vars, tmmode, phi, energy, len(times)
    )

    glob_attrs = {**glob_attrs, **_file_attrs(file, cdf_name, z_vars, len(times))}

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
