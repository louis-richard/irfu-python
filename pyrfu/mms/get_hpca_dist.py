#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .hpca_pad import _hpca_elevations

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

mass_and_charge = {
    "hydrogen+": [1.04535e-2, 1],
    "helium+": [4.18138e-2, 1],
    "helium++": [4.18138e-2, 2],
    "oxygen+": [0.167255, 1],
    "oxygen++": [0.167255, 2],
}


def _half_spins(inp, azimuth):
    r"""First sample of inp of each half-spin of azimuth, and whether the
    half-spin is complete in inp with non-zero azimuths."""
    t_inp = inp.time.data.astype("datetime64[ns]")
    t_azi = (azimuth.time.data.astype("datetime64[ns]") - t_inp[0]).astype(np.int64)
    t_inp = (t_inp - t_inp[0]).astype(np.int64)
    n_az = azimuth.shape[1]
    d_t = np.median(np.diff(t_inp))

    # The azimuth times are the centres of the half-spins, 7.5 samples after
    # their first samples (times shifted to the middle of the accumulations, as
    # by get_ts and get_data), or their starts (Epoch_Angles in the files)
    for offset in [(n_az - 1) / 2 * d_t, 0.0]:
        t_first = t_azi - offset
        i_start = np.searchsorted(t_inp, t_first - d_t / 4)
        i_first = np.minimum(i_start, len(t_inp) - 1)
        complete = np.abs(t_inp[i_first] - t_first) < d_t / 4

        if np.any(complete):
            break

    i_last = np.minimum(i_start + n_az - 1, len(t_inp) - 1)
    complete &= i_start + n_az <= len(t_inp)
    complete &= np.abs(t_inp[i_last] - t_inp[i_first] - (n_az - 1) * d_t) < d_t / 2

    # Half-spins without azimuths (zeros), from the first energy and anode
    complete &= np.any(azimuth.data[:, :, 0, 0] != 0, axis=1)

    return i_start, complete


def get_hpca_dist(
    inp,
    azimuth,
    elevation=None,
    source: str = "default",
    data_path: str = "",
):
    r"""Returns pseudo-3D particle data structures containing mms hpca data
    for use with spd_slice2d, as SPEDAS mms_get_hpca_dist.

    Parameters
    ----------
    inp : xarray.DataArray
        HPCA ion spec; [nt, npo16, ner63].
    azimuth : xarray.DataArray
        Azimuthal angles per energy; [nT, naz16, npo16, ner63], one record per
        half-spin, at its centre (as :func:`pyrfu.mms.get_data` gives it) or at
        its start (Epoch_Angles in the files).
    elevation : array_like, Optional
        Polar angles (colatitudes, deg) of the 16 anodes. Default reads
        mms?_hpca_centroid_elevation_angle in the file of `inp`.
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to read the anode elevations from (if `elevation` is None).
        Default uses default in `pyrfu/mms/config.json`.
    data_path : str, Optional
        Path of MMS data (for `source` "local").

    Returns
    -------
    out : dict
        HPCA distribution: one distribution per half-spin of `azimuth` (data,
        phi and dphi of shape [nT, ner63, naz16, npo16]), with the times of
        `azimuth` (time), NaN for the half-spins not complete in `inp` or
        without azimuths.

    Raises
    ------
    ValueError
        If the times of `azimuth` are not monotonic, or if no half-spin is
        complete in `inp` with azimuths.

    Notes
    -----
    The data of a half-spin are the naz16 samples of `inp` from its first
    sample (start azimuth 0). They were shifted by one sample (as in SPEDAS):
    each sample had the azimuth of the next one, and the first sample was the
    last of the previous half-spin. The elevations were the anode numbers of
    `inp`.

    """

    # check if the time series is monotonic to avoid doing incorrect
    # calculations when there's a problem with the CDF files
    time_data = azimuth.time.data

    if np.any(time_data[1:] <= time_data[:-1]):
        raise ValueError("non-monotonic data found in the Epoch_Angles")

    i_start, complete = _half_spins(inp, azimuth)

    if not np.any(complete):
        raise ValueError(
            "no complete half-spin of inp with azimuths: the times of azimuth "
            "must be the centres or the starts of the half-spins of inp"
        )

    if elevation is None:
        elevation = _hpca_elevations(inp, source, data_path)

    elevation = np.asarray(elevation, dtype=np.float64).ravel()

    # final dimensions for a single distribution (energy-azimuth-elevation)
    n_half, n_az, n_po, n_en = azimuth.shape
    dim = (n_en, n_az, n_po)

    specie = inp.attrs["FIELDNAM"].split(" ")[0].lower()

    mass, charge = mass_and_charge[specie]

    # energy bins and elevations are constant
    out_energy = np.broadcast_to(inp.ccomp.data[:, None, None], dim).copy()
    out_theta = np.broadcast_to(90.0 - elevation[None, None, :], dim).copy()

    # azimuths, from half-spin-azimuth-elevation-energy
    out_phi = np.transpose(azimuth.data, [0, 3, 1, 2]).astype(np.float64)

    # median step between subsequent azimuths of each half-spin (median to
    # discard the step across 0 = 360)
    out_dphi = np.median(np.diff(azimuth.data, axis=1), axis=1)
    out_dphi = np.transpose(out_dphi, [0, 2, 1])[:, :, None, :]
    out_dphi = np.broadcast_to(out_dphi, out_phi.shape).copy()

    out_data = np.full(out_phi.shape, np.nan)

    for i in np.flatnonzero(complete):
        out_data[i] = np.transpose(inp.data[i_start[i] : i_start[i] + n_az], [2, 0, 1])

    out_phi[~complete] = np.nan
    out_dphi[~complete] = np.nan

    out = {
        "time": time_data,
        "data": out_data,
        "bins": np.ones(dim),
        "theta": out_theta,
        "phi": out_phi,
        "energy": out_energy,
        "dtheta": np.full(dim, 22.5),
        "dphi": out_dphi,
        "denergy": np.zeros(dim),
        "n_energy": n_en,
        "n_theta": n_po,
        "n_phi": n_az,
        "n_times": n_half,
        "project_name": "MMS",
        "species": specie,
        "charge": charge,
        "units_name": "df_cm",
        "mass": mass,
    }

    return out
