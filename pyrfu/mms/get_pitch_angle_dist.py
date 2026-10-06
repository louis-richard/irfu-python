#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.pyrf.normalize import normalize
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.time_clip import time_clip

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def _bin_reduce(data0, finite, in_bin, method, weights):
    r"""Mean (over theta, then phi, as irfu-matlab), sum, or solid-angle
    weighted mean of the samples in one pitch-angle bin, (time, energy)."""
    in_bin = in_bin.astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method == "mean":
            row_sum = np.einsum("tepk,tpk->tep", data0, in_bin)
            row_num = np.einsum("tepk,tpk->tep", finite, in_bin)
            out = np.nanmean(row_sum / row_num, axis=2)
        elif method == "sum":
            out = np.einsum("tepk,tpk->te", data0, in_bin)
        else:
            sum_w = np.einsum("tepk,tpk,k->te", data0, in_bin, weights)
            out = sum_w / np.einsum("tepk,tpk,k->te", finite, in_bin, weights)

    return out


def get_pitch_angle_dist(vdf, b_xyz, tint: list = None, verbose=True, **kwargs):
    r"""Computes the pitch angle distributions from particle data, as
    irfu-matlab mms.get_pitchangledist.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution, with (time, energy) energies, phi as (time, phi)
        or (phi,) and theta as (theta,), in degrees.
    b_xyz : xarray.DataArray
        Time series of the magnetic field in the same coordinate system as
        vdf (e.g., DMPA for FPI), resampled to the times of vdf.
    tint : list of str, Optional
        Time interval for closeup.
    verbose : bool, Optional
        Log the choice of pitch angles. Default is True.

    Returns
    -------
    pad : xarray.Dataset
        Particle pitch angle distribution, with data (time, energy, pitch
        angle), energy (time, energy) (each time step keeps its energy table)
        and theta (time, pitch angle) the centres of the pitch angle bins.

    Other Parameters
    ----------------
    angles : int or float or array_like
        Number of pitch angle bins of equal width, or the bin edges in
        degrees. Default is 12 bins of 15 degrees.
    meanorsum : {"mean", "sum", "sum_weighted"}
        Method in each bin: "mean" averages over theta, then over phi (as
        irfu-matlab); "sum" sums the samples (0 for an empty bin); and
        "sum_weighted" is the mean weighted by the solid angle of the
        samples (sin theta for the uniform FPI angular grids). Default is
        "mean".

    Raises
    ------
    ValueError
        If angles or meanorsum is not understood.

    Notes
    -----
    A sample is in a bin if its pitch angle is between the bin edges,
    inclusive (as irfu-matlab), so a sample on an edge is in both bins.

    Examples
    --------
    >>> from pyrfu import mms

    Define time intervals

    >>> tint_long = ["2017-07-24T12:48:34.000", "2017-07-24T12:58:20.000"]
    >>> tint_zoom = ["2017-07-24T12:49:18.000", "2017-07-24T12:49:30.000"]

    Load ions velocity distribution for MMS1

    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint_long, 1)

    Load magnetic field in the spacecraft coordinates system.

    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint_long, 1)

    Compute pitch angle distribution

    >>> options = dict(angles=24)
    >>> pad_i = mms.get_pitch_angle_dist(vdf, b_dmpa, tint_zoom, **options)

    """

    # Default pitch angles. 15 degree angle widths
    angles_v = np.linspace(15, 180, int(180 / 15))
    d_angles = np.median(np.diff(angles_v)) * np.ones(len(angles_v))

    if "angles" in kwargs:
        if isinstance(kwargs["angles"], (int, float)):
            n_angles = int(kwargs["angles"])  # Make sure input is integer
            d_angles = 180 / n_angles
            angles_v = np.linspace(d_angles, 180, n_angles)
            d_angles = d_angles * np.ones(n_angles)

            if verbose:
                logger.info("User defined number of pitch angles.")

        elif isinstance(kwargs["angles"], (list, np.ndarray)):
            angles_v = np.asarray(kwargs["angles"], dtype=np.float64)
            d_angles = np.diff(angles_v)
            angles_v = angles_v[1:]

            if verbose:
                logger.info("User defined pitch angle limits.")

        else:
            raise ValueError("angles parameter not understood.")

    # Method
    mean_or_sum = kwargs.get("meanorsum", "mean")

    if mean_or_sum not in ["mean", "sum", "sum_weighted"]:
        raise ValueError(f"meanorsum {mean_or_sum!r} not understood.")

    pitch_angles = angles_v - d_angles / 2

    vdf_data = vdf.data
    energy = vdf.energy.data

    # Azimuths as (time, phi)
    phi = vdf.phi.data
    if phi.ndim == 1:
        phi = np.broadcast_to(phi, (len(vdf.time), len(phi)))

    if tint is not None:
        b_xyz = time_clip(b_xyz, tint)
        vdf_data = time_clip(vdf_data, tint)
        in_tint = np.isin(vdf.time.data, vdf_data.time.data)
        energy, phi = energy[in_tint], phi[in_tint]

    time = vdf_data.time.data
    data = vdf_data.data
    theta = np.deg2rad(vdf.theta.data)

    b_hat = normalize(resample(b_xyz, vdf_data)).data

    # Cosine of the pitch angles of the particle velocities (minus the look
    # directions), (time, phi, theta): independent of energy
    phi = np.deg2rad(phi)[:, :, None]
    cos_pa = -np.cos(phi) * np.sin(theta) * b_hat[:, 0, None, None]
    cos_pa -= np.sin(phi) * np.sin(theta) * b_hat[:, 1, None, None]
    cos_pa -= np.cos(theta) * b_hat[:, 2, None, None]
    theta_b = np.rad2deg(np.arccos(np.clip(cos_pa, -1.0, 1.0)))

    # Samples with data, and the data with zeros elsewhere, for the bin sums
    finite = np.isfinite(data)
    data0 = np.where(finite, data, 0.0)
    finite = finite.astype(np.float64)
    weights = np.sin(theta)

    pad_arr = np.stack(
        [
            _bin_reduce(
                data0,
                finite,
                (theta_b >= angle - d_angle) & (theta_b <= angle),
                mean_or_sum,
                weights,
            )
            for angle, d_angle in zip(angles_v, d_angles)
        ],
        axis=-1,
    )

    pad = xr.Dataset(
        {
            "data": (["time", "idx0", "idx1"], pad_arr),
            "energy": (["time", "idx0"], energy),
            "theta": (
                ["time", "idx1"],
                np.tile(pitch_angles, (len(time), 1)),
            ),
            "time": time,
            "idx0": np.arange(energy.shape[1]),
            "idx1": np.arange(len(pitch_angles)),
        },
    )

    pad.attrs = {
        **vdf.attrs,
        "mean_or_sum": mean_or_sum,
        "delta_pitchangle_minus": d_angles * 0.5,
        "delta_pitchangle_plus": d_angles * 0.5,
    }

    pad.time.attrs = dict(vdf.time.attrs)
    pad.energy.attrs = dict(vdf.energy.attrs)
    pad.data.attrs["UNITS"] = vdf.data.attrs["UNITS"]

    return pad
