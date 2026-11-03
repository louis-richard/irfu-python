#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.ts_skymap import ts_skymap
from .get_hpca_dist import get_hpca_dist
from .vdf_convert import vdf_convert

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.9"
__status__ = "Prototype"


def _circular_mean(angles, axis):
    r"""Circular mean of angles (degrees) along axis, in [0, 360)."""
    rad = np.deg2rad(angles)
    mean = np.arctan2(np.mean(np.sin(rad), axis=axis), np.mean(np.cos(rad), axis=axis))
    return np.rad2deg(mean) % 360.0


def _sides(phi, colat):
    r"""Anodes of the two sides of the HPCA field of view (azimuths 180 degrees
    apart), each sorted by polar angle."""
    offset = (phi - phi[0] + 180.0) % 360.0 - 180.0
    sides = [np.flatnonzero(np.abs(offset) < 90.0)]
    sides.append(np.flatnonzero(np.abs(offset) >= 90.0))
    sides = [side[np.argsort(colat[side], kind="stable")] for side in sides]

    if len(sides[0]) != len(sides[1]) or np.any(colat[sides[0]] != colat[sides[1]]):
        raise ValueError("the two sides of HPCA have different anode elevations")

    return sides


def hpca_skymap(
    inp,
    azimuth,
    elevation=None,
    to_psd: bool = True,
    source: str = "default",
    data_path: str = "",
):
    r"""Skymap distribution of an HPCA ion species, one per half-spin, as the
    FPI distributions (look directions in DBCS).

    Parameters
    ----------
    inp : xarray.DataArray
        HPCA ion spec; [nt, npo16, ner63], e.g. ``dpfoplus_hpca_brst_l2``.
    azimuth : xarray.DataArray
        Azimuthal angles per energy; [nT, naz16, npo16, ner63], as for
        :func:`pyrfu.mms.get_hpca_dist`.
    elevation : array_like, Optional
        Polar angles (colatitudes, deg) of the 16 anodes. Default reads
        mms?_hpca_centroid_elevation_angle in the file of `inp`.
    to_psd : bool, Optional
        Convert the fluxes to phase-space densities (s^3/cm^6), as the FPI
        distributions. Default is True.
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to read the anode elevations from (if `elevation` is None).
    data_path : str, Optional
        Path of MMS data (for `source` "local").

    Returns
    -------
    out : xarray.Dataset
        Skymap (time, energy, phi, theta) at the times of `azimuth`: 63
        energies (eV, kinetic: the energies per charge times the charge), 32
        azimuths (the 16 of each side of the instrument) and 8 polar angles
        of the look directions, with their widths (``delta_phi_*``,
        ``delta_theta_*``), the ``species`` (e.g. "oxygen+") and ``charge``.
        NaN for the half-spins not complete in `inp`.

    See Also
    --------
    pyrfu.mms.get_hpca_dist

    Notes
    -----
    The angles of the files are the directions of the particles (as in IDL
    SPEDAS ``mms_get_hpca_dist``); those of the skymap are the look
    directions (opposite), as the FPI distributions, so that
    :func:`pyrfu.mms.reduce`, :func:`pyrfu.mms.vdf_omni` and the other
    skymap routines apply. The 16 anodes form two fans 180 degrees apart in
    azimuth with the same 8 polar angles: each half-spin covers all the
    directions once. The azimuth of a sample drifts by about 11 degrees (one
    azimuth step) over the energy sweep; the skymap has one azimuth per
    sample, the circular mean over the energies, so that the azimuths of the
    lowest and highest energies are off by up to half a step.

    """

    dist = get_hpca_dist(inp, azimuth, elevation, source, data_path)

    # half-spin, energy, azimuth, anode
    phi_p, data = dist["phi"], dist["data"]
    colat_p = 90.0 - dist["theta"][0, 0, :]
    n_t, n_en, n_az, _ = data.shape

    complete = np.all(np.isfinite(phi_p), axis=(1, 2, 3))
    sides = _sides(phi_p[np.argmax(complete), 0, 0, :], colat_p)
    n_th = len(sides[0])

    # Look directions: opposite to the particles
    phi = np.full((n_t, 2 * n_az), np.nan)
    sky = np.full((n_t, n_en, 2 * n_az, n_th), np.nan)

    for i, side in enumerate(sides):
        phi_side = _circular_mean(phi_p[complete][..., side[0]], axis=1)
        phi[complete, i * n_az : (i + 1) * n_az] = (phi_side + 180.0) % 360.0
        sky[:, :, i * n_az : (i + 1) * n_az, :] = data[..., side[::-1]]

    theta = 180.0 - colat_p[sides[0][::-1]]

    # Azimuths in increasing order at each time (the first complete half-spin
    # for the others)
    phi[~complete] = phi[np.argmax(complete)]
    order = np.argsort(phi, axis=1)
    phi = np.take_along_axis(phi, order, axis=1)
    sky = np.take_along_axis(sky, order[:, None, :, None], axis=2)

    d_phi = np.nanmean(dist["dphi"], axis=(1, 2, 3)) / 2.0

    # Kinetic energies, and fluxes per unit kinetic energy: the energies and
    # the fluxes of the files are per charge (energy per charge E/q, flux per
    # unit E/q)
    energy = dist["charge"] * np.asarray(inp.ccomp.data, dtype=np.float64)
    energy = np.tile(energy, (n_t, 1))
    sky /= dist["charge"]

    # Azimuthal widths: half the steps; polar widths: the 22.5 degree anodes
    glob_attrs = {
        "species": dist["species"],
        "charge": dist["charge"],
        "delta_phi_minus": np.tile(d_phi[:, None], (1, 2 * n_az)),
        "delta_phi_plus": np.tile(d_phi[:, None], (1, 2 * n_az)),
        "delta_theta_minus": np.full(n_th, 11.25),
        "delta_theta_plus": np.full(n_th, 11.25),
    }

    if "GLOBAL" in inp.attrs:
        glob_attrs = {**inp.attrs["GLOBAL"], **glob_attrs}

    attrs = {k: v for k, v in inp.attrs.items() if k != "GLOBAL"}
    attrs["COORDINATE_SYSTEM"] = azimuth.attrs.get("COORDINATE_SYSTEM", "DBCS")
    coords_attrs = {
        "time": dict(azimuth.time.attrs),
        "energy": {"UNITS": "eV"},
        "phi": {"UNITS": "deg"},
        "theta": {"UNITS": "deg"},
    }

    out = ts_skymap(
        dist["time"],
        sky,
        energy,
        phi,
        theta,
        energy0=energy[0],
        energy1=energy[0],
        esteptable=np.zeros(n_t, dtype=np.uint8),
        attrs=attrs,
        glob_attrs=glob_attrs,
        coords_attrs=coords_attrs,
    )

    if to_psd:
        out = vdf_convert(out, "s^3/cm^6")

    return out
