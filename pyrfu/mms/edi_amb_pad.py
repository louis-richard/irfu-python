#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Pitch-angle bins of the four channels at 0 and 180 degrees, 11.25 degrees
# wide: channel 1 the closest to 0 and to 180 degrees (irfu-matlab)
_D_ANGLE = 180.0 / 16.0
_CENTRES = np.array([0.5, 1.5, 2.5, 3.5, 12.5, 13.5, 14.5, 15.5]) * _D_ANGLE
_FLUXES = [f"flux{i:d}_0" for i in range(1, 5)]
_FLUXES += [f"flux{i:d}_180" for i in range(4, 0, -1)]


def edi_amb_pad(amb, error: bool = False):
    r"""Pitch-angle distribution of the EDI ambient electrons, from the
    fluxes of the four channels at 0 and 180 degrees of the burst amb-pm2
    files, as irfu-matlab ``mms.get_data("Flux-amb-pm2_edi_brst_l2", ...)``.

    Parameters
    ----------
    amb : xarray.Dataset
        EDI ambient dataset of the burst amb-pm2 files,
        ``mms.get_data("flux-amb-pm2_edi_brst_l2", tint, mms_id)``, with the
        fluxes ``flux1_0`` to ``flux4_180``, their errors (``*_delta``) and
        the energy of the GDUs (``energy_gdu1``).
    error : bool, Optional
        Distribution of the errors of the fluxes (``flux*_delta``, irfu-matlab
        ``Flux-err-amb-pm2``) instead of the fluxes. Default is False.

    Returns
    -------
    pad : xarray.Dataset
        Pitch-angle distribution (time, energy, pitch angle), as
        `get_pitch_angle_dist`: one energy, that of the GDU of each time (eV,
        10 % wide), and 8 pitch angles of 11.25 degrees, 0 to 45 degrees
        (channels 1 to 4) and 135 to 180 degrees (channels 4 to 1), in the
        units of the fluxes (cm^-2 s^-1).

    Raises
    ------
    ValueError
        If `amb` does not have the fluxes of the four channels (e.g. the
        survey files, with channel 1 only) or the energy of the GDUs.

    Notes
    -----
    The energy is that of GDU1 in the files (``energy_gdu1``, at the time of
    the last record before each flux; equal to that of GDU2 in the files
    checked): 500 eV in 2017 and 2019, 250 eV in 2022. irfu-matlab uses 500
    eV for all the files.

    See Also
    --------
    pyrfu.mms.get_data : EDI ambient datasets, ``flux-amb-pm2_edi_*``.
    pyrfu.mms.pad_palim : Select pitch angles.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-01-01T01:51:30", "2017-01-01T01:53:00"]
    >>> amb = mms.get_data("flux-amb-pm2_edi_brst_l2", tint, 1)
    >>> pad = mms.edi_amb_pad(amb)

    """

    names = [f"{name}_delta" if error else name for name in _FLUXES]
    missing = [name for name in [*names, "energy_gdu1"] if name not in amb]

    if missing:
        raise ValueError(
            f"amb has no {', '.join(missing)}: the burst amb-pm2 dataset is "
            "needed (fluxes of the four channels and energy of the GDUs)"
        )

    fluxes = [amb[name] for name in names]
    time = fluxes[0][fluxes[0].dims[0]].data
    data = np.stack([flux.data for flux in fluxes], axis=1)[:, None, :]

    # Energy of GDU1 at its last record before each time (its first record
    # for the times before it)
    gdu = amb["energy_gdu1"]
    i_energy = np.searchsorted(gdu[gdu.dims[0]].data, time, side="right") - 1
    energy = np.asarray(gdu.data, dtype=np.float64)[np.clip(i_energy, 0, None)]
    energy = energy[:, None]

    out = xr.Dataset(
        {
            "data": (["time", "idx0", "idx1"], data),
            "energy": (["time", "idx0"], energy),
            "theta": (["time", "idx1"], np.tile(_CENTRES, (len(time), 1))),
            "time": time,
            "idx0": np.arange(1),
            "idx1": np.arange(len(_CENTRES)),
        },
    )
    out.attrs = {
        "species": "electrons",
        "delta_energy_minus": 0.05 * energy,
        "delta_energy_plus": 0.05 * energy,
        "delta_pitchangle_minus": np.full(len(_CENTRES), 0.5 * _D_ANGLE),
        "delta_pitchangle_plus": np.full(len(_CENTRES), 0.5 * _D_ANGLE),
    }
    out.data.attrs = {"UNITS": fluxes[0].attrs.get("UNITS")}
    out.energy.attrs = {"UNITS": "eV"}
    out.theta.attrs = {"UNITS": "deg"}

    return out
