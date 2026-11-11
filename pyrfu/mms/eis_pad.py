#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.normalize import normalize
from ..pyrf.resample import resample
from ..pyrf.ts_vec_xyz import ts_vec_xyz
from .feeps_avg_4sc import _without_gaps


def _calc_angle(look_, vec):
    vec_hat = normalize(vec)
    # Clipped: rounding takes the cosine of (anti)parallel directions slightly
    # beyond 1 (NaN, the telescope left out)
    cos_theta = np.clip(np.sum(vec_hat.data * look_.data, axis=1), -1.0, 1.0)
    theta_ = np.rad2deg(np.pi - np.arccos(cos_theta))
    return theta_


def _pad_mean(pa_file, flux_file, pa_label, delta_pa):
    r"""Mean of the telescopes whose field of view (+-10 deg) overlaps each
    pitch-angle bin, (time, bin, energy); NaN without telescope or data."""
    pa_hangw = 10.0  # deg, angular response of the telescopes
    pa_flux = np.full([len(pa_file), len(pa_label), flux_file.shape[2]], np.nan)

    for (i), (j, pa_lbl) in itertools.product(
        range(len(pa_file)),
        enumerate(pa_label),
    ):
        cond_ = np.logical_and(
            pa_file[i, :] + pa_hangw >= pa_lbl - delta_pa,
            pa_file[i, :] - pa_hangw < pa_lbl + delta_pa,
        )
        ind = np.where(cond_)[0]
        if ind.size != 0:
            # NaN where no telescope has data: an expected empty bin
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
                pa_flux[i, j, :] = np.nanmean(flux_file[i, ind, :], axis=0)

    return pa_flux


def _pad_sum(pa_file, flux_file, pa_width, n_pabins):
    r"""Sum of the telescopes whose centre pitch angle is in each bin, [k w,
    (k + 1) w), 180 deg in the last one, (time, bin, energy); NaN without
    telescope or data. Each telescope is in one bin."""
    i_bin = np.floor(np.where(np.isfinite(pa_file), pa_file, -1.0) / pa_width)
    i_bin = np.minimum(i_bin, n_pabins - 1)
    in_bin = (i_bin[:, :, None] == np.arange(n_pabins)).astype(np.float64)
    finite = np.isfinite(flux_file)

    total = np.einsum("tsb,tse->tbe", in_bin, np.where(finite, flux_file, 0.0))
    n_finite = np.einsum("tsb,tse->tbe", in_bin, finite.astype(np.float64))

    return np.where(n_finite > 0, total, np.nan)


def eis_pad(
    inp_allt,
    vec: xr.DataArray = None,
    energy: list = None,
    pa_width: int = 15,
    method: str = "mean",
):
    r"""Calculates Pitch Angle Distributions (PADs) using data from the MMS
    Energetic Ion Spectrometer (EIS)

    Parameters
    ----------
    inp_allt : xarray.Dataset
        Energy spectrum for all telescopes, with their look directions in GSE
        (from :func:`pyrfu.mms.get_eis_allt`).
    vec : xarray.DataArray or array_like
        Magnetic field in GSE, the frame of the look directions: a time
        series (resampled to the times of `inp_allt`, none outside its time
        range and in its data gaps, as :func:`pyrfu.mms.feeps_pitch_angles`)
        or a constant direction.
    energy : array_like, Optional
        Energy range to include in the calculation: the channels with their
        lower or upper edge in it. Default is all the channels.
    pa_width : int, Optional
        Size of the pitch angle bins, in degrees, a divisor of 180. Default is
        15.
    method : {"mean", "sum"}, Optional
        "mean" (default): the mean of the telescopes whose field of view
        overlaps the bin, as IDL SPEDAS, for every unit. "sum": the sum of the
        telescopes whose centre pitch angle is in the bin, for counts (each
        telescope in one bin, see Notes).

    Returns
    -------
    pa_flux : xarray.DataArray
        Pitch-angle angle distribution for every energy channels in the
        `energy` range, with the units (``UNITS``, ``LABLAXIS``), species,
        energy widths of these channels (``delta_energy_minus``,
        ``delta_energy_plus``), ``pa_width`` and ``method`` in its attributes
        (it had none).

    Raises
    ------
    ValueError
        If `vec` is not given (the pitch angles were taken from X-GSE), is
        neither a time series nor a direction, if no channel is in the
        `energy` range, if `pa_width` doesn't divide 180 (the bins had
        gaps), or if `method` is not "mean" or "sum".

    Notes
    -----
    The pitch angles are those of the particles, which move opposite to the
    look direction of the telescope: 180 deg minus the angle between the look
    direction and the field. With the field of the EIS files, they equal their
    ``pitch_angle_t?``; with the FGM field, the pitch angles differ by the
    angle between the two fields (a fraction of a degree in survey). Each bin
    averages the telescopes whose field of view (+-10 deg) overlaps it, zero
    fluxes included, as IDL SPEDAS ``mms_eis_pad`` and pySPEDAS: the zeros were
    left out, which biased the PADs high and left the bins without counts NaN.
    The means are for every unit, counts included: a telescope is in up to
    three bins, so a sum over them would count its particles up to three
    times. ``method="sum"`` puts each telescope in the bin of its centre pitch
    angle instead, [k ``pa_width``, (k + 1) ``pa_width``), 180 deg in the
    last bin (a telescope on an edge is in the upper bin, where
    :func:`pyrfu.mms.get_pitch_angle_dist` puts a sample on an edge in both),
    and sums them, as ``get_pitch_angle_dist(meanorsum="sum")``: the bins
    hold all the counts once. A bin without telescope, or with NaN only, is
    NaN.

    See Also
    --------
    pyrfu.mms.get_eis_allt

    """

    time_ = inp_allt.time.data

    if not np.isclose(180.0 / pa_width, round(180.0 / pa_width)):
        raise ValueError(f"pa_width must divide 180, not {pa_width}")

    if method not in ["mean", "sum"]:
        raise ValueError(f"method must be 'mean' or 'sum', not {method!r}")

    if vec is None:
        raise ValueError(
            "give the magnetic field in GSE (vec): the pitch angles were taken "
            "from X-GSE"
        )

    if isinstance(vec, xr.DataArray):
        # No B (no pitch angle) outside the time range of vec and in its gaps
        # (it was held at its last value and interpolated across)
        vec = _without_gaps(vec, resample(vec, inp_allt.time))
    else:
        vec = np.asarray(vec, dtype=np.float64)

        if vec.shape != (3,):
            raise ValueError("vec must be a time series or a direction (3 values)")

        vec = ts_vec_xyz(time_, np.tile(vec, (len(time_), 1)))

    if energy is None:
        energy = inp_allt.energy.data[[0, -1]]

    # set up the number of pa bins to create
    n_pabins = int(180.0 / pa_width)
    pa_label = list(180.0 * np.arange(0, n_pabins) / n_pabins + pa_width / 2.0)

    delta_pa = pa_width / 2.0

    scopes = list(filter(lambda x: x.startswith("t"), inp_allt))

    pa_file = np.zeros([len(time_), len(scopes)])

    # Lower and upper edges of the channels (the lower one was E + delta_minus)
    e_minu_w = np.asarray(inp_allt.delta_energy_minus, dtype=np.float64)
    e_plus_w = np.asarray(inp_allt.delta_energy_plus, dtype=np.float64)
    e_minu = inp_allt.energy.data - e_minu_w
    e_plus = inp_allt.energy.data + e_plus_w

    cond_low = np.logical_and(e_minu >= energy[0], e_minu <= energy[1])
    cond_hig = np.logical_and(e_plus >= energy[0], e_plus <= energy[1])
    ener_idx = np.where(np.logical_or(cond_low, cond_hig))[0]

    if ener_idx.size == 0:
        raise ValueError(f"No energy channel in the energy range {energy}")

    flux_file = np.zeros([len(time_), len(scopes), len(ener_idx)])

    for i_s, scope in enumerate(scopes):
        # get pa from each detector
        pa_file[:, i_s] = _calc_angle(inp_allt[f"look_{scope}"], vec)

        # get energy range of interest
        flux_file[:, i_s, :] = inp_allt[scope][:, ener_idx]

    # The zero fluxes are kept, as IDL and pySPEDAS (they were set to NaN)
    if method == "sum":
        pa_flux = _pad_sum(pa_file, flux_file, pa_width, n_pabins)
    else:
        pa_flux = _pad_mean(pa_file, flux_file, pa_label, delta_pa)

    attrs = {
        "species": inp_allt.attrs.get("species", ""),
        "delta_energy_minus": np.broadcast_to(e_minu_w, e_minu.shape)[ener_idx],
        "delta_energy_plus": np.broadcast_to(e_plus_w, e_plus.shape)[ener_idx],
        "pa_width": pa_width,
        "method": method,
    }

    # Units of the telescopes (none without, they were "")
    for key in ["UNITS", "LABLAXIS"]:
        if key in inp_allt[scopes[0]].attrs:
            attrs[key] = inp_allt[scopes[0]].attrs[key]

    pa_flux = xr.DataArray(
        pa_flux,
        coords=[time_, pa_label, inp_allt.energy.data[ener_idx]],
        dims=["time", "theta", "energy"],
        attrs=attrs,
    )
    pa_flux.theta.attrs["UNITS"] = "deg"
    pa_flux.energy.attrs = dict(inp_allt.energy.attrs)

    return pa_flux
