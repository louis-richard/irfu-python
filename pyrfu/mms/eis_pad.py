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

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _calc_angle(look_, vec):
    vec_hat = normalize(vec)
    theta_ = np.rad2deg(
        np.pi - np.arccos(np.sum(vec_hat.data * look_.data, axis=1)),
    )
    return theta_


def eis_pad(
    inp_allt,
    vec: xr.DataArray = None,
    energy: list = None,
    pa_width: int = 15,
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
        series (resampled to the times of `inp_allt`) or a constant direction.
    energy : array_like, Optional
        Energy range to include in the calculation: the channels with their
        lower or upper edge in it. Default is all the channels.
    pa_width : int, Optional
        Size of the pitch angle bins, in degrees, a divisor of 180. Default is
        15.

    Returns
    -------
    pa_flux : xarray.DataArray
        Pitch-angle angle distribution for every energy channels in the
        `energy` range, with the units, species, energy widths of these
        channels (``delta_energy_minus``, ``delta_energy_plus``) and
        ``pa_width`` in its attributes (it had none).

    Raises
    ------
    ValueError
        If `vec` is not given (the pitch angles were taken from X-GSE), is
        neither a time series nor a direction, if no channel is in the
        `energy` range, or if `pa_width` doesn't divide 180 (the bins had
        gaps).

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

    See Also
    --------
    pyrfu.mms.get_eis_allt

    """

    time_ = inp_allt.time.data

    if not np.isclose(180.0 / pa_width, round(180.0 / pa_width)):
        raise ValueError(f"pa_width must divide 180, not {pa_width}")

    if vec is None:
        raise ValueError(
            "give the magnetic field in GSE (vec): the pitch angles were taken "
            "from X-GSE"
        )

    if isinstance(vec, xr.DataArray):
        vec = resample(vec, inp_allt.time)
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

    # Account for angular response (finite field of view) of instruments
    pa_hangw = 10.0  # deg
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
    shape_ = [len(time_), n_pabins, len(ener_idx)]
    pa_flux = np.zeros(shape_)

    pa_flux[...] = np.nan

    for i_s, scope in enumerate(scopes):
        # get pa from each detector
        pa_file[:, i_s] = _calc_angle(inp_allt[f"look_{scope}"], vec)

        # get energy range of interest
        flux_file[:, i_s, :] = inp_allt[scope][:, ener_idx]

    # The zero fluxes are kept, as IDL and pySPEDAS (they were set to NaN)

    for (i), (j, pa_lbl) in itertools.product(
        range(len(time_)),
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

    attrs = {
        "UNITS": inp_allt[scopes[0]].attrs.get("UNITS", ""),
        "species": inp_allt.attrs.get("species", ""),
        "delta_energy_minus": np.broadcast_to(e_minu_w, e_minu.shape)[ener_idx],
        "delta_energy_plus": np.broadcast_to(e_plus_w, e_plus.shape)[ener_idx],
        "pa_width": pa_width,
    }

    pa_flux = xr.DataArray(
        pa_flux,
        coords=[time_, pa_label, inp_allt.energy.data[ener_idx]],
        dims=["time", "theta", "energy"],
        attrs=attrs,
    )

    return pa_flux
