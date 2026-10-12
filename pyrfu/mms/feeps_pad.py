#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

from .feeps_active_eyes import feeps_active_eyes

# Local imports
from .feeps_pitch_angles import feeps_pitch_angles

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Angular response (finite field of view) of instruments electrons can use
# +/- 21.4 deg on each pitch angle as average response angle; ions can start
# with +/-10 deg, but both need to be further refined
angular_repsonse = {"electron": 21.4, "ion": 10}


def _pa_data_map(idx_maps, d_type, d_rate):
    pa_data_map = {}

    if d_rate == "srvy":
        pa_data_map[f"top-{d_type}"] = idx_maps[f"{d_type}-top"]
        pa_data_map[f"bottom-{d_type}"] = idx_maps[f"{d_type}-bottom"]
    else:
        # note: the following are indices of the top/bottom sensors in pa_data
        # they should be consistent with pa_dlimits.labels
        pa_data_map["top-electron"] = np.arange(9)
        pa_data_map["bottom-electron"] = np.arange(9, 18)

        # and ions:
        pa_data_map["top-ion"] = [0, 1, 2]
        pa_data_map["bottom-ion"] = [3, 4, 5]

    return pa_data_map


def _dpa_dflux(inp_dataset, pitch_angles, pa_data_map, energy, d_type, mms_id):
    pa_times = pitch_angles.time
    pa_data = pitch_angles.data

    trange = np.datetime_as_string(
        np.hstack([np.min(pa_times.data), np.max(pa_times.data)]),
        "ns",
    )

    eyes = feeps_active_eyes(inp_dataset.attrs, list(trange), mms_id)

    sensor_types = ["top", "bottom"]

    n_times = len(pa_times)
    n_top = len(pa_data_map[f"top-{d_type}"])
    n_bottom = len(pa_data_map[f"bottom-{d_type}"])

    dflux, dpa = [np.zeros([n_times, n_top + n_bottom]) for _ in range(2)]

    for s_type in sensor_types:
        pa_map = pa_data_map[f"{s_type}-{d_type}"]

        particle_idxs = [eye - 1 for eye in eyes[s_type]]

        for isen, sensor_num in enumerate(particle_idxs):
            var_name = f"{s_type}-{sensor_num + 1:d}"

            data = inp_dataset[var_name].data
            energies = inp_dataset[inp_dataset[var_name].dims[1]].data

            # remove any 0s before averaging (on a float copy, not the input)
            data = np.where(data == 0, np.nan, data.astype(float))

            # assumes all energies are NaNs if the first is
            if np.isnan(energies[0]):
                continue

            # energy indices to use:
            idx = np.where(
                np.logical_and(energies >= energy[0], energies <= energy[1]),
            )[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                dflux[:, pa_map[isen]] = np.nanmean(data[:, idx], axis=1)

            dpa[:, pa_map[isen]] = pa_data[:, pa_map[isen]]

    # we need to replace the 0.0s left in after populating dpa with NaNs;
    # these 0.0s are left in there because these points aren't covered by
    # sensors loaded for this datatype/d_ratee
    dpa[dpa == 0] = "nan"

    return dpa, dflux


def _pa_flux(pa_times, pa_bins, pa_labels, dpa, dflux, d_type):
    n_pabins = len(pa_bins) - 1
    # Account for angular response
    dangresp = angular_repsonse[d_type]

    pa_flux = np.zeros([len(pa_times), int(n_pabins)])
    delta_pa = (pa_bins[1] - pa_bins[0]) / 2.0

    # Now loop through PA bins, find the telescopes where there is data in
    # those bins at each time and average it up! (comparisons with NaN pitch
    # angles are False)
    for ipa in range(n_pabins):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ind = (dpa + dangresp >= pa_labels[ipa] - delta_pa) & (
                dpa - dangresp < pa_labels[ipa] + delta_pa
            )
            pa_flux[:, ipa] = np.nanmean(np.where(ind, dflux, np.nan), axis=1)

    pa_flux[pa_flux == 0] = "nan"  # fill any missed bins with NAN

    return pa_flux


def feeps_pad(
    inp_dataset,
    b_bcs,
    bin_size: float = 16.3636,
    energy: list = None,
    angles_from_bfield: bool = False,
):
    r"""Compute pitch angle distribution using FEEPS data.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Energy spectrum of all eyes.
    b_bcs : xarray.DataArray
        Time series of the magnetic field in body coordinates (BCS), e.g. the
        FGM survey L2 field as in IDL SPEDAS. Not used for burst data, unless
        `angles_from_bfield` is True.
    bin_size : float, Optional
        Width of the pitch angles bins in degrees. Default is 16.3636. As in
        IDL SPEDAS and pyspedas, there are floor(180 / bin_size) bins of
        exactly this width from 0, so the last bin ends below 180 degrees
        when `bin_size` doesn't divide 180.
    energy : array_like, Optional
        Energy range of particles in keV, starting at 32 keV or above. Default
        is [70., 600.]
    angles_from_bfield : bool, Optional
        For burst data, compute the pitch angles from `b_bcs` instead of using
        the pitch angles of the CDF (``pitch_angle``). Default is False.

    Returns
    -------
    pad : xarray.DataArray
        Time series of the pitch angle distribution.

    Raises
    ------
    ValueError
        If the integral channel of the eyes is not split (16 channels), the
        energy range starts below 32 keV, or the data rate or species is not
        supported.

    Warns
    -----
    UserWarning
        If no energy channel is in the energy range (the PAD is then NaN).

    Notes
    -----
    The integral channel must be split before (`feeps_split_integral_ch`),
    otherwise the electron integral channel (about 580 keV) would be averaged
    as a spectrum channel.

    As in IDL SPEDAS and pyspedas, burst data use the pitch angles of the CDF
    by default, and survey data the pitch angles computed from `b_bcs`
    (`feeps_pitch_angles`).

    """

    for k in filter(lambda x: x.startswith(("top", "bottom")), inp_dataset):
        if inp_dataset[k].shape[1] == 16:
            raise ValueError(
                f"{k} has 16 energy channels: split the integral channel first "
                "with feeps_split_integral_ch (or use feeps_corrections)"
            )

    if energy is None:
        energy = [70.0, 600.0]

    if energy[0] < 32.0:
        raise ValueError("Please use a starting energy of 32 keV or above")

    eye_names = list(filter(lambda x: x.startswith(("top", "bottom")), inp_dataset))
    in_range = [
        np.any(
            (inp_dataset[inp_dataset[k].dims[1]].data >= energy[0])
            & (inp_dataset[inp_dataset[k].dims[1]].data <= energy[1])
        )
        for k in eye_names
    ]

    if not any(in_range):
        warnings.warn(
            f"No FEEPS energy channel in {energy} keV: the PAD is NaN", UserWarning
        )

    time = inp_dataset.time.data
    attrs = {**inp_dataset.attrs, "energy_range": energy}
    mms_id, d_type, d_rate = list(map(attrs.get, ["mmsId", "dtype", "tmmode"]))

    if d_rate not in ["srvy", "brst"]:
        raise ValueError(f"Invalid data rate {d_rate}: must be srvy or brst")

    if d_type not in ["electron", "ion"]:
        raise ValueError(f"Invalid species {d_type}: must be electron or ion")

    # Bins of bin_size degrees from 0, as IDL SPEDAS and pyspedas (the
    # tolerance keeps e.g. 180 / 7 at 7 bins)
    n_pabins = int(np.floor(180.0 / bin_size + 1e-6))
    pa_bins = [bin_size * pa_bin for pa_bin in range(n_pabins + 1)]
    pa_labels = [pa_bin + bin_size / 2.0 for pa_bin in pa_bins[:-1]]

    if d_rate == "brst" and not angles_from_bfield and "pitch_angle" in inp_dataset:
        # pitch angles of the CDF, fill values (inactive eyes) to NaN
        pitch_angles = inp_dataset["pitch_angle"].astype(float)
        pitch_angles = pitch_angles.where(
            (pitch_angles >= 0.0) & (pitch_angles <= 180.0)
        )
        idx_maps = None
    else:
        pitch_angles, idx_maps = feeps_pitch_angles(inp_dataset, b_bcs)

    pa_data_map = _pa_data_map(idx_maps, d_type, d_rate)

    dpa, dflux = _dpa_dflux(
        inp_dataset,
        pitch_angles,
        pa_data_map,
        energy,
        d_type,
        mms_id,
    )

    pa_flux = _pa_flux(
        pitch_angles.time,
        pa_bins,
        pa_labels,
        dpa,
        dflux,
        d_type,
    )

    pad = xr.DataArray(
        pa_flux,
        coords=[time, pa_labels],
        dims=["time", "theta"],
        attrs=attrs,
    )

    # units of the eyes (intensity, count rate, ...)
    pad.attrs["UNITS"] = inp_dataset[eye_names[0]].attrs.get(
        "UNITS", "1/(cm^2 s sr keV)"
    )

    return pad
