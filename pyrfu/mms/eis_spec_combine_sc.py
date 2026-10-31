#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ._eis_units import _check_same_units

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _idx_closest(lst0, lst1):
    return [(np.abs(np.asarray(lst0) - k)).argmin() for k in lst1]


def _nearest(t_ref, t_inp, tol):
    r"""Index in t_inp of the time nearest to each time of t_ref, -1 if it is
    more than tol away (times sorted, in ns)."""
    out = np.full(len(t_ref), -1)

    if len(t_inp) == 0:
        return out

    idx = np.searchsorted(t_inp, t_ref).clip(0, len(t_inp) - 1)
    idx_prev = (idx - 1).clip(0)
    closer = np.abs(t_inp[idx_prev] - t_ref) < np.abs(t_inp[idx] - t_ref)
    idx = np.where(closer, idx_prev, idx)
    near = np.abs(t_inp[idx] - t_ref) <= tol
    out[near] = idx[near]

    return out


def _paired_samples(ref, inp):
    r"""Index of the sample of inp paired with each sample of ref (-1 for
    none): for full-resolution spectra (with sectors), the same sector nearest
    in time within half a spin, as the sector gives the look direction;
    otherwise (spin averages), the nearest time within half a time step."""
    t_ref = ref.time.data.astype("datetime64[ns]").astype(np.int64)
    t_inp = inp.time.data.astype("datetime64[ns]").astype(np.int64)
    d_t = np.median(np.diff(t_ref)) if len(t_ref) > 1 else np.inf

    if "sector" not in ref.coords or "sector" not in inp.coords:
        return _nearest(t_ref, t_inp, d_t / 2)

    sec_ref, sec_inp = ref.sector.data, inp.sector.data

    # Half a spin: the samples of a spin have different sectors
    tol = len(np.unique(sec_ref)) * d_t / 2
    out = np.full(len(t_ref), -1)

    for sector in np.unique(sec_ref):
        i_ref = np.flatnonzero(sec_ref == sector)
        i_inp = np.flatnonzero(sec_inp == sector)
        # None for a sector without samples in inp (IndexError)
        idx = _nearest(t_ref[i_ref], t_inp[i_inp], tol)
        out[i_ref[idx >= 0]] = i_inp[idx[idx >= 0]]

    return out


def eis_spec_combine_sc(omni_vars, method: str = "mean"):
    r"""Combines omni-directional energy spectrogram variable from EIS on
    multiple MMS spacecraft.

    Parameters
    ----------
    omni_vars : list of xarray.DataArray
        Omni-directional energy spectrograms of all spacecraft (from
        :func:`pyrfu.mms.eis_omni`).
    method : {"mean", "sum"}, Optional
        Method to combine spectra. Default is "mean". For counts, "sum"
        gives the total counts of the spacecraft, but drops where a
        spacecraft has no data. IDL SPEDAS sums counts and averages cps and
        fluxes; pySPEDAS averages every unit.

    Returns
    -------
    omni_spec : xarray.DataArray
        Combined omni-directional energy spectrogram, on the times of the
        spacecraft with the fewest samples and the energies of the one with
        the fewest channels (the others at their closest channels).

    Raises
    ------
    ValueError
        If method is not "mean" or "sum", or if the spectra have different
        ``UNITS``.

    Notes
    -----
    Full-resolution spectra (with the ``sector`` coordinate of
    :func:`pyrfu.mms.eis_omni`) are combined sample by sample at the same
    sector, the nearest in time within half a spin: the sector gives the look
    direction. Spin-averaged spectra are combined at the nearest time, within
    half a spin. A spacecraft without a paired sample is left out (NaN), and
    the mean or sum is NaN where no spacecraft has data (the sum was 0). IDL
    SPEDAS ``mms_eis_spec_combine_sc`` and pySPEDAS combine the samples by
    index, which matches neither the times nor the look directions (on MMS2-4,
    2017-07-11, the sectors of the same index differed throughout, the look
    directions by 61 to 86 deg).

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_omni, pyrfu.mms.eis_pad_combine_sc

    Examples
    --------
    >>> from pyrfu.mms import get_eis_allt, eis_omni, eis_spec_combine_sc

    Define time interval

    >>> tint = ["2017-07-23T16:10:00", "2017-07-23T18:10:00"]

    Load EIS ExTOF flux spectrograms for all 6 telescopes for all spacecraft

    >>> extof_allt_mms = []
    >>> for ic in range(2, 5):
    ...     extof_allt_mms.append(get_eis_allt("flux_extof_proton_srvy_l2",
    ...                             tint, ic))

    Compute the omni-direction flux spectrogram for all spacecraft

    >>> extof_omni_mms = []
    >>> for extof_allt in extof_allt_mms:
    ...     extof_omni_mms.append(eis_omni(extof_allt))

    Combine spectrograms of all spacecraft

    >>> extof_omni_mmsx = eis_spec_combine_sc(extof_omni_mms)

    """

    if method.lower() not in ["mean", "sum"]:
        raise ValueError(f"method must be mean or sum, not {method}")

    # The units of the reference spacecraft were taken
    _check_same_units(omni_vars, "spectra")

    reftime_sc_loc = np.argmin([len(x_.time.data) for x_ in omni_vars])
    refener_sc_loc = np.argmin([len(x_.energy.data) for x_ in omni_vars])

    time_refprobe = omni_vars[reftime_sc_loc]
    ener_refprobe = omni_vars[refener_sc_loc]

    nt_ref, ne_ref = [time_refprobe.shape[0], ener_refprobe.shape[1]]

    # time x energy x spacecraft
    omni_spec_data = np.full([nt_ref, ne_ref, len(omni_vars)], np.nan)

    # Paired samples (they were paired by index) at the closest channels
    for i_p, flux_ in enumerate(omni_vars):
        idx_closest = _idx_closest(flux_.energy.data, ener_refprobe.energy.data)
        idx_time = _paired_samples(time_refprobe, flux_)
        paired = idx_time >= 0
        omni_spec_data[paired, :, i_p] = flux_.data[idx_time[paired]][:, idx_closest]

    # Average (or sum) omni flux over all spacecraft
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method.lower() == "mean":
            omni_spec = np.nanmean(omni_spec_data, axis=2)
        else:
            omni_spec = np.nansum(omni_spec_data, axis=2)
            omni_spec[np.all(np.isnan(omni_spec_data), axis=2)] = np.nan

    omni_spec = xr.DataArray(
        omni_spec,
        coords=[time_refprobe.time.data, ener_refprobe.energy.data],
        dims=["time", "energy"],
        attrs={**time_refprobe.attrs},
    )

    for key in ["spin", "sector"]:
        if key in time_refprobe.coords:
            omni_spec = omni_spec.assign_coords({key: time_refprobe[key]})

    # Energy widths of the energies of the output (they were read as
    # energy_dplus and energy_dminus: KeyError)
    for key in ["delta_energy_plus", "delta_energy_minus"]:
        if key in ener_refprobe.attrs:
            omni_spec.attrs[key] = ener_refprobe.attrs[key]
        else:
            omni_spec.attrs.pop(key, None)

    return omni_spec
