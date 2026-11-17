#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from .db_get_ts import _db_get_ts_dict

# Local imports
from .feeps_active_eyes import feeps_active_eyes

data_units_keys = {
    "flux": "intensity",
    "counts": "counts",
    "cps": "count_rate",
    "mask": "sector_mask",
    "fluxerr": "intensity_error",
}

# Variables of the L2 files from which the counts and the intensity errors
# are derived (percent_error = 100 / sqrt(counts))
_DERIVED = {
    "counts": ["percent_error", "count_rate"],
    "intensity_error": ["percent_error", "count_rate", "intensity"],
}


def _counts(percent_error, count_rate):
    r"""Counts (100 / percent_error)^2 of the L2 files, 0 where the count rate
    is 0 (percent_error 100 or a negative fill value there), NaN where the
    count rate is NaN (no data: percent_error 100 there, 1 count) or
    percent_error is not valid with counts (quality indicator 3)."""
    valid = np.isfinite(percent_error) & (percent_error > 0)
    counts = np.full(percent_error.shape, np.nan)
    np.divide(100.0, percent_error, out=counts, where=valid)
    counts = np.round(counts**2)

    counts = np.where(count_rate == 0, 0.0, counts)

    return np.where(np.isnan(count_rate), np.nan, counts)


def _derived_eye(name, data, e_id, cdf_name):
    r"""Counts (name "counts") or intensity errors ("intensity_error") of the
    eye e_id from its L2 variables in data, with the coordinates and
    attributes of its count rate (intensity)."""
    percent_error = np.asarray(data[cdf_name(e_id, "percent_error")].data, float)
    count_rate = data[cdf_name(e_id, "count_rate")]
    counts = _counts(percent_error, np.asarray(count_rate.data, float))

    if name == "counts":
        attrs = {**count_rate.attrs, "UNITS": "counts"}
        return count_rate.copy(data=counts).assign_attrs(attrs)

    # Relative error of the intensity, that of the counts (a constant factor
    # per channel between the two); 0 without counts
    intensity = data[cdf_name(e_id, "intensity")]
    error = np.where(
        counts == 0, 0.0, np.asarray(intensity.data, float) * percent_error / 100
    )
    return intensity.copy(data=np.where(np.isnan(counts), np.nan, error))


def _eye_cdf_name(pref, name, e_id, mms_id):
    # Name of the variable name (data, quality indicator, energy bounds) of
    # the eye e_id ("top-1", "bottom-12", ...)
    suf, sensor_id = e_id.split("-")
    return f"mms{mms_id:d}_{pref}_{suf}_{name}_sensorid_{sensor_id}"


def get_feeps_alleyes(
    tar_var,
    tint,
    mms_id,
    verbose: bool = True,
    data_path: str = "",
    source: str = "default",
    quality_flag: int | None = 3,
):
    r"""Read energy spectrum of the selected specie in the selected energy
    range for all FEEPS eyes.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like
        {data_unit}{specie}_{data_rate}_{data_lvl}, e.g. "fluxe_brst_l2". The
        units are "flux" (intensity, 1/(cm^2 s sr keV)), "cps" (count rate),
        "counts", "fluxerr" (error of the intensity, L2) and "mask" (sector
        masks).
    tint : array_like or xarray.DataArray
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`.
    mms_id : int or float or str
        Index of the spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Path of MMS data. Default uses the local path of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to fetch the data from. Default uses the default resource of
        the MMS configuration. Each file is read (downloaded) once for all the
        eyes.
    quality_flag : int or None, Optional
        The samples of an eye whose quality indicator is greater than or equal
        to `quality_flag` are set to NaN (L2 data only, not for the sector
        masks). Default is 3, which removes the samples not valid for science
        (3) and the calibration data (4), as in pyspedas. None keeps all the
        samples.

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the available eyes of the
        Fly's Eye Energetic Particle Spectrometer (FEEPS). For the sector masks
        ("mask"), the onboard mask of the 64 spin sectors of each eye
        (``sectornum``) at each time.

    Raises
    ------
    ValueError
        If the units or the species in `tar_var` are invalid, or if the
        intensity errors are requested from other than the L2 files.

    Notes
    -----
    The FEEPS quality indicators are, for each eye and sample, 0 (valid for
    science), 1 (caution: survey sector partly masked onboard), 2 (problematic:
    less than 50 % of the survey sector contaminated), 3 (not valid for
    science: contaminated, or survey sector fully masked onboard, with zero
    counts) and 4 (calibration). The MMS CMAD (section 6.4.3) recommends not
    to use 3 and 4 for science. IDL SPEDAS doesn't use the quality indicators.

    The L2 files have no counts: the L2 counts ("counts") are derived from
    the statistical error ``percent_error`` = 100 / sqrt(N) of the files,
    N = (100 / percent_error)^2 (an integer), 0 where the count rate is 0,
    NaN where the count rate is NaN (no data) or ``percent_error`` is not
    valid with a count rate (quality indicator 3). The intensity errors
    ("fluxerr") are intensity * percent_error / 100, 0 without counts: the
    intensity is the count rate times a calibration factor of each eye and
    channel, so that its relative error is that of the counts. The L1b counts
    are read from the L1b files. IDL SPEDAS and pyspedas don't use
    ``percent_error``.

    The data are clipped to `tint` when they are read, so the spin averages
    (`feeps_spin_avg`, `feeps_pad_spinavg`) and the sector spectrograms
    (`feeps_sector_spec`), which use complete spins only, leave out the
    partial spins at the start and end of `tint`. IDL SPEDAS and pyspedas make
    their products from the whole files and clip them only when asked
    (``time_clip``). To get the spins at the edges, read an interval longer by
    one spin (about 20 s) on each side and clip the products with
    `pyrfu.pyrf.time_clip`.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    Read electron energy spectrum for all FEEPS eyes

    >>> feeps_all_eyes = mms.get_feeps_alleyes("fluxe_brst_l2", tint_brst, 2)

    """

    mms_id = int(mms_id)
    specie = tar_var.split("_")[0][-1]

    var = {
        "tmmode": tar_var.split("_")[1],
        "lev": tar_var.split("_")[2],
        "mmsId": mms_id,
        "units_name": tar_var.split("_")[0][:-1].lower(),
    }

    if specie == "e":
        var["dtype"] = "electron"
        var["species"] = "e"
    elif specie == "i":
        var["dtype"] = "ion"
        var["species"] = "i"
    else:
        raise ValueError("Invalid specie")

    if var["units_name"] not in data_units_keys:
        raise ValueError(
            f"Invalid units {var['units_name']}: must be one of "
            f"{', '.join(data_units_keys)}"
        )

    data_units = data_units_keys[var["units_name"]]

    # Counts and intensity errors derived from percent_error (L2)
    derived = data_units in _DERIVED and var["lev"] == "l2"

    if data_units == "intensity_error" and not derived:
        raise ValueError(
            f"FEEPS intensity errors are derived from the L2 files: use "
            f"fluxerr{specie}_{var['tmmode']}_l2"
        )

    dset_name = f"mms{mms_id:d}_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"
    pref = f"epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    active_eyes = feeps_active_eyes(var, tint, mms_id)

    e_ids = [f"{k}-{s:d}" for k in active_eyes for s in active_eyes[k]]

    def eye_cdf_name(e_id, name):
        return _eye_cdf_name(pref, name, e_id, mms_id)

    # Read all the variables at once (one download per file from the SDC or AWS)
    cdf_names = {
        "spinsectnum": f"mms{mms_id:d}_{pref}_spinsectnum",
        "pitch_angle": f"mms{mms_id:d}_{pref}_pitch_angle",
    }

    if derived:
        sources = {
            f"{name}_{e_id}": eye_cdf_name(e_id, name)
            for e_id in e_ids
            for name in _DERIVED[data_units]
        }
    else:
        sources = {e_id: eye_cdf_name(e_id, data_units) for e_id in e_ids}

    cdf_names.update(sources)

    # Quality indicators of the eyes (L2 only)
    quality_cut = (
        quality_flag is not None and var["lev"] == "l2" and data_units != "sector_mask"
    )

    if quality_cut:
        for e_id in e_ids:
            cdf_names[f"quality_{e_id}"] = _eye_cdf_name(
                pref, "quality_indicator", e_id, mms_id
            )

    # Lower and upper bounds of the energy channels of the eyes (one record)
    bounds = {}

    if data_units != "sector_mask":
        for e_id in e_ids:
            for bound in ["lower", "upper"]:
                bounds[f"energy_{bound}_bound_{e_id}"] = _eye_cdf_name(
                    pref, f"energy_{bound}_bound", e_id, mms_id
                )

    data = _db_get_ts_dict(
        dset_name,
        list(cdf_names.values()),
        tint,
        verbose,
        data_path,
        source,
        support_names=list(bounds.values()),
    )
    out_dict = {
        key: data[cdf_name]
        for key, cdf_name in cdf_names.items()
        if not (derived and key in sources)
    }

    if derived:
        for e_id in e_ids:
            out_dict[e_id] = _derived_eye(data_units, data, e_id, eye_cdf_name)

    for e_id in e_ids:
        eye = out_dict[e_id]

        if quality_cut:
            # samples not to be used for science set to NaN
            bad = out_dict.pop(f"quality_{e_id}").data >= quality_flag
            eye_data = eye.data.astype(np.promote_types(eye.dtype, np.float32))
            eye = eye.copy(data=np.where(bad[:, None], np.nan, eye_data))

        eye.attrs["tmmode"] = var["tmmode"]
        eye.attrs["lev"] = var["lev"]
        eye.attrs["mms_id"] = mms_id
        eye.attrs["dtype"] = var["dtype"]
        eye.attrs["species"] = f"{var['dtype']}s"
        # spin sectors (masks) are the same for all eyes, energies are not
        if data_units == "sector_mask":
            out_dict[e_id] = eye.rename({eye.dims[1]: "sectornum"})
        else:
            eye = eye.rename({eye.dims[1]: f"energy_{e_id}"})

            # channel bounds along the energies, when the files have them
            for bound in ["lower", "upper"]:
                name = f"energy_{bound}_bound_{e_id}"
                if bounds[name] in data:
                    coord = (f"energy_{e_id}", data[bounds[name]])
                    eye = eye.assign_coords({name: coord})

            out_dict[e_id] = eye

    out = xr.Dataset(out_dict)

    out.attrs = var

    return out
