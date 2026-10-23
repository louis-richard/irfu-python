#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from ..pyrf.ts_scalar import ts_scalar
from .correct_edp_probe_timing import correct_edp_probe_timing
from .dft_time_shift import dft_time_shift

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.6"
__status__ = "Prototype"

# Tip-to-tip lengths of the spin-plane and axial booms in m: nominal, and
# with the boom shortening of Pedersen et al. (1998), as irfu-matlab
# mms.fk_powerspec_E80
_BOOMS = {False: (120.0, 29.2), True: (96.0, 14.5)}


def _dft_timing(time, pot):
    r"""Probe potentials at the times of V1, the channel delays of the burst
    sampling undone with a DFT shift, as irfu-matlab mms.fk_powerspectrum
    (correct_edp_probe_timing interpolates)."""

    def _shifted(data, delay_ns):
        # Sampled delay_ns after the time tags: the values at the time tags
        delay = np.timedelta64(delay_ns, "ns")
        return dft_time_shift(ts_scalar(time + delay, data), -delay_ns * 1e-9).data

    # V3 and V5, and the differences V1 - V2, V3 - V4 and V5 - V6 (E12, E34
    # and E56 times their lengths, which cancel), sampled after V1
    v_3 = _shifted(pot[:, 2], 7629)
    v_5 = _shifted(pot[:, 4], 15259)
    dv_12 = _shifted(pot[:, 0] - pot[:, 1], 26703)
    dv_34 = _shifted(pot[:, 2] - pot[:, 3], 30518)
    dv_56 = _shifted(pot[:, 4] - pot[:, 5], 34332)

    return np.column_stack(
        [pot[:, 0], pot[:, 0] - dv_12, v_3, v_3 - dv_34, v_5, v_5 - dv_56]
    )


def probe_fields(
    v6: DataArray, timing: str = "dft", boom_shortening: bool = False
) -> Dataset:
    r"""Electric fields from the potentials of the six probes of MMS EDP, for
    probe interferometry: the single-probe fields (reconstructed from each
    probe and the spacecraft potential), the diagonal fields (between
    neighbouring spin-plane probes) and the double-probe fields, as
    irfu-matlab ``mms.fk_powerspectrum`` and ``mms.fk_powerspec_E80``.

    Parameters
    ----------
    v6 : xarray.DataArray
        Time series of the probe-to-spacecraft potentials of the six probes
        in V (time, 6), e.g., ``mms.get_data("v6_edp_brst_l2", ...)``,
        uniformly sampled.
    timing : {"dft", "resample", None}, Optional
        Correction of the channel delays of the burst sampling (V3 7.629 us,
        V5 15.259 us, E12 26.703 us, E34 30.518 us and E56 34.332 us after
        V1), to the times of V1: by a DFT shift (``mms.dft_time_shift``), as
        ``mms.fk_powerspectrum`` (default); by interpolation, as
        ``mms.correct_edp_probe_timing``; or None for potentials already
        corrected (or not burst).
    boom_shortening : bool, Optional
        Use the effective boom lengths of Pedersen et al. (1998) [1]_, 96 m
        (spin plane) and 14.5 m (axial), as ``mms.fk_powerspec_E80``, instead
        of the tip-to-tip lengths 120 m and 29.2 m. Default is False.

    Returns
    -------
    xarray.Dataset
        * v : the potentials of the six probes (time, probe) in V, corrected
          for the channel delays.
        * v_sc : the spacecraft potential estimated by each spin-plane boom,
          (V1 + V2) / 2 and (V3 + V4) / 2 (time, pair) in V.
        * e_probe : the single-probe fields E1, ..., E6 (time, probe) in mV/m:
          (V1 - V34) / (L / 2), (V34 - V2) / (L / 2), (V3 - V12) / (L / 2),
          (V12 - V4) / (L / 2), and for the axial probes (V5 - V_sc) /
          (L_axial / 2) and (V_sc - V6) / (L_axial / 2), with V12 and V34 the
          estimates of v_sc and V_sc their mean.
        * e_diag : the diagonal fields E13, E14, E23 and E24 (time, diag) in
          mV/m, (Vi - Vj) / d with d = L / sqrt(2) the distance between
          neighbouring spin-plane probes.
        * e_pair : the double-probe fields E12, E34 and E56 (time, boom) in
          mV/m, (V1 - V2) / L, (V3 - V4) / L, (V5 - V6) / L_axial.

        The attributes give the lengths used (l_spin, l_axial, d_diag, in m)
        and the timing correction.

    Raises
    ------
    TypeError
        If v6 is not a DataArray.
    ValueError
        If v6 does not have six probes, if timing is not "dft", "resample" or
        None, or if v6 has NaN with timing="dft" (the DFT spreads them).

    See Also
    --------
    pyrfu.mms.correct_edp_probe_timing, pyrfu.mms.dft_time_shift,
    pyrfu.mms.probe_align_times

    Notes
    -----
    Each field Eij is the component of the electric field along the
    direction from probe i to probe j (Vi - Vj over their distance), and the
    single-probe fields are those of their boom: E1 and E2 along the
    direction from probe 1 to probe 2, at the positions of probes 1 and 2,
    60 m apart; E3 and E4 along 3 to 4; E5 and E6 along 5 to 6. The fields
    are in the frame of the probes (not despun). The parallel diagonal fields
    used for interferometry are E13 and E42 = -E24, and E14 and E32 = -E23.

    The single-probe fields (reconstructed field interferometry) depend on
    the estimate of the spacecraft potential at the centre. Steinvall et al.
    (2022) [2]_ find them accurate for wave vectors within about 30 deg of the
    spin-plane boom, or 10 deg of the axial one, and the axial ones only in
    dense plasmas (n of 10 cm^-3 or more); the diagonal fields do not depend
    on the spacecraft potential and are accurate in the spin plane except
    near 45 deg from the booms.

    References
    ----------
    .. [1] Pedersen, A., Mozer, F., and Gustafsson, G. (1998), Electric field
           measurements in a tenuous plasma with spherical double probes, in
           Measurement Techniques in Space Plasmas: Fields, Geophys. Monogr.
           Ser., 103.
    .. [2] Steinvall, K., Khotyaintsev, Y. V., and Graham, D. B. (2022), On
           the applicability of single-spacecraft interferometry methods using
           electric field probes, J. Geophys. Res. Space Phys., 127,
           e2021JA030143, doi:10.1029/2021JA030143.

    Examples
    --------
    >>> from pyrfu import mms

    >>> tint = ["2017-05-05T20:09:50.80", "2017-05-05T20:09:50.86"]
    >>> v6 = mms.get_data("v6_edp_brst_l2", tint, 1)
    >>> fields = mms.probe_fields(v6)
    >>> e_5, e_6 = fields.e_probe.sel(probe=5), fields.e_probe.sel(probe=6)

    """

    if not isinstance(v6, DataArray):
        raise TypeError("v6 must be a xarray.DataArray")

    if v6.ndim != 2 or v6.shape[1] != 6:
        raise ValueError("v6 must have the potentials of the six probes (time, 6)")

    if timing not in ("dft", "resample", None):
        raise ValueError(f'timing must be "dft", "resample" or None, not {timing!r}')

    time = v6.time.data
    pot = np.asarray(v6.data, dtype=np.float64)

    # Potentials at the times of V1
    if timing == "dft":
        if not np.all(np.isfinite(pot)):
            raise ValueError('v6 has NaN: the DFT spreads them (timing="resample")')

        pot = _dft_timing(time, pot)
    elif timing == "resample":
        pot = correct_edp_probe_timing(v6).data

    l_spin, l_axial = _BOOMS[bool(boom_shortening)]
    d_diag = l_spin / np.sqrt(2)
    v_1, v_2, v_3, v_4, v_5, v_6 = pot.T

    # Spacecraft potential estimated by each spin-plane boom
    v_sc12, v_sc34 = (v_1 + v_2) / 2, (v_3 + v_4) / 2
    v_sc = (v_sc12 + v_sc34) / 2

    # Single-probe fields (mV/m): each probe against the estimate of the
    # other boom, over half its boom
    e_probe = np.column_stack(
        [
            (v_1 - v_sc34) / (l_spin / 2),
            (v_sc34 - v_2) / (l_spin / 2),
            (v_3 - v_sc12) / (l_spin / 2),
            (v_sc12 - v_4) / (l_spin / 2),
            (v_5 - v_sc) / (l_axial / 2),
            (v_sc - v_6) / (l_axial / 2),
        ]
    )

    # Diagonal fields between neighbouring spin-plane probes (mV/m)
    e_diag = (
        np.column_stack([(v_1 - v_3), (v_1 - v_4), (v_2 - v_3), (v_2 - v_4)]) / d_diag
    )

    # Double-probe fields along each boom (mV/m)
    e_pair = np.column_stack(
        [(v_1 - v_2) / l_spin, (v_3 - v_4) / l_spin, (v_5 - v_6) / l_axial]
    )

    out = xr.Dataset(
        {
            "v": (["time", "probe"], pot, {"UNITS": "V"}),
            "v_sc": (
                ["time", "pair"],
                np.column_stack([v_sc12, v_sc34]),
                {"UNITS": "V"},
            ),
            "e_probe": (["time", "probe"], e_probe * 1e3, {"UNITS": "mV/m"}),
            "e_diag": (["time", "diag"], e_diag * 1e3, {"UNITS": "mV/m"}),
            "e_pair": (["time", "boom"], e_pair * 1e3, {"UNITS": "mV/m"}),
        },
        coords={
            "time": time,
            "probe": np.arange(1, 7),
            "pair": ["12", "34"],
            "diag": ["13", "14", "23", "24"],
            "boom": ["12", "34", "56"],
        },
        attrs={
            "l_spin": l_spin,
            "l_axial": l_axial,
            "d_diag": d_diag,
            "timing": str(timing),
        },
    )

    return out
