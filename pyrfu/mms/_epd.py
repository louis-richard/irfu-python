#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Helpers shared by the EPD routines (EIS, FEEPS, HPCA)."""

# Built-in imports
import warnings

# 3rd party imports
import numpy as np


def _spin_reduce(data, starts, stops, method: str = "mean"):
    r"""Mean or sum of `data` along its first axis (time) over each spin
    [starts[i], stops[i]), NaN-aware: NaN for a spin without data (also for
    the sum, which was 0, as data). The spins have the same number of samples,
    as the complete spins of EIS, FEEPS and HPCA (one array for all of them; it
    was a loop over the spins in each routine)."""
    data = np.asarray(data)
    starts, stops = np.asarray(starts), np.asarray(stops)

    if len(starts) == 0:
        return np.empty((0, *data.shape[1:]), dtype=np.float64)

    length = stops[0] - starts[0]

    if np.any(stops - starts != length):
        raise ValueError("the spins must have the same number of samples")

    # (spin, sample in the spin, ...)
    spins = data[starts[:, None] + np.arange(length)]

    if method == "sum" and spins.dtype.kind in "iub":
        return spins.sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method == "sum":
            return np.where(
                np.isfinite(spins).any(axis=1), np.nansum(spins, axis=1), np.nan
            )

        return np.nanmean(spins, axis=1)
