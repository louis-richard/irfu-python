#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.8"
__status__ = "Prototype"

# Shared by the EIS routines for fluxes only (eis_combine_proton_spec,
# eis_moments, eis_proton_correction)


def _check_flux(*inps):
    r"""ValueError unless the inputs (DataArrays) have the same units, a
    differential particle flux, as the flux of the EIS files,
    1/(cm^2 s sr keV) (counts are ' ', cps 1/s). Inputs without UNITS (built
    by hand) are not checked."""
    units = sorted({inp.attrs["UNITS"] for inp in inps if "UNITS" in inp.attrs})

    if len(units) > 1:
        raise ValueError(f"the inputs have different units: {units}")

    if units and "cm^2" not in units[0]:
        raise ValueError(
            f"the data must be a flux, 1/(cm^2 s sr keV), not {units[0]!r}"
        )
