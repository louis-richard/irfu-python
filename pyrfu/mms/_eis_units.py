#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Shared by the EIS routines for fluxes only (eis_combine_proton_spec,
# eis_moments, eis_proton_correction) and by the combinations of spacecraft


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


def _check_same_attrs(inps, keys, name):
    r"""ValueError unless the inputs (DataArrays) have the same attributes
    `keys` (e.g. the species, the method of the PADs), or none of them (they
    were combined whatever they were)."""
    for key in keys:
        values = sorted({str(inp.attrs.get(key)) for inp in inps})

        if len(values) > 1:
            raise ValueError(f"The {name} have different {key}: {values}")
