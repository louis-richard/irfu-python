#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Physical constants used across pyrfu that are not in
:mod:`scipy.constants`, which should be used for the fundamental constants.
"""

__all__ = ["R_E"]

#: Earth radius [km]: the IGRF reference radius, as ``irf_units`` in
#: irfu-matlab. Positions in pyrfu are in km, so ``r_xyz / R_E`` gives them
#: in Earth radii.
R_E = 6371.2
