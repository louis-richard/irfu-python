#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Louis Richard
from .igrf import igrf
from .ion_anisotropy_thresh import ion_anisotropy_thresh
from .magnetopause_normal import magnetopause_normal

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["igrf", "magnetopause_normal", "ion_anisotropy_thresh"]
