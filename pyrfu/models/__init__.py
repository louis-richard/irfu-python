#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Louis Richard
from .igrf import igrf
from .magnetopause_normal import magnetopause_normal

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

__all__ = ["igrf", "magnetopause_normal"]
