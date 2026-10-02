#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

from pyrfu import constants, dispersion, lp, maven, mms, models, plot, pyrf

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.4.21"
__status__ = "Prototype"

__all__ = [
    "constants",
    "dispersion",
    "lp",
    "maven",
    "mms",
    "models",
    "plot",
    "pyrf",
]

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)
