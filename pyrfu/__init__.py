#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging

from pyrfu import constants, dispersion, lp, maven, mms, models, plot, pyrf, solo

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
    "solo",
]

# Show the messages of pyrfu (INFO and above) without configuring the root
# logger, which belongs to the application. Silence them with
# logging.getLogger("pyrfu").setLevel(logging.WARNING).
_logger = logging.getLogger(__name__)

if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
        ),
    )
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
