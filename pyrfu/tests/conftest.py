#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
import sys

# 3rd party imports
import numpy as np
import pytest

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.1"
__status__ = "Prototype"

# Seeded at import, before pytest collects the test modules: the random
# parametrisations (@data(random.random(), ...)) are drawn at collection
random.seed(0)
np.random.seed(0)


@pytest.fixture(autouse=True)
def _seed_and_close_figures():
    # Each test starts from the same random state, whatever ran before it,
    # and closes the figures it opened (more than 20 were left open)
    random.seed(0)
    np.random.seed(0)
    yield
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
