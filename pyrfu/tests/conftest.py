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

# In this conftest rather than in pyproject.toml, so that they also apply to the
# tests of the installed package (pytest --pyargs pyrfu.tests)
MARKERS = ["network: needs internet access (run with PYRFU_NETWORK_TESTS=1)"]

# NumPy floating-point warnings on the random test data (sqrt, arccos, log10 of
# out-of-domain values, means of empty slices)
FILTERWARNINGS = [
    "ignore:invalid value encountered:RuntimeWarning",
    "ignore:divide by zero encountered:RuntimeWarning",
    "ignore:overflow encountered:RuntimeWarning",
    "ignore:Mean of empty slice:RuntimeWarning",
]


def pytest_configure(config):
    r"""Register the markers and the warning filters of the tests."""
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)

    for warning_filter in FILTERWARNINGS:
        config.addinivalue_line("filterwarnings", warning_filter)


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
