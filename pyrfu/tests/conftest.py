#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import importlib
import random
import sys

# 3rd party imports
import numpy as np
import pytest

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
def _mms_config(monkeypatch, tmp_path):
    # Never the MMS configuration of the user: a file in the temporary
    # directory of the test, the default settings until a test writes it
    # (resource "local", so that the tests without source read their mocked
    # local files). The tests of db_init patch MMS_CFG_PATH too.
    db_init = importlib.import_module("pyrfu.mms.db_init")
    path = str(tmp_path / "mms_config.json")
    monkeypatch.setattr(db_init, "MMS_CFG_PATH", path, raising=False)


@pytest.fixture(autouse=True)
def _no_mms_cache(monkeypatch, tmp_path_factory):
    # The tests never use the MMS file cache of the user: off, in a temporary
    # directory (the cache tests turn it on there)
    # (the module, not the db_init function of pyrfu.mms)
    cache = importlib.import_module("pyrfu.mms._internal.cache")
    db_init = importlib.import_module("pyrfu.mms.db_init")
    cache_dir = str(tmp_path_factory.getbasetemp() / "mms_cache")

    def read_config():
        config = db_init._read_config()  # pylint: disable=protected-access
        config["cache"] = {"enabled": False, "size": 1e9, "max_age": 4.0}
        config["cache"]["dir"] = cache_dir
        return config

    monkeypatch.setattr(cache, "_read_config", read_config)


@pytest.fixture(autouse=True)
def _seed_and_close_figures():
    # Each test starts from the same random state, whatever ran before it,
    # and closes the figures it opened
    random.seed(0)
    np.random.seed(0)
    yield
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
