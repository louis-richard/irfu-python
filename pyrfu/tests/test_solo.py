#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import random
import subprocess
import sys
import tempfile
import unittest

# 3rd party imports
import numpy as np
from ddt import data, ddt, unpack

# Local imports
from .. import solo

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.4"
__status__ = "Prototype"


def _use_empty_data_dir(test_case):
    r"""Point the SolO configuration at an empty temporary directory for the
    test, and restore the configuration file of the package afterwards, so
    that the default data path doesn't depend on the user's configuration."""
    config_path = os.path.join(os.path.dirname(solo.__file__), "config.json")

    with open(config_path, "rb") as file:
        config = file.read()

    def restore():
        with open(config_path, "wb") as file:
            file.write(config)

    test_case.addCleanup(restore)

    data_dir = tempfile.TemporaryDirectory()
    test_case.addCleanup(data_dir.cleanup)
    solo.db_init(data_dir.name)


class SoloImportTestCase(unittest.TestCase):
    def test_import_pyrfu_solo(self):
        # In a fresh interpreter, as this module already imports pyrfu.solo
        code = "import pyrfu; print(pyrfu.solo.read_tnr.__name__)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            text=True,
        )
        self.assertEqual(result.stdout, "read_tnr\n")


class DbInitTestCase(unittest.TestCase):
    # db_init rewrites the SolO configuration file of the package: restore it
    def setUp(self):
        self.config_path = os.path.join(os.path.dirname(solo.__file__), "config.json")
        with open(self.config_path, "rb") as file:
            config = file.read()

        def restore():
            with open(self.config_path, "wb") as file:
                file.write(config)

        self.addCleanup(restore)

    def test_db_init_inpput(self):
        with self.assertRaises(AssertionError):
            solo.db_init("/Volumes/solo/remote/data")

    def test_db_init_output(self):
        self.assertIsNone(solo.db_init(os.getcwd()))


@ddt
class ReadLFRDensityTestCase(unittest.TestCase):
    @data(
        ([], ".", False),
        ([np.datetime64("2023-01-01T00:00:00"), "2023-01-01T00:10:00"], ".", False),
        (["2023-01-01T00:00:00", np.datetime64("2023-01-01T00:10:00")], ".", False),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], "/bazinga", False),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], ".", "i am groot"),
    )
    @unpack
    def test_read_lfr_density_input(self, tint, data_path, tree):
        with self.assertRaises(AssertionError):
            solo.read_lfr_density(tint, data_path, tree)

    def test_read_lfr_density_output(self):
        _use_empty_data_dir(self)
        tint = ["2023-01-01T00:00:00", "2023-01-01T00:10:00"]
        self.assertIsNone(solo.read_lfr_density(tint))


@ddt
class ReadTNRTestCase(unittest.TestCase):
    @data(
        ([], 1, "."),
        ([np.datetime64("2023-01-01T00:00:00"), "2023-01-01T00:10:00"], 1, "."),
        (["2023-01-01T00:00:00", np.datetime64("2023-01-01T00:10:00")], 1, "."),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], random.random(), "."),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], 1, "/bazinga"),
    )
    @unpack
    def test_read_tnr_input(self, tint, sensor, data_path):
        with self.assertRaises(AssertionError):
            solo.read_tnr(tint, sensor, data_path)

    @data(
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], 1, ""),
        (["2023-01-01T00:00:00", "2023-01-01T00:10:00"], 2, ""),
    )
    @unpack
    def test_read_tnr_output(self, tint, sensor, data_path):
        _use_empty_data_dir(self)
        self.assertIsNone(solo.read_tnr(tint, sensor, data_path))


if __name__ == "__main__":
    unittest.main()
