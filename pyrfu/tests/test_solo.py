#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# Built-in imports
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# 3rd party imports
import numpy as np
from ddt import data, ddt, unpack

# Local imports
from .. import solo
from ..solo.db_init import config_path

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.4"
__status__ = "Prototype"


def _use_temp_config(test_case):
    r"""Use a temporary user configuration directory for the test, so that
    the user's and the package's configuration files are never written."""
    config_dir = tempfile.TemporaryDirectory()
    test_case.addCleanup(config_dir.cleanup)
    patch = mock.patch(
        "pyrfu._user_config.platformdirs.user_config_dir",
        return_value=config_dir.name,
    )
    patch.start()
    test_case.addCleanup(patch.stop)
    return config_dir.name


def _use_empty_data_dir(test_case):
    r"""Point the SolO configuration at an empty temporary directory for the
    test, so that the default data path doesn't depend on the user's
    configuration."""
    _use_temp_config(test_case)
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
    def setUp(self):
        self.config_dir = _use_temp_config(self)

    def test_db_init_user_config(self):
        # Saved in the user configuration directory: the configuration file of
        # the package (tracked, shipped in the wheel) used to be rewritten
        package_path = os.path.join(os.path.dirname(solo.__file__), "config.json")
        with open(package_path, "rb") as file:
            package_config = file.read()

        with tempfile.TemporaryDirectory() as data_dir:
            solo.db_init(data_dir)
            path = config_path()
            with open(path, encoding="utf-8") as file:
                config = json.load(file)

        self.assertEqual(path, os.path.join(self.config_dir, "solo_config.json"))
        self.assertEqual(config["local_data_dir"], os.path.normpath(data_dir))
        with open(package_path, "rb") as file:
            self.assertEqual(file.read(), package_config)

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
