#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os
import tempfile
import unittest
from unittest import mock

# Local imports
from .. import maven

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2026"
__license__ = "MIT"
__version__ = "2.4.21"
__status__ = "Prototype"


class DbInitTestCase(unittest.TestCase):
    def setUp(self):
        # Temporary user configuration directory: the user's and the
        # package's configuration files are never written
        config_dir = tempfile.TemporaryDirectory()
        self.addCleanup(config_dir.cleanup)
        self.config_dir = config_dir.name
        patch = mock.patch(
            "pyrfu._user_config.platformdirs.user_config_dir",
            return_value=self.config_dir,
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_db_init_user_config(self):
        # Saved in the user configuration directory: the configuration file of
        # the package (tracked, shipped in the wheel) used to be rewritten
        package_path = os.path.join(os.path.dirname(maven.__file__), "config.json")
        with open(package_path, "rb") as file:
            package_config = file.read()

        with tempfile.TemporaryDirectory() as data_dir:
            maven.db_init(data_dir)
            path = os.path.join(self.config_dir, "maven_config.json")
            with open(path, encoding="utf-8") as file:
                config = json.load(file)

        self.assertEqual(config["local_data_dir"], os.path.normpath(data_dir))
        with open(package_path, "rb") as file:
            self.assertEqual(file.read(), package_config)


if __name__ == "__main__":
    unittest.main()
