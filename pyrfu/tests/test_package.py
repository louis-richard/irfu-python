#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import importlib
import logging
import os
import tempfile
import unittest
from unittest import mock

# Local imports
import pyrfu

from .. import _user_config

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2026"
__license__ = "MIT"
__version__ = "2.5.12"
__status__ = "Prototype"


class PackageLoggerTestCase(unittest.TestCase):
    """Tests of the logger of pyrfu."""

    def test_logger_handler_once(self):
        """The handler of the pyrfu logger is added once, not again when the
        package is imported again (reload); the root logger is not
        configured."""
        logger = logging.getLogger("pyrfu")
        handlers = list(logger.handlers)
        root_handlers = list(logging.getLogger().handlers)

        importlib.reload(pyrfu)

        self.assertListEqual(logger.handlers, handlers)
        self.assertListEqual(logging.getLogger().handlers, root_handlers)


class UserConfigTestCase(unittest.TestCase):
    """Tests of the user configuration files."""

    def test_user_config_path(self):
        """Copied from the package configuration the first time, then kept;
        the package configuration if the user directory cannot be written."""
        with tempfile.TemporaryDirectory() as root:
            package_cfg = os.path.join(root, "package.json")
            with open(package_cfg, "w", encoding="utf-8") as file:
                file.write('{"a": 1}')

            user_dir = os.path.join(root, "user")
            with mock.patch.object(
                _user_config.platformdirs, "user_config_dir", return_value=user_dir
            ):
                with mock.patch.object(
                    _user_config.shutil, "copyfile", side_effect=OSError("read-only")
                ):
                    path = _user_config.user_config_path("c.json", package_cfg)
                self.assertEqual(path, package_cfg)

                path = _user_config.user_config_path("c.json", package_cfg)
                self.assertEqual(path, os.path.join(user_dir, "c.json"))
                with open(path, encoding="utf-8") as file:
                    self.assertEqual(file.read(), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
