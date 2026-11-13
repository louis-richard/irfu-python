#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast

# Built-in imports
import importlib
import inspect
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import typing
import unittest
from unittest import mock

# Local imports
import pyrfu
from pyrfu._core import user_config as _user_config


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

    def test_local_data_dir(self):
        """The local data directory of the MAVEN and SolO configurations: set
        after the check of the path (no file for a wrong path), at once (a
        failed write leaves the previous file)."""
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "c.json")
            config_path = mock.Mock(return_value=path)
            with self.assertRaises(FileNotFoundError):
                _user_config.set_local_data_dir(config_path, os.path.join(root, "x"))
            config_path.assert_not_called()

            with open(path, "w", encoding="utf-8") as file:
                file.write('{"local_data_dir": "."}')
            _user_config.set_local_data_dir(config_path, root)
            self.assertEqual(_user_config.local_data_dir(path), os.path.normpath(root))

            with mock.patch.object(_user_config.os, "replace", side_effect=OSError):
                with self.assertRaises(OSError):
                    _user_config.set_local_data_dir(config_path, root + os.sep)
            self.assertEqual(_user_config.local_data_dir(path), os.path.normpath(root))
            self.assertListEqual(os.listdir(root), ["c.json"])


class ImportSideEffectsTestCase(unittest.TestCase):
    """What import pyrfu does."""

    def test_import_writes_no_file_and_loads_no_boto3(self):
        """import pyrfu created the MMS configuration file in the user
        directory and loaded boto3 (about 0.45 s): the file is created on the
        first use of pyrfu.mms.MMS_CFG_PATH, boto3 when S3 is listed."""
        code = textwrap.dedent("""
            import os, sys, tempfile
            from unittest import mock
            import platformdirs

            config_dir = tempfile.mkdtemp()
            with mock.patch.object(
                platformdirs, "user_config_dir", return_value=config_dir
            ):
                import pyrfu
                assert os.listdir(config_dir) == [], os.listdir(config_dir)
                assert "boto3" not in sys.modules
                from pyrfu.mms import MMS_CFG_PATH
                assert MMS_CFG_PATH == os.path.join(config_dir, "mms_config.json")
                assert os.listdir(config_dir) == ["mms_config.json"]
            """)
        root = os.path.dirname(os.path.dirname(os.path.abspath(pyrfu.__file__)))
        env = {**os.environ, "PYTHONPATH": root}
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mms_cfg_path_on_first_use(self):
        """MMS_CFG_PATH is not a module constant any more: the module
        __getattr__ of pyrfu.mms and pyrfu.mms.db_init computes it."""
        db_init = importlib.import_module("pyrfu.mms.db_init")
        self.assertNotIn("MMS_CFG_PATH", vars(db_init))
        self.assertEqual(pyrfu.mms.MMS_CFG_PATH, db_init.MMS_CFG_PATH)
        self.assertTrue(os.path.isfile(pyrfu.mms.MMS_CFG_PATH))

        for module in [pyrfu.mms, db_init]:
            with self.assertRaises(AttributeError):
                getattr(module, "no_such_name")


class LiteralFlagsTestCase(unittest.TestCase):
    """The Literal annotations of the closed sets of flags."""

    def test_literals_match_docstrings(self):
        """The values of each Literal parameter of the public functions are
        those of the set in its docstring ("name : {...}"), so that the two
        don't drift apart."""
        n_literals = 0

        for sub in pyrfu.__all__:
            module = getattr(pyrfu, sub)

            for name in getattr(module, "__all__", []):
                func = getattr(module, name)

                if not inspect.isfunction(func):
                    continue

                hints = typing.get_type_hints(func)
                doc = inspect.getdoc(func) or ""

                for param, hint in hints.items():
                    values = [
                        v
                        for arg in typing.get_args(hint) or (hint,)
                        if typing.get_origin(arg) is typing.Literal
                        for v in typing.get_args(arg)
                    ]
                    if typing.get_origin(hint) is typing.Literal:
                        values = list(typing.get_args(hint))

                    if not values:
                        continue

                    n_literals += 1
                    # "param : {...}", also in a group "a, param : {...}"
                    names = rf"(?:[a-z_0-9, ]+, )?{param}(?:, [a-z_0-9, ]+)?"
                    match = re.search(rf"^\s*{names} : \{{([^}}]*)\}}", doc, re.M)
                    with self.subTest(func=f"{sub}.{name}", param=param):
                        self.assertIsNotNone(match)
                        documented = ast.literal_eval(
                            "[" + " ".join(match.group(1).split()) + "]"
                        )
                        self.assertEqual(
                            set(values), {v for v in documented if v is not None}
                        )

        # The flags annotated in 2.5.14 (38 parameters)
        self.assertGreaterEqual(n_literals, 38)


if __name__ == "__main__":
    unittest.main()
