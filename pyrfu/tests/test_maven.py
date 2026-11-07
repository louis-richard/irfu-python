#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import importlib
import io
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


class DownloadDataTestCase(unittest.TestCase):
    FILES = [
        "mvn_mag_l2_2015001pl_20150101_v01_r01.sts",
        "mvn_mag_l2_2015001ss_20150101_v01_r01.sts",
        "mvn_mag_l2_2015001pl_20150101_v01_r01.xml",
        "mvn_swe_l2_svyspec_20150101_v04_r01.cdf",
    ]

    def _download(self, var):
        # LASP mocked: the search returns FILES, a download returns the file
        # name as content; the names of the files written in data_path
        module = importlib.import_module("pyrfu.maven.download_data")
        session = mock.MagicMock()

        def get(url, **_):
            if "file_info" in url:
                return mock.Mock(
                    json=lambda: {
                        "files": [{"file_name": f, "file_size": 1} for f in self.FILES]
                    }
                )

            response = mock.MagicMock()
            response.__enter__.return_value.raw = io.BytesIO(url.encode())
            return response

        session.get.side_effect = get

        with tempfile.TemporaryDirectory() as data_path:
            with mock.patch.object(module, "_login_lasp", return_value=session):
                maven.download_data(
                    var, ["2015-01-01T00:00", "2015-01-01T06:00"], data_path=data_path
                )

            return sorted(name for _, _, names in os.walk(data_path) for name in names)

    def test_download_data_plan(self):
        # MAG with a plan: its .sts files, as before
        result = self._download({"inst": "mag", "level": "l2", "plan": "pl"})
        self.assertListEqual(result, [self.FILES[0]])

        var = {"inst": "mag", "level": "l2", "plan": "pl", "file_extension": "xml"}
        self.assertListEqual(self._download(var), [self.FILES[2]])

    def test_download_data_without_plan(self):
        # Without plan (as documented: KeyError) all the files found, or those
        # of file_extension (the other instruments downloaded nothing)
        result = self._download({"inst": "swe", "level": "l2"})
        self.assertListEqual(result, sorted(self.FILES))

        var = {"inst": "swe", "level": "l2", "file_extension": "cdf"}
        self.assertListEqual(self._download(var), [self.FILES[3]])

        with self.assertRaises(FileNotFoundError):
            self._download({"inst": "mag", "level": "l2", "plan": "bazinga"})


if __name__ == "__main__":
    unittest.main()
