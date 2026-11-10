#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import importlib
import json
import os
import tempfile
import unittest
from unittest import mock

# 3rd party imports
import requests

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

    @staticmethod
    def _session(files, error=None):
        # LASP mocked: the search returns the files, a download the URL as
        # content (error: of the downloads)
        session = mock.MagicMock()

        def get(url, **kwargs):
            if "file_info" in url:
                return mock.Mock(
                    json=lambda: {
                        "files": [{"file_name": f, "file_size": 1} for f in files]
                    }
                )

            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.iter_content.return_value = [url.encode()]
            if error is not None:
                response.raise_for_status.side_effect = error
            return response

        session.get.side_effect = get
        return session

    def _download(self, var, session=None, **kwargs):
        # The names of the files written in data_path
        module = importlib.import_module("pyrfu.maven.download_data")
        session = session or self._session(self.FILES)

        with tempfile.TemporaryDirectory() as data_path:
            with mock.patch.object(module, "_login_lasp", return_value=session):
                maven.download_data(
                    var,
                    ["2015-01-01T00:00", "2015-01-01T06:00"],
                    data_path=data_path,
                    **kwargs,
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

    def test_download_data_content(self):
        # The content downloaded, in year/month directories, nothing else (a
        # temporary copy of each file was left); timeouts on the requests
        session = self._session(self.FILES[3:])
        module = importlib.import_module("pyrfu.maven.download_data")
        with tempfile.TemporaryDirectory() as data_path:
            with mock.patch.object(module, "_login_lasp", return_value=session):
                maven.download_data(
                    {"inst": "swe", "level": "l2"},
                    ["2015-01-01T00:00", "2015-01-02T06:00"],
                    data_path=data_path,
                )

            out_file = os.path.join(data_path, "swe", "l2", "2015", "01", self.FILES[3])
            with open(out_file, encoding="utf-8") as file:
                self.assertTrue(file.read().endswith(f"download?file={self.FILES[3]}"))
            self.assertListEqual(os.listdir(os.path.dirname(out_file)), [self.FILES[3]])

        for call in session.get.call_args_list:
            self.assertIsNotNone(call.kwargs.get("timeout"))

        # Two days: the end date as given (one more day for one day)
        url = session.get.call_args_list[0].args[0]
        self.assertIn("start_date=2015-01-01&end_date=2015-01-02", url)

    def test_download_data_http_errors(self):
        # An error page is not saved as a file, a failed search is raised
        session = self._session(self.FILES[3:], requests.HTTPError("404"))
        with self.assertRaises(requests.HTTPError):
            self._download({"inst": "swe", "level": "l2"}, session)

        session = mock.MagicMock()
        session.get.return_value.raise_for_status.side_effect = requests.HTTPError(
            "500"
        )
        with self.assertRaises(requests.HTTPError):
            self._download({"inst": "swe", "level": "l2"}, session)

    def test_download_data_login_and_path(self):
        # Login not implemented (public data), logged; by default the local
        # data directory of the configuration
        module = importlib.import_module("pyrfu.maven.download_data")
        self.assertIsInstance(module._login_lasp(), requests.Session)

        with self.assertLogs("pyrfu.maven.download_data", "INFO") as logs:
            self._download({"inst": "swe", "level": "l2"}, login="user")
        self.assertIn("login not implemented", "".join(logs.output))

        with tempfile.TemporaryDirectory() as root:
            config = os.path.join(root, "maven_config.json")
            with open(config, "w", encoding="utf-8") as file:
                json.dump({"local_data_dir": root}, file)

            file = {"file_name": self.FILES[3]}
            var = {"inst": "swe", "level": "l2"}
            with mock.patch.object(module, "config_path", return_value=config):
                out_path, out_file, _ = module._make_path(file, var, "url/")

            self.assertEqual(out_path, os.path.join(root, "swe", "l2", "2015", "01"))
            self.assertEqual(out_file, os.path.join(out_path, self.FILES[3]))


if __name__ == "__main__":
    unittest.main()
