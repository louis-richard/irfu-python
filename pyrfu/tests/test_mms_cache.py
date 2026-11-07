#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import importlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

# 3rd party imports
import numpy as np
import platformdirs

# Local imports
from pyrfu import mms, pyrf
from pyrfu.mms import _cache

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.10"
__status__ = "Prototype"

FGM_SRVY = "mms1_fgm_srvy_l2_20190914_v5.200.0.cdf"
FGM_BRST = "mms1_fgm_brst_l2_20190914075400_v5.200.0.cdf"
TINT = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]


class _CacheTestCase(unittest.TestCase):
    # The cache on, in a temporary directory
    def setUp(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.dir = os.path.join(tmp_dir.name, "cache")
        self.settings = {"enabled": True, "size": 1e9, "max_age": 4.0}
        self.settings["dir"] = self.dir
        patch = mock.patch.object(
            _cache, "_read_config", side_effect=lambda: {"cache": dict(self.settings)}
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _path(self, *parts):
        return os.path.join(self.dir, *parts)

    def _write(self, parts, content=b"0123456789", age=0.0):
        path = self._path(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as file:
            file.write(content)

        last_use = time.time() - age
        os.utime(path, (last_use, last_use))
        return path


class CachePathTestCase(unittest.TestCase):
    def test_mms_path(self):
        # The layout of the SDC: the day for burst only, the data type if any
        var = {"inst": "fgm", "tmmode": "brst", "lev": "l2", "dtype": ""}
        self.assertEqual(
            _cache.mms_path("1", var, FGM_BRST),
            f"mms1/fgm/brst/l2/2019/09/14/{FGM_BRST}",
        )

        var = {"inst": "fpi", "tmmode": "fast", "lev": "l2", "dtype": "dis-dist"}
        name = "mms2_fpi_fast_l2_dis-dist_20190914070000_v3.4.0.cdf"
        self.assertEqual(
            _cache.mms_path("2", var, name), f"mms2/fpi/fast/l2/dis-dist/2019/09/{name}"
        )

        # A name without time tag: in the directory of the product
        self.assertEqual(
            _cache.mms_path("1", var, "other.cdf"),
            "mms1/fpi/fast/l2/dis-dist/other.cdf",
        )

    def test_remote_file(self):
        remote = _cache.RemoteFile("aws", f"mms1/fgm/srvy/l2/2019/09/{FGM_SRVY}", 3)
        self.assertEqual(str(remote), FGM_SRVY)
        self.assertEqual(_cache.RemoteFile(**vars(remote)), remote)

    def test_file_size(self):
        self.assertEqual(_cache.file_size(12), 12)
        self.assertEqual(_cache.file_size("12"), 12)
        self.assertIsNone(_cache.file_size(None))
        self.assertIsNone(_cache.file_size("unknown"))


class CacheSettingsTestCase(unittest.TestCase):
    def test_settings(self):
        # The user cache directory by default; a given directory, expanded
        config = {"enabled": 1, "size": "5e9", "max_age": 24, "dir": ""}

        with mock.patch.object(_cache, "_read_config", return_value={"cache": config}):
            settings = _cache._settings()
            self.assertTrue(_cache.enabled())

        self.assertEqual(
            settings,
            {
                "enabled": True,
                "size": 5e9,
                "max_age": 24.0,
                "dir": os.path.join(platformdirs.user_cache_dir("pyrfu"), "mms"),
            },
        )

        config = dict(config, enabled=False, dir=os.path.join("~", "mms_cache"))

        with mock.patch.object(_cache, "_read_config", return_value={"cache": config}):
            settings = _cache._settings()
            self.assertFalse(_cache.enabled())

        self.assertEqual(
            settings["dir"], os.path.join(os.path.expanduser("~"), "mms_cache")
        )

    def test_settings_default_off_in_tests(self):
        # The tests never use the cache of the user (conftest)
        self.assertFalse(_cache.enabled())
        self.assertNotEqual(
            _cache._settings()["dir"],
            os.path.join(platformdirs.user_cache_dir("pyrfu"), "mms"),
        )


class CacheListingTestCase(_CacheTestCase):
    KEY = {"source": "aws", "tint": TINT}

    def test_listing(self):
        # Listed once, then from the cache; another key is listed again
        lister = mock.Mock(return_value=[{"path": "a"}])

        self.assertEqual(_cache.listing(self.KEY, lister), [{"path": "a"}])
        self.assertEqual(_cache.listing(self.KEY, lister), [{"path": "a"}])
        self.assertEqual(lister.call_count, 1)

        _cache.listing(dict(self.KEY, source="sdc"), lister)
        self.assertEqual(lister.call_count, 2)
        self.assertEqual(len(os.listdir(self._path("listings"))), 2)

    def test_listing_expired(self):
        # Made again after the maximum age
        lister = mock.Mock(return_value=[{"path": "a"}])
        _cache.listing(self.KEY, lister)
        path = _cache._listing_path(_cache._settings(), self.KEY)

        with open(path, "w", encoding="utf-8") as file:
            json.dump({"created": time.time() - 5 * 3600.0, "items": []}, file)

        self.assertEqual(_cache.listing(self.KEY, lister), [{"path": "a"}])
        self.assertEqual(lister.call_count, 2)

    def test_listing_empty_or_corrupted(self):
        # An empty listing is not kept (the files may come later); a corrupted
        # one is made again
        lister = mock.Mock(return_value=[])
        _cache.listing(self.KEY, lister)
        _cache.listing(self.KEY, lister)
        self.assertEqual(lister.call_count, 2)
        self.assertFalse(os.path.exists(self._path("listings")))

        path = _cache._listing_path(_cache._settings(), self.KEY)
        os.makedirs(os.path.dirname(path))

        with open(path, "w", encoding="utf-8") as file:
            file.write("{not json")

        lister.return_value = [{"path": "a"}]
        self.assertEqual(_cache.listing(self.KEY, lister), [{"path": "a"}])

    def test_listing_write_error(self):
        # A cache that cannot be written gives a warning and the listing
        with mock.patch.object(_cache, "_atomic_write", side_effect=OSError("full")):
            with self.assertLogs("pyrfu.mms._cache", "WARNING"):
                items = _cache.listing(self.KEY, lambda: [{"path": "a"}])

        self.assertEqual(items, [{"path": "a"}])


class CacheReadTestCase(_CacheTestCase):
    def setUp(self):
        super().setUp()
        self.remote = _cache.RemoteFile(
            "aws", f"mms1/fgm/srvy/l2/2019/09/{FGM_SRVY}", 3
        )
        self.path = self._path("aws", *self.remote.path.split("/"))

    def test_read(self):
        # Fetched once, then from the cache, in the layout of the SDC; the
        # sub-directory of the source is for the user only (all the levels
        # below were 0755: os.makedirs gives its mode to the last one), the
        # cache directory is left as it is (here readable by the group, in
        # the private temporary directory of the test); a read is a use
        os.makedirs(self.dir, mode=0o740)
        os.chmod(self.dir, 0o740)
        fetch = mock.Mock(return_value=b"cdf")

        self.assertEqual(_cache.read(self.remote, fetch), b"cdf")
        with open(self.path, "rb") as file:
            self.assertEqual(file.read(), b"cdf")

        os.utime(self.path, (1.0, 1.0))
        self.assertEqual(_cache.read(self.remote, fetch), b"cdf")
        self.assertEqual(fetch.call_count, 1)
        self.assertGreater(os.stat(self.path).st_mtime, time.time() - 60.0)
        self.assertListEqual(
            [n for n in os.listdir(os.path.dirname(self.path))], [FGM_SRVY]
        )

        if sys.platform != "win32":
            self.assertEqual(os.stat(self._path("aws")).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(self.dir).st_mode & 0o777, 0o740)
            _cache.listing({"k": 1}, lambda: [{"path": "a"}])
            self.assertEqual(os.stat(self._path("listings")).st_mode & 0o777, 0o700)

    def test_read_incomplete(self):
        # A cached file of another size than listed is read again
        self._write(["aws", *self.remote.path.split("/")], b"cd")
        fetch = mock.Mock(return_value=b"cdf")

        with self.assertLogs("pyrfu.mms._cache", "WARNING"):
            self.assertEqual(_cache.read(self.remote, fetch), b"cdf")

        self.assertEqual(fetch.call_count, 1)
        with open(self.path, "rb") as file:
            self.assertEqual(file.read(), b"cdf")

    def test_read_unknown_size(self):
        # Without the size of the listing, the cached file is used
        remote = _cache.RemoteFile("sdc", "ancillary/mms1/defatt/MMS1_DEFATT.V00")
        self._write(["sdc", "ancillary", "mms1", "defatt", "MMS1_DEFATT.V00"], b"x")
        fetch = mock.Mock()

        self.assertEqual(_cache.read(remote, fetch), b"x")
        fetch.assert_not_called()

    def test_read_larger_than_cache(self):
        # Not cached
        self.settings["size"] = 2
        self.assertEqual(_cache.read(self.remote, lambda: b"cdf"), b"cdf")
        self.assertFalse(os.path.exists(self.path))

    def test_read_write_error(self):
        # A cache that cannot be written gives a warning and the content
        with mock.patch.object(_cache, "_atomic_write", side_effect=OSError("full")):
            with self.assertLogs("pyrfu.mms._cache", "WARNING"):
                self.assertEqual(_cache.read(self.remote, lambda: b"cdf"), b"cdf")

        self.assertFalse(os.path.exists(self.path))

    def test_read_utime_error(self):
        # The time of last use cannot be refreshed: the cached file is used
        self._write(["aws", *self.remote.path.split("/")], b"cdf")

        with mock.patch.object(_cache.os, "utime", side_effect=OSError("read-only")):
            self.assertEqual(_cache.read(self.remote, mock.Mock()), b"cdf")

    def test_atomic_write(self):
        # No partial file is left when the write fails
        os.makedirs(os.path.dirname(self.path))

        with mock.patch.object(_cache.os, "replace", side_effect=OSError("full")):
            with self.assertRaises(OSError):
                _cache._atomic_write(self.path, b"cdf")

        self.assertListEqual(os.listdir(os.path.dirname(self.path)), [])

        _cache._atomic_write(self.path, "text")
        with open(self.path, "r", encoding="utf-8") as file:
            self.assertEqual(file.read(), "text")


class CacheEvictTestCase(_CacheTestCase):
    def test_evict_age(self):
        # The files and listings not used for the maximum age are removed, the
        # interrupted downloads too
        old = [
            self._write(["aws", "old.cdf"], age=5 * 3600.0),
            self._write(["listings", "old.json"], age=5 * 3600.0),
            self._write(["sdc", "tmp.part"], age=5 * 3600.0),
        ]
        recent = self._write(["sdc", "recent.cdf"], age=3 * 3600.0)

        _cache._evict(_cache._settings())

        self.assertFalse(any(os.path.exists(path) for path in old))
        self.assertTrue(os.path.exists(recent))

    def test_evict_size(self):
        # The least recently used first, until the cache is not larger than
        # its maximum size; downloads in progress are kept
        self.settings["size"] = 25
        paths = [self._write(["aws", f"{i}.cdf"], age=100.0 - i) for i in range(3)]
        part = self._write(["aws", "new.part"], age=200.0)

        _cache._evict(_cache._settings())

        self.assertListEqual(
            [os.path.exists(path) for path in paths], [False, True, True]
        )
        self.assertTrue(os.path.exists(part))

        # Files larger than the cache: all removed
        self.settings["size"] = 5
        _cache._evict(_cache._settings())
        self.assertListEqual([os.path.exists(path) for path in paths], [False] * 3)

    @unittest.skipIf(sys.platform == "win32", "symbolic links")
    def test_evict_vanished(self):
        # A file removed meanwhile (here a broken link) is skipped
        os.makedirs(self._path("aws"))
        os.symlink(self._path("aws", "missing"), self._path("aws", "link.cdf"))

        self.assertListEqual(_cache._cached_files(_cache._settings()), [])
        _cache._remove(self._path("aws", "missing"))


class CacheInfoTestCase(_CacheTestCase):
    def test_cache_info_clear_cache(self):
        # The files and the listings; clear_cache removes these only
        self._write(["aws", "a.cdf"])
        self._write(["sdc", "ancillary", "b.V00"], b"01234")
        self._write(["listings", "c.json"])
        other = self._write(["notes.txt"])

        self.assertDictEqual(
            mms.cache_info(),
            {
                "enabled": True,
                "dir": self.dir,
                "max_size": 1e9,
                "max_age": 4.0,
                "n_files": 2,
                "size": 15,
                "n_listings": 1,
            },
        )

        with self.assertLogs("pyrfu.mms._cache", "INFO"):
            mms.clear_cache()

        self.assertEqual(mms.cache_info()["n_files"], 0)
        self.assertEqual(mms.cache_info()["n_listings"], 0)
        self.assertTrue(os.path.exists(other))


class GetDataCacheTestCase(_CacheTestCase):
    def setUp(self):
        super().setUp()
        self.module = importlib.import_module("pyrfu.mms.get_data")
        self.config = {"aws": "", "sdc": {"rights": "public", "username": "username"}}
        time_ = pyrf.iso86012datetime64(np.array(TINT))
        self.ts = pyrf.ts_vec_xyz(time_, np.ones((2, 3)))

        for name, kwargs in [
            ("_read_config", {"side_effect": lambda: self.config}),
            ("get_ts", {"return_value": self.ts}),
        ]:
            patch = mock.patch.object(self.module, name, **kwargs)
            patch.start()
            self.addCleanup(patch.stop)

    def _aws(self):
        key = f"spdf/cdaweb/data/mms/mms1/fgm/srvy/l2/2019/09/{FGM_SRVY}"
        s3_obj = mock.Mock(bucket_name="bucket", key=key)
        files = [{"s3_obj": s3_obj, "full_name": key, "file_size": 3}]
        resource = mock.Mock()
        resource.Object.return_value.get.side_effect = lambda: {
            "Body": io.BytesIO(b"cdf")
        }
        patches = [
            mock.patch.object(self.module, "list_files_aws", return_value=files),
            mock.patch.object(self.module, "_s3_resource", return_value=resource),
        ]
        mocks = [patch.start() for patch in patches]

        for patch in patches:
            self.addCleanup(patch.stop)

        return mocks[0], resource

    def test_get_data_aws(self):
        # A second get_data reads no network, nor does another variable of the
        # same files (the same listing)
        list_files, resource = self._aws()

        for var_str in ["b_gse_fgm_srvy_l2", "b_gse_fgm_srvy_l2", "b_gsm_fgm_srvy_l2"]:
            out = mms.get_data(var_str, TINT, 1, verbose=False, source="aws")
            self.assertIs(out, self.ts)

        self.assertEqual(list_files.call_count, 1)
        resource.Object.assert_called_once_with(
            "bucket", f"spdf/cdaweb/data/mms/mms1/fgm/srvy/l2/2019/09/{FGM_SRVY}"
        )
        self.assertListEqual(
            [call.args[0] for call in self.module.get_ts.call_args_list], [b"cdf"] * 3
        )
        self.assertTrue(
            os.path.exists(
                self._path("aws", "mms1", "fgm", "srvy", "l2", "2019", "09", FGM_SRVY)
            )
        )

    def test_get_data_sdc(self):
        # The SDC login only if a file is read; the listings depend on the SDC
        # rights (SITL listings differ)
        files = [{"file_name": FGM_BRST, "size": "3", "url": "url"}]
        session = mock.Mock()
        session.get.return_value = mock.Mock(content=b"cdf")

        with (
            mock.patch.object(self.module, "list_files_sdc", return_value=files) as lst,
            mock.patch.object(
                self.module, "_login_lasp", return_value=(session, {}, "")
            ) as login,
        ):
            for _ in range(2):
                mms.get_data("b_gse_fgm_brst_l2", TINT, 1, verbose=False, source="sdc")

            self.assertEqual((lst.call_count, login.call_count), (1, 1))
            self.assertEqual(session.get.call_count, 1)

            self.config["sdc"] = {"rights": "sitl", "username": "louis"}
            mms.get_data("b_gse_fgm_brst_l2", TINT, 1, verbose=False, source="sdc")
            self.assertEqual((lst.call_count, login.call_count), (2, 1))

        self.assertTrue(
            os.path.exists(
                self._path(
                    "sdc", "mms1", "fgm", "brst", "l2", "2019", "09", "14", FGM_BRST
                )
            )
        )

    def test_download_sdc_session(self):
        # With the session of the caller, no login
        remote = _cache.RemoteFile("sdc", f"mms1/fgm/brst/l2/{FGM_BRST}", 3, "url")
        session = mock.Mock()
        session.get.return_value = mock.Mock(content=b"cdf")

        with mock.patch.object(self.module, "_login_lasp") as login:
            content = self.module._get_file_content_sources(
                "sdc", remote, session, {"h": "v"}
            )

        self.assertEqual(content, b"cdf")
        login.assert_not_called()
        self.assertEqual(session.get.call_args.kwargs["headers"], {"h": "v"})

    def test_get_data_cache_off(self):
        # As without cache: listed and read each time, nothing written
        self.settings["enabled"] = False
        list_files, _ = self._aws()
        s3_obj = list_files.return_value[0]["s3_obj"]
        s3_obj.get.side_effect = lambda: {"Body": io.BytesIO(b"cdf")}

        for _ in range(2):
            mms.get_data("b_gse_fgm_srvy_l2", TINT, 1, verbose=False, source="aws")

        self.assertEqual((list_files.call_count, s3_obj.get.call_count), (2, 2))
        self.assertFalse(os.path.exists(self.dir))

    def test_download_errors(self):
        # The errors of AWS and the SDC are raised, and the file is not cached
        list_files, resource = self._aws()
        resource.Object.return_value.get.side_effect = OSError("connection reset")

        with self.assertRaises(OSError):
            mms.get_data("b_gse_fgm_srvy_l2", TINT, 1, verbose=False, source="aws")

        self.assertEqual(mms.cache_info()["n_files"], 0)
        self.assertEqual(list_files.call_count, 1)


if __name__ == "__main__":
    unittest.main()
