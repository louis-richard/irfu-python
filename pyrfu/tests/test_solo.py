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
import pycdfpp
import xarray as xr
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

    @staticmethod
    def _tnr_records(sweep, band, power, sensor=4):
        # Synthetic L2 file: one record per (sweep, band), 1 s apart, the sensor
        # on the first channel, auto1 equal to power at the 32 frequencies of
        # the band
        n_rec = len(sweep)
        time = np.datetime64("2023-01-01T00:00:00", "ns")
        time = time + np.arange(n_rec) * np.timedelta64(1, "s")
        spectra = np.repeat(np.array(power, np.float64)[:, None], 32, axis=1)
        variables = {
            "epoch": pycdfpp.to_tt2000(time),
            "auto1": spectra,
            "auto2": np.zeros_like(spectra),
            "magnetic_spectral_power1": spectra,
            "magnetic_spectral_power2": np.zeros_like(spectra),
            "sweep_num": np.array(sweep, np.uint32),
            "tnr_band": np.array(band, np.uint8),
            "sensor_config": np.tile(np.array([sensor, 5], np.uint8), (n_rec, 1)),
            "front_end": np.ones(n_rec, dtype=np.uint8),
            "tnr_band_freq": np.geomspace(4e3, 1e6, 128)
            .reshape(4, 32)
            .astype(np.uint32),
        }
        return {k: xr.DataArray(v) for k, v in variables.items()}, time

    @classmethod
    def _tnr_l2(cls, sensor, n_sweeps):
        # Sweeps 1, ..., n_sweeps of the four bands, spectra equal to the sweep
        # number at every frequency
        sweep = np.repeat(np.arange(1, n_sweeps + 1), 4)
        band = np.tile(np.arange(4), n_sweeps)
        return cls._tnr_records(sweep, band, sweep, sensor)

    @staticmethod
    def _read_tnr(variables, sensor=4, n_files=1):
        module = sys.modules["pyrfu.solo.read_tnr"]
        files = [f"solo_L2_rpw-tnr-surv_2023010{i + 1}_V01.cdf" for i in range(n_files)]

        with mock.patch.object(module, "_list_files_tnr_l2", return_value=files):
            with mock.patch.object(module, "read_cdf", side_effect=variables):
                tint = ["2023-01-01T00:00:00", "2023-01-02T00:00:00"]
                return solo.read_tnr(tint, sensor)

    @data(4, 7)
    def test_read_tnr_data(self, sensor):
        # Every sweep (the first and the last were dropped), at the time of its
        # first record, below 100 kHz, without subtraction (was minus the
        # integral over frequency divided by the number of sweeps)
        variables, time = self._tnr_l2(sensor, 5)
        out = self._read_tnr([variables], sensor)

        freq = variables["tnr_band_freq"].data.ravel() / 1e3
        n_freq = np.sum(freq < 1e2)
        np.testing.assert_array_equal(out.time.data, time[::4])
        np.testing.assert_allclose(out.frequency.data, freq[:n_freq])
        np.testing.assert_array_equal(
            out.data, np.tile(np.arange(1.0, 6.0)[:, None], n_freq)
        )

    def test_read_tnr_missing_band_and_empty_sweep(self):
        # Sweep 2 misses band 1 (NaN there), sweep 3 has no power (skipped):
        # sweep 4 keeps its own time (time and spectra were out of step)
        sweep = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        band = [0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        power = [1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 4, 4, 4, 4]
        variables, time = self._tnr_records(sweep, band, power)
        out = self._read_tnr([variables])

        np.testing.assert_array_equal(out.time.data, time[[0, 4, 11]])
        np.testing.assert_array_equal(out.data[:, 0], [1.0, 2.0, 4.0])
        self.assertTrue(np.isnan(out.data[1, 32:64]).all())
        self.assertFalse(np.isnan(out.data[[0, 2]]).any())

    def test_read_tnr_interleaved_sweeps(self):
        # A record of sweep 2 after the first one of sweep 3: SWEEP_NUM
        # decreases by 1, which wrapped in uint32 and counted as a reset,
        # splitting sweep 2 in two spectra
        sweep = [1, 1, 1, 1, 2, 2, 2, 3, 2, 3, 3, 3]
        band = [0, 1, 2, 3, 0, 1, 2, 0, 3, 1, 2, 3]
        variables, time = self._tnr_records(sweep, band, sweep)
        out = self._read_tnr([variables])

        np.testing.assert_array_equal(out.time.data, time[[0, 4, 7]])
        np.testing.assert_array_equal(out.data[:, 0], [1.0, 2.0, 3.0])
        self.assertFalse(np.isnan(out.data).any())

    def test_read_tnr_sweep_reset(self):
        # SWEEP_NUM jumps from 202 to 5 (a reset): the sweeps after it are
        # renumbered, including the last record (the slice missed it)
        sweep = np.repeat([200, 201, 202, 5, 6], 4)
        band = np.tile(np.arange(4), 5)
        variables, time = self._tnr_records(sweep, band, np.repeat(np.arange(1, 6), 4))
        out = self._read_tnr([variables])

        np.testing.assert_array_equal(out.time.data, time[::4])
        np.testing.assert_array_equal(out.data[:, 0], np.arange(1.0, 6.0))
        self.assertFalse(np.isnan(out.data).any())

    def test_read_tnr_no_sensor(self):
        # A file without the sensor is skipped (raised ValueError); ValueError
        # only if no file has the sensor
        variables_4, _ = self._tnr_l2(4, 3)
        variables_7, _ = self._tnr_l2(7, 3)
        out = self._read_tnr([variables_7, variables_4], sensor=4, n_files=2)
        self.assertEqual(len(out), 3)

        with self.assertRaises(ValueError):
            self._read_tnr([variables_7], sensor=4)


if __name__ == "__main__":
    unittest.main()
