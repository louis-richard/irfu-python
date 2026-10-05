#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import importlib
import itertools
import json
import os
import random
import string
import tempfile
import unittest
import warnings
from contextlib import nullcontext
from unittest import mock

# 3rd party imports
import keyring
import keyring.credentials
import keyring.errors
import numpy as np
import pycdfpp
import requests
import xarray as xr
from botocore import UNSIGNED
from botocore.exceptions import ClientError
from ddt import data, ddt, idata, unpack
from scipy import constants

# Local imports
from .. import mms, pyrf
from ..mms.feeps_flat_field_corrections import g_corr
from ..mms.get_ts import get_ts
from ..mms.psd_moments import _moms
from . import (
    generate_data,
    generate_defatt,
    generate_spectr,
    generate_timeline,
    generate_ts,
    generate_vdf,
)

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Module (pyrfu.mms.list_files_aws is also the name of the function)
list_files_aws_module = importlib.import_module("..mms.list_files_aws", __package__)

TEST_TINT = ["2019-01-01T00:00:00.000000000", "2019-01-01T00:10:00.000000000"]

data_units_keys = {
    "flux": "1/(cm^2 s sr keV)",
    "counts": "counts",
    "cps": "1/s",
    "mask": "sector_mask",
}


def generate_feeps(f_s, n_pts, data_rate, dtype, lev, units_name, mms_id):

    units_key = data_units_keys[units_name.lower()]

    var = {
        "tmmode": data_rate,
        "dtype": dtype,
        "lev": lev,
        "units_name": units_name,
        "species": dtype[0],
        "mmsId": mms_id,
    }

    eyes = mms.feeps_active_eyes(var, TEST_TINT, mms_id)
    keys = [f"{k}-{eyes[k][i]}" for k in eyes for i in range(len(eyes[k]))]
    feeps_dict = {
        k: generate_spectr(f_s, n_pts, 16, dict(sensor=f"energy-{k}", UNITS=units_key))
        for k in keys
    }

    feeps_dict["spinsectnum"] = pyrf.ts_scalar(
        generate_timeline(f_s, n_pts), np.tile(np.arange(12), n_pts // 12 + 1)[:n_pts]
    )

    feeps_alle = xr.Dataset(feeps_dict)
    feeps_alle.attrs = {**var}

    return feeps_alle


def generate_eis(f_s, n_pts, data_rate, dtype, lev, specie, data_unit, mms_id):
    pref = f"mms{mms_id:d}_epd_eis"
    pref = f"{pref}_{data_rate}_{lev}_{dtype}"

    if data_rate == "brst":
        pref = f"{pref}_{data_rate}_{dtype}"
    else:
        pref = f"{pref}_{dtype}"

    suf = f"{specie}_P1_{data_unit.lower()}_t"

    keys = [f"{pref}_{suf}{t:d}" for t in range(6)]

    spin_nums = pyrf.ts_scalar(
        generate_timeline(f_s, n_pts),
        np.sort(np.tile(np.arange(n_pts // 12 + 1), (12,)))[1 : n_pts + 1],
    )
    sectors = pyrf.ts_scalar(
        generate_timeline(f_s, n_pts),
        np.tile(np.arange(12), n_pts // 12 + 1)[1 : n_pts + 1],
    )

    if dtype.lower() == "extof":
        energies = np.array(
            [
                47.645324,
                54.928681,
                62.419454,
                70.833554,
                80.315371,
                91.00098,
                103.018894,
                116.554129,
                131.801143,
                148.970297,
                168.295534,
                190.060874,
                214.590996,
                242.245343,
                273.466432,
                308.768669,
                348.73539,
                394.035378,
                445.404668,
                503.597543,
                569.429005,
                643.764143,
                727.683404,
                822.660211,
                930.654627,
            ]
        )
    else:
        energies = np.array(
            [
                10.51516,
                11.509144,
                12.612351,
                13.817409,
                15.111664,
                16.55435,
                18.134081,
                19.857029,
                21.774935,
                23.807037,
                26.021971,
                28.526016,
                31.215776,
                34.228877,
                37.604494,
                41.116729,
                45.29041,
                51.412368,
                58.570702,
                65.951929,
                75.09237,
            ]
        )

    eis_dict = {"spin": spin_nums, "sector": sectors}

    for i, k in enumerate(keys):
        eis_dict[f"t{i:d}"] = generate_spectr(f_s, n_pts, len(energies), "energy")
        eis_dict[f"look_t{i:d}"] = generate_ts(f_s, n_pts, tensor_order=1)

    # glob_attrs = {**outdict["spin"].attrs["GLOBAL"], **var}
    glob_attrs = {
        "delta_energy_plus": 0.5 * np.ones(len(energies)),
        "delta_energy_minus": 0.5 * np.ones(len(energies)),
        "species": specie,
        "randattrs": "".join(random.choice(string.ascii_lowercase) for _ in range(10)),
    }

    # Build Dataset
    eis = xr.Dataset(eis_dict, attrs=glob_attrs)
    eis = eis.assign_coords(energy=energies)

    return eis


def _mms_keys():
    test_path = os.path.dirname(os.path.abspath(__file__))

    root_path = os.path.join(os.path.split(test_path)[0], "mms")

    with open(
        os.sep.join([root_path, "mms_keys.json"]), "r", encoding="utf-8"
    ) as json_file:
        keys_ = json.load(json_file)

    all_keys = list(
        np.hstack([list(instrument.keys()) for instrument in keys_.values()])
    )
    return all_keys


@ddt
class CalcEpsilonTestCase(unittest.TestCase):
    @data(
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="bazinga"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="bazinga"),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
        ),
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="ions"),
            generate_ts(32.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
        ),
    )
    @unpack
    def test_calc_epsilon_input(self, vdf, model_vdf, n_s, sc_pot):
        with self.assertRaises(ValueError):
            mms.calculate_epsilon(vdf, model_vdf, n_s, sc_pot)

    @data(
        (
            generate_vdf(
                64.0,
                100,
                (32, 16, 16),
                energy01=False,
                species="ions",
                units="s^3/cm^6",
            ),
            generate_vdf(
                64.0,
                100,
                (32, 16, 16),
                energy01=False,
                species="ions",
                units="s^3/cm^6",
            ),
            {},
        ),
        (
            generate_vdf(
                64.0, 100, (32, 16, 16), energy01=False, species="ions", units="s^3/m^6"
            ),
            generate_vdf(
                64.0,
                100,
                (32, 16, 16),
                energy01=False,
                species="ions",
                units="s^3/km^6",
            ),
            {},
        ),
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="electrons"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=False, species="electrons"),
            {},
        ),
        (
            generate_vdf(64.0, 100, (32, 16, 16), energy01=True, species="ions"),
            generate_vdf(64.0, 100, (32, 16, 16), energy01=True, species="ions"),
            {},
        ),
    )
    @unpack
    def test_calc_epsilon_output(self, vdf, model_vdf, kwargs):
        mms.calculate_epsilon(
            vdf,
            model_vdf,
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            **kwargs,
        )

    @staticmethod
    def _maxwellian(widths="attrs", alternating=False):
        # FPI-like skymap of a drifting proton Maxwellian (n = 1 cm^-3)
        energy1 = None
        if alternating:
            energy1 = 10.0 * 3000.0 ** ((np.arange(32) + 0.5) / 31)

        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(
            1.0, 1000.0, [200.0, 0.0, 0.0], n_t=4, energy1=energy1
        )
        vdf.data.attrs["UNITS"] = "s^3/cm^6"

        if widths == "missing":
            del vdf.attrs["delta_energy_minus"], vdf.attrs["delta_energy_plus"]
        elif widths == "None":
            # get_dist sets missing widths to None
            vdf.attrs["delta_energy_minus"] = vdf.attrs["delta_energy_plus"] = None

        n_s = pyrf.ts_scalar(vdf.time.data, np.ones(4))
        return vdf, n_s

    @data(
        ("attrs", False),
        ("missing", False),
        ("None", False),
        ("missing", True),
    )
    @unpack
    def test_calc_epsilon_values(self, widths, alternating):
        # epsilon = int |f - f_model| d3v / (2 n): 0 for identical distributions
        # and 1/2 for a zero model. A single energy table without energy widths
        # (IndexError) and widths set to None (TypeError) used to crash.
        vdf, n_s = self._maxwellian(widths, alternating)
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.zeros(4))

        eps = mms.calculate_epsilon(vdf, vdf.copy(deep=True), n_s, sc_pot)
        np.testing.assert_allclose(eps.data, 0.0, atol=1e-12)

        model = vdf.copy(deep=True)
        model.data.data[...] = 0.0
        eps = mms.calculate_epsilon(vdf, model, n_s, sc_pot)
        np.testing.assert_allclose(eps.data, 0.5, rtol=0.01)

    def test_calc_epsilon_straddling_sc_pot(self):
        # Channels whose lower edge is below the spacecraft potential used to be
        # dropped (sqrt of a negative energy is NaN); compare with a direct sum
        vdf, n_s = self._maxwellian()
        v_sc = -25.0  # ions: the corrected energy is E + V = E - 25 eV
        model = vdf.copy(deep=True)
        model.data.data[...] = 0.0
        eps = mms.calculate_epsilon(
            vdf, model, n_s, pyrf.ts_scalar(vdf.time.data, np.full(4, v_sc))
        )

        energy = vdf.energy.data[0]
        e_minus = vdf.attrs["delta_energy_minus"][0]
        e_plus = vdf.attrs["delta_energy_plus"][0]
        self.assertTrue(np.any((energy - e_minus < 25.0) & (energy + e_plus > 25.0)))

        def speed(e_kin):
            e_corr = np.clip(e_kin - 25.0, 0.0, None)
            return np.sqrt(2 * constants.elementary_charge * e_corr / constants.m_p)

        w_v = speed(energy) ** 2 * (speed(energy + e_plus) - speed(energy - e_minus))
        w_ang = np.sin(np.deg2rad(vdf.theta.data)) * np.deg2rad(11.25) ** 2
        f_si = vdf.data.data[0] * 1e12
        expected = np.sum(f_si * w_v[:, None, None] * w_ang[None, None, :]) / 2e6
        np.testing.assert_allclose(eps.data, expected, rtol=1e-10)


class CorrectEdpProbeTimingTestCase(unittest.TestCase):
    def test_correct_edp_probe_timing_values(self):
        # Probe potentials linear in time, sampled as in the EDP processing:
        # V3 and V5 7.629 and 15.259 us late, V2, V4, V6 reconstructed from
        # E12, E34, E56 sampled 26.703, 30.518 and 34.332 us late
        time = generate_timeline(8192.0, 100)
        t_sec = (time - time[0]) / np.timedelta64(1, "s")
        offset = np.array([1.0, 1.5, -2.0, 0.5, 3.0, 2.5])
        rate = np.array([1e3, -2e3, 3e3, 1e3, -1e3, 4e3])

        def pot(i, delay=0.0):
            return offset[i] + rate[i] * (t_sec + delay)

        v_odd = [pot(0), pot(2, 7.629e-6), pot(4, 15.259e-6)]
        v_l2 = []
        for i, (delay, length) in enumerate(
            zip([26.703e-6, 30.518e-6, 34.332e-6], [0.120, 0.120, 0.0292])
        ):
            e_meas = (pot(2 * i, delay) - pot(2 * i + 1, delay)) / length
            v_l2 += [v_odd[i], v_odd[i] - e_meas * length]

        sc_pot = xr.DataArray(
            np.column_stack(v_l2),
            coords=[time, np.arange(1, 7)],
            dims=["time", "probe"],
            attrs={"UNITS": "V"},
        )
        sc_pot_data = sc_pot.data.copy()

        result = mms.correct_edp_probe_timing(sc_pot)

        expected = np.column_stack([pot(i) for i in range(6)])
        np.testing.assert_allclose(result.data, expected, atol=1e-9)
        self.assertTupleEqual(result.dims, ("time", "probe"))
        np.testing.assert_array_equal(result.probe.data, np.arange(1, 7))
        self.assertEqual(result.attrs["UNITS"], "V")
        np.testing.assert_array_equal(sc_pot.data, sc_pot_data)


class ConfigCacheTestCase(unittest.TestCase):
    # The configuration and the SDC session are cached until the configuration
    # file changes (e.g., mms.db_init)

    @staticmethod
    def _write_config(path, default, rights="public", mtime_ns=None):
        config = {
            "default": default,
            "local": ".",
            "sdc": {"rights": rights, "username": "u"},
            "aws": "",
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(config, file)
        if mtime_ns is not None:
            os.utime(path, ns=(mtime_ns, mtime_ns))

    def test_load_config_refresh(self):
        module = importlib.import_module("pyrfu.mms.db_get_ts")
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.json")
            self._write_config(path, "local", mtime_ns=10**18)
            with mock.patch.object(module, "MMS_CFG_PATH", path):
                module._load_config_cached.cache_clear()
                config = module._load_config()
                config["default"] = "changed by the caller"
                self.assertEqual(module._load_config()["default"], "local")

                self._write_config(path, "aws", mtime_ns=10**18 + 10**9)
                self.assertEqual(module._load_config()["default"], "aws")
                module._load_config_cached.cache_clear()

    def test_login_lasp_refresh(self):
        module = importlib.import_module("pyrfu.mms.list_files_sdc")
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.json")
            self._write_config(path, "sdc", mtime_ns=10**18)
            with (
                mock.patch.object(module, "MMS_CFG_PATH", path),
                mock.patch.object(module.requests, "Session") as session_cls,
                mock.patch.object(module, "_get_credential", return_value=None),
            ):
                module._login_lasp_cached.cache_clear()
                first = module._login_lasp()
                self.assertIs(module._login_lasp()[0], first[0])
                self.assertEqual(session_cls.call_count, 1)

                self._write_config(path, "sdc", mtime_ns=10**18 + 10**9)
                module._login_lasp()
                self.assertEqual(session_cls.call_count, 2)
                module._login_lasp_cached.cache_clear()


class _MemoryKeyring:
    # In-memory keyring backend with a given priority (>= 1: system keyring)
    def __init__(self, priority):
        self.priority = priority
        self.passwords = {}
        self.file_path = "/nowhere/keyring_pass.cfg"

    def get_credential(self, service, username):
        password = self.passwords.get((service, username))
        return (
            keyring.credentials.SimpleCredential(username, password)
            if password
            else None
        )

    def set_password(self, service, username, password):
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        if self.passwords.pop((service, username), None) is None:
            raise keyring.errors.PasswordDeleteError("not found")


class DbInitCredentialsTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_init")
        self.plaintext = _MemoryKeyring(0.5)

    def _db_init(self, system, **kwargs):
        with tempfile.TemporaryDirectory() as root:
            with (
                mock.patch.object(
                    self.module, "MMS_CFG_PATH", os.path.join(root, "c.json")
                ),
                mock.patch.object(
                    self.module.keyring, "get_keyring", return_value=system
                ),
                mock.patch.object(
                    self.module, "PlaintextKeyring", return_value=self.plaintext
                ),
                mock.patch.object(self.module.keyring, "set_keyring") as set_keyring,
            ):
                self.module.db_init(local=root, sdc="sitl", **kwargs)
                credential = self.module._get_credential(kwargs["sdc_username"])

        set_keyring.assert_not_called()  # the process keyring is not changed
        return credential

    def test_db_init_system_keyring(self):
        # Saved in the system keyring, and the old plaintext copy is removed
        system = _MemoryKeyring(5)
        self.plaintext.set_password("mms-sdc", "louis", "old")

        credential = self._db_init(system, sdc_username="louis", sdc_password="new")

        self.assertEqual(credential.password, "new")
        self.assertDictEqual(system.passwords, {("mms-sdc", "louis"): "new"})
        self.assertDictEqual(self.plaintext.passwords, {})

    def test_db_init_no_system_keyring(self):
        # Without system keyring, saved in plain text with a warning
        system = _MemoryKeyring(0)
        with self.assertLogs("pyrfu", level="WARNING"):
            credential = self._db_init(system, sdc_username="louis", sdc_password="pw")

        self.assertEqual(credential.password, "pw")
        self.assertDictEqual(system.passwords, {})
        self.assertDictEqual(self.plaintext.passwords, {("mms-sdc", "louis"): "pw"})

    def test_get_credential_previous_versions(self):
        # Credentials saved in plain text by previous versions are still found
        self.plaintext.set_password("mms-sdc", "louis", "old")
        with (
            mock.patch.object(
                self.module.keyring, "get_keyring", return_value=_MemoryKeyring(5)
            ),
            mock.patch.object(
                self.module, "PlaintextKeyring", return_value=self.plaintext
            ),
        ):
            self.assertEqual(self.module._get_credential("louis").password, "old")


class MmsConfigPathTestCase(unittest.TestCase):
    # The MMS configuration lives in the user configuration directory, created
    # from the package configuration the first time
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_init")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.package = os.path.join(self.tmp_dir.name, "package_config.json")
        with open(self.package, "w", encoding="utf-8") as file:
            json.dump({"default": "aws"}, file)
        self.user_dir = os.path.join(self.tmp_dir.name, "user", "pyrfu")

        for patch in [
            mock.patch.object(self.module, "_PACKAGE_CFG_PATH", self.package),
            mock.patch.object(
                self.module.platformdirs, "user_config_dir", return_value=self.user_dir
            ),
        ]:
            patch.start()
            self.addCleanup(patch.stop)

    def test_config_path_migration(self):
        # Created from the package configuration the first time, then kept
        path = self.module._user_config_path()
        self.assertEqual(path, os.path.join(self.user_dir, "mms_config.json"))
        with open(path, encoding="utf-8") as file:
            self.assertDictEqual(json.load(file), {"default": "aws"})

        with open(path, "w", encoding="utf-8") as file:
            json.dump({"default": "sdc"}, file)
        self.assertEqual(self.module._user_config_path(), path)
        with open(path, encoding="utf-8") as file:
            self.assertDictEqual(json.load(file), {"default": "sdc"})

    def test_config_path_not_writable(self):
        # The package configuration is used if the user directory is not writable
        with mock.patch.object(self.module.shutil, "copyfile", side_effect=OSError):
            self.assertEqual(self.module._user_config_path(), self.package)


class DbGetTsTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_get_ts")

    def _ts(self, i_file, value):
        # 3 samples per file, 1 s apart, starting 10 s after each other
        times = generate_timeline(
            1.0, 3, ref_time=f"2019-01-01T00:00:{10 * i_file:02d}"
        )
        return pyrf.ts_scalar(times, np.full(3, value, dtype=float))

    def test_db_get_ts_dict_one_read_per_file(self):
        # All the variables are read from each file once (one download from the
        # SDC or AWS), and appended in time
        session = mock.Mock()
        sources = (["file0", "file1"], session, {})
        names = ["var_a", "var_b", "var_c"]

        def get_ts(content, cdf_name, tint):
            return self._ts(int(content[-1:]), names.index(cdf_name))

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ):
            with mock.patch.object(
                self.module,
                "_get_file_content_sources",
                side_effect=lambda r, f, *_: f.encode(),
            ) as download:
                with mock.patch.object(self.module, "get_ts", side_effect=get_ts):
                    out = self.module._db_get_ts_dict(
                        "mms1_feeps_brst_l2_electron",
                        names,
                        ["a", "b"],
                        False,
                        "",
                        "SDC",
                    )

        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args_list[0].args[:2], ("sdc", "file0"))
        session.close.assert_not_called()  # shared, cached session

        for value, name in enumerate(names):
            self.assertEqual(len(out[name]), 6)
            np.testing.assert_array_equal(out[name].data, value)

    def test_db_get_ts_no_file(self):
        session = mock.Mock()

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=([], session, {})
        ):
            with self.assertRaises(FileNotFoundError):
                mms.db_get_ts("mms1_fgm_srvy_l2", "b", ["a", "b"], source="sdc")

        session.close.assert_not_called()  # shared, cached session

    def test_resolve_source(self):
        with mock.patch.object(
            self.module, "_load_config", return_value={"default": "local"}
        ):
            self.assertEqual(self.module._resolve_source(""), "local")
            self.assertEqual(self.module._resolve_source(None), "local")
            self.assertEqual(self.module._resolve_source("Default"), "local")

        self.assertEqual(self.module._resolve_source("AWS"), "aws")

        with self.assertRaises(ValueError):
            self.module._resolve_source("cdaweb")

    def test_tokenize_no_dtype(self):
        mms_id, var = self.module._tokenize("mms1_fgm_srvy_l2")
        self.assertEqual(mms_id, "1")
        self.assertDictEqual(
            var, {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
        )


class DbGetVariableTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.db_get_variable")

    def test_db_get_variable_source(self):
        # Read from the first file of the resource, the session is kept
        session = mock.Mock()
        sources = (["file0", "file1"], session, {})

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ) as list_files:
            with mock.patch.object(
                self.module, "_get_file_content_sources", return_value=b"cdf"
            ) as download:
                with mock.patch.object(self.module, "get_variable") as get_variable:
                    mms.db_get_variable(
                        "mms2_feeps_brst_l2_electron",
                        "sensor_ids",
                        ["a", "b"],
                        verbose=False,
                        source="aws",
                    )

        self.assertEqual(list_files.call_args.args[0], "aws")
        self.assertEqual(list_files.call_args.args[3]["dtype"], "electron")
        download.assert_called_once()
        self.assertEqual(download.call_args.args[:2], ("aws", "file0"))
        get_variable.assert_called_once_with(b"cdf", "sensor_ids")
        session.close.assert_not_called()  # shared, cached session

    def test_db_get_variable_no_file(self):
        session = mock.Mock()

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=([], session, {})
        ):
            with self.assertRaises(FileNotFoundError):
                mms.db_get_variable(
                    "mms1_fgm_srvy_l2", "label", ["a", "b"], source="sdc"
                )

        session.close.assert_not_called()  # shared, cached session

    def test_get_variable_ndim(self):
        # N-D variables (only 1-D variables were supported): dims x, y, ...
        module = importlib.import_module("pyrfu.mms.get_variable")
        variable = mock.Mock(values=np.arange(12).reshape(1, 12), attributes={})

        with mock.patch.object(module, "load", return_value={"ids": variable}):
            out = module.get_variable(b"cdf", "ids")

        self.assertTupleEqual(out.dims, ("x", "y"))
        np.testing.assert_array_equal(out.data, variable.values)


class DbInitTestCase(unittest.TestCase):
    # db_init writes to a temporary configuration file and in-memory keyrings,
    # never to the user's configuration or keyring
    def setUp(self):
        module = importlib.import_module("pyrfu.mms.db_init")
        self.tmp_dir = tempfile.TemporaryDirectory()
        patches = [
            mock.patch.object(
                module, "MMS_CFG_PATH", os.path.join(self.tmp_dir.name, "c.json")
            ),
            mock.patch.object(
                module.keyring, "get_keyring", return_value=_MemoryKeyring(5)
            ),
            mock.patch.object(
                module, "PlaintextKeyring", return_value=_MemoryKeyring(0.5)
            ),
            mock.patch.object(module.keyring, "set_keyring"),
            mock.patch.object(module.keyring, "get_credential", return_value=None),
            mock.patch.object(module.keyring, "set_password"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp_dir.cleanup)

    def test_db_init_input(self):
        with self.assertRaises(NotImplementedError):
            mms.db_init(default="bazinga!", local=os.getcwd(), sdc="public")

        with self.assertRaises(FileNotFoundError):
            mms.db_init(default="local", local="bazinga!", sdc="public")

        with self.assertRaises(ValueError):
            mms.db_init(default="sdc", local=os.getcwd(), sdc="bazinga!")

    def test_db_init_output(self):
        self.assertIsNone(mms.db_init(local=os.getcwd()))


@ddt
class Def2PsdTestCase(unittest.TestCase):
    @data(np.random.random((100, 32, 32, 16)))
    def test_def2psd_input(self, value):
        with self.assertRaises(TypeError):
            mms.def2psd(value)

    @data(generate_vdf(64.0, 100, (32, 32, 16), False, "I AM GROOT!!", "s^3/cm^6"))
    def test_def2psd_input_mass_ratio(self, value):
        with self.assertRaises(ValueError):
            mms.def2psd(value)

    @data(generate_vdf(64.0, 100, (32, 32, 16), False, "ions", "bazinga"))
    def test_def2psd_input_convert(self, value):
        with self.assertRaises(ValueError):
            mms.def2psd(value)

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["keV/(cm^2 s sr keV)", "eV/(cm^2 s sr eV)", "1/(cm^2 s sr)"],
        )
    )
    @unpack
    def test_def2psd_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.def2psd(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.def2psd(spectr)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Dpf2PsdTestCase(unittest.TestCase):
    @data(
        ("I AM GROOT!!", "s^3/cm^6"),
        ("ions", "bazinga"),
    )
    @unpack
    def test_dpf2psd_input(self, species, units):
        with self.assertRaises(ValueError):
            mms.dpf2psd(generate_vdf(64.0, 100, (32, 32, 16), False, species, units))

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["1/(cm^2 s sr keV)", "1/(cm^2 s sr eV)"],
        )
    )
    @unpack
    def test_dpf2psd_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.dpf2psd(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.dpf2psd(spectr)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Dsl2GseTestCase(unittest.TestCase):
    def test_dsl2gse_input(self):
        with self.assertRaises(TypeError):
            mms.dsl2gse(
                generate_ts(64.0, 42, tensor_order=1), np.random.random((42, 3)), 1
            )

    @data(
        xr.Dataset({"z_dec": generate_ts(64.0, 42), "z_ra": generate_ts(64.0, 42)}),
        np.random.random(3),
    )
    def test_dsl2gse_output(self, value):
        result = mms.dsl2gse(generate_ts(64.0, 42, tensor_order=1), value, 1)
        self.assertIsInstance(result, xr.DataArray)
        result = mms.dsl2gse(generate_ts(64.0, 42, tensor_order=1), value, -1)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Dsl2GsmTestCase(unittest.TestCase):
    def test_dsl2gsm_input(self):
        with self.assertRaises(TypeError):
            mms.dsl2gsm(
                generate_ts(64.0, 42, tensor_order=1), np.random.random((42, 3)), 1
            )

    @data(
        xr.Dataset({"z_dec": generate_ts(64.0, 42), "z_ra": generate_ts(64.0, 42)}),
        np.random.random(3),
    )
    def test_dsl2gsm_output(self, value):
        result = mms.dsl2gsm(generate_ts(64.0, 42, tensor_order=1), value, 1)
        self.assertIsInstance(result, xr.DataArray)
        result = mms.dsl2gsm(generate_ts(64.0, 42, tensor_order=1), value, -1)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class DslTransformationValuesTestCase(unittest.TestCase):
    @data((mms.dsl2gse, "GSE"), (mms.dsl2gsm, "GSM"))
    @unpack
    def test_dsl_transformation_values(self, func, frame):
        b_xyz = pyrf.ts_vec_xyz(
            generate_timeline(1.0, 5), np.tile([1.0, 2.0, 3.0], (5, 1))
        )

        # Spin axis along z: DSL and the target frame coincide
        np.testing.assert_allclose(
            func(b_xyz, np.array([0.0, 0.0, 1.0])).data, b_xyz.data, atol=1e-12
        )

        spin_axis = np.array([0.1, -0.2, 0.97])
        forward = func(b_xyz, spin_axis / np.linalg.norm(spin_axis))
        backward = func(forward, spin_axis / np.linalg.norm(spin_axis), -1)
        np.testing.assert_allclose(backward.data, b_xyz.data)

        # The GSE/GSM -> DSL output used to be labelled GSE/GSM
        self.assertEqual(forward.attrs["COORDINATE_SYSTEM"], frame)
        self.assertEqual(backward.attrs["COORDINATE_SYSTEM"], "DSL")

        # A spin axis that isn't a unit vector gives the same rotation (it used
        # to scale and distort the field)
        np.testing.assert_allclose(func(b_xyz, 3.0 * spin_axis).data, forward.data)
        np.testing.assert_allclose(
            np.linalg.norm(forward.data, axis=1), np.linalg.norm(b_xyz.data, axis=1)
        )


@ddt
class EisCombineProtonPadTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_combine_proton_pad_output(self, tmmode, specie, unit):
        mms_id = random.randint(1, 5)
        phxtof_allt = generate_eis(
            64.0, 100, tmmode, "phxtof", "l2", specie, unit, mms_id
        )
        extof_allt = generate_eis(
            64.0, 100, tmmode, "extof", "l2", specie, unit, mms_id
        )
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.DataArray)

    @idata(itertools.product([99, 100], repeat=2))
    @unpack
    def test_eis_combine_proton_pad_input(self, n_phxtof, n_extof):
        phxtof_allt = generate_eis(
            64.0, n_phxtof, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(
            64.0, n_extof, "brst", "extof", "l2", "proton", "flux", 1
        )
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.DataArray)

    @data(None, [1, 0, 0], generate_ts(64.0, 10, tensor_order=1))
    def test_eis_combine_proton_pad_vec(self, vec):
        phxtof_allt = generate_eis(
            64.0, 100, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt, vec)
        self.assertIsInstance(result, xr.DataArray)

    def test_eis_combine_proton_pad_options(self):
        phxtof_allt = generate_eis(
            64.0, 100, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        result = mms.eis_combine_proton_pad(phxtof_allt, extof_allt, None, despin=True)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisCombineProtonSpecTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_combine_proton_spec_output(self, tmmode, specie, unit):
        mms_id = random.randint(1, 5)
        phxtof_allt = generate_eis(
            64.0, 100, tmmode, "phxtof", "l2", specie, unit, mms_id
        )
        extof_allt = generate_eis(
            64.0, 100, tmmode, "extof", "l2", specie, unit, mms_id
        )
        result = mms.eis_combine_proton_spec(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.Dataset)

    @idata(itertools.product([99, 100], repeat=2))
    @unpack
    def test_eis_combine_proton_spec_ctimes(self, n_phxtof, n_extof):
        phxtof_allt = generate_eis(
            64.0, n_phxtof, "brst", "phxtof", "l2", "proton", "flux", 1
        )
        extof_allt = generate_eis(
            64.0, n_extof, "brst", "extof", "l2", "proton", "flux", 1
        )
        result = mms.eis_combine_proton_spec(phxtof_allt, extof_allt)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class EisMomentsTestCase(unittest.TestCase):
    @staticmethod
    def _maxwellian_flux(mass, n_cc=1.0, kt_kev=10.0, n_t=3):
        # Omni-directional differential flux (1/(cm^2 s sr keV)) of an isotropic
        # Maxwellian, on a wide energy grid (keV) so the partial moments are full
        energy = np.geomspace(0.01, 2000.0, 400)
        k_t = kt_kev * 1e3 * constants.e
        e_j = energy * 1e3 * constants.e
        vdf = n_cc * 1e6 * (mass / (2 * np.pi * k_t)) ** 1.5 * np.exp(-e_j / k_t)
        flux = 2 * e_j / mass**2 * vdf * 1e3 * constants.e / 1e4
        times = generate_timeline(1.0, n_t)
        flux = np.tile(flux, (n_t, 1))
        return xr.DataArray(flux, coords=[times, energy], dims=["time", "energy"])

    @data(("proton", 1), ("alpha", 4), ("oxygen", 16))
    @unpack
    def test_eis_moments_maxwellian(self, specie, mass_number):
        # n = 1 cm^-3, kT = 10 keV: P = n kT = 1.602 nPa (the 2nd order integral
        # is the energy density, 1.5 times larger), <|v|> = sqrt(8 kT / (pi m))
        mass = mass_number * constants.proton_mass
        n_i, v_i, p_i, t_i = mms.eis_moments(self._maxwellian_flux(mass), specie)
        k_t = 1e4 * constants.e

        np.testing.assert_allclose(n_i.data, 1.0, rtol=1e-4)
        np.testing.assert_allclose(p_i.data, 1e15 * k_t, rtol=1e-4)
        np.testing.assert_allclose(t_i.data, 1e4, rtol=1e-4)
        np.testing.assert_allclose(
            v_i.data, np.sqrt(8 * k_t / (np.pi * mass)) / 1e3, rtol=1e-4
        )

    def test_eis_moments_nan_channel(self):
        flux = self._maxwellian_flux(constants.proton_mass)
        flux.data[1, 10] = np.nan
        n_i, _, p_i, _ = mms.eis_moments(flux)

        self.assertTrue(np.isnan(n_i.data[1]) and np.isnan(p_i.data[1]))
        np.testing.assert_allclose(n_i.data[[0, 2]], 1.0, rtol=1e-4)

    def test_eis_moments_input(self):
        with self.assertRaises(ValueError):
            mms.eis_moments(self._maxwellian_flux(constants.proton_mass), "helium")


@ddt
class EisOmniTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_omni_output(self, tmmode, dtype, specie, unit):
        eis = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        result = mms.eis_omni(eis, "mean")
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisPadTestCase(unittest.TestCase):
    @data(None, [1, 0, 0], generate_ts(64.0, 10, tensor_order=1))
    def test_eis_pad_output(self, vec):
        eis = generate_eis(64.0, 100, "brst", "extof", "l2", "proton", "flux", 1)
        result = mms.eis_pad(eis, vec)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisPadSpinAvgTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_pad_spin_avg_output(self, tmmode, dtype, specie, unit):
        eis = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        eis_pad = mms.eis_pad(eis)
        result = mms.eis_pad_spinavg(eis_pad, eis.spin)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class EisProtonCorrectionTestCase(unittest.TestCase):
    def test_eis_proton_correction_dataarray(self):
        flux_eis = generate_spectr(64, 100, 16, "energy")
        result = mms.eis_proton_correction(flux_eis)
        self.assertIsInstance(result, xr.DataArray)

    @idata(
        itertools.product(
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_proton_correction_dataset(self, tmmode, dtype, specie, unit):
        flux_eis = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        result = mms.eis_proton_correction(flux_eis)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class EisSpinAvgTestCase(unittest.TestCase):
    @idata(
        itertools.product(
            ["mean", "sum"],
            ["srvy", "brst"],
            ["extof", "phxtof"],
            ["proton", "alpha", "oxygen"],
            ["flux", "cps", "counts"],
        )
    )
    @unpack
    def test_eis_spin_avg_output(self, method, tmmode, dtype, specie, unit):
        eis_allt = generate_eis(
            64.0, 100, tmmode, dtype, "l2", specie, unit, random.randint(1, 4)
        )
        result = mms.eis_spin_avg(eis_allt, method)
        self.assertIsInstance(result, xr.Dataset)


def _lh_wave(v_ph, theta, f_s=1024.0, f_w=40.0):
    # Potential phi(x - v t) along d (perpendicular to B = 40 nT z): at the
    # spacecraft E.d = (dphi / dt) / v and dB_par = phi n e mu0 / B, so that
    # phi_E = v int(E.d) dt = phi_B = phi. An unrelated E along z x d picks
    # out the direction.
    n_t = int(4 * f_s)
    t_sec = np.arange(n_t) / f_s
    time = np.datetime64("2019-01-01T00:00:00", "ns")
    time = time + (t_sec * 1e9).astype("timedelta64[ns]")
    b_0, n_0, window = 40.0, 10.0, np.hanning(n_t)

    phi = 50.0 * np.sin(2 * np.pi * f_w * t_sec) * window  # V
    e_d = np.gradient(phi, t_sec) / v_ph  # mV/m
    e_p = np.max(np.abs(e_d)) * np.sin(2 * np.pi * 25.0 * t_sec) * window
    d_vec = np.array([np.cos(theta), np.sin(theta), 0.0])
    d_perp = np.array([-np.sin(theta), np.cos(theta), 0.0])
    d_b = phi * n_0 * 1e6 * constants.e * constants.mu_0 / (b_0 * 1e-18)  # nT

    e_xyz = pyrf.ts_vec_xyz(time, np.outer(e_d, d_vec) + np.outer(e_p, d_perp))
    b_scm = pyrf.ts_vec_xyz(time, np.outer(d_b, [0.0, 0.0, 1.0]))
    b_xyz = pyrf.ts_vec_xyz(time, np.tile([0.0, 0.0, b_0], (n_t, 1)))
    n_e = pyrf.ts_scalar(time, np.full(n_t, n_0))
    tints = ["2019-01-01T00:00:01.300", "2019-01-01T00:00:02.700"]

    return tints, e_xyz, b_scm, b_xyz, n_e, d_vec


@ddt
class LhWaveAnalysisTestCase(unittest.TestCase):
    @data(300.0, 1200.0)
    def test_lh_wave_analysis_values(self, v_ph):
        tints, e_xyz, b_scm, b_xyz, n_e, d_vec = _lh_wave(v_ph, np.deg2rad(30.0))
        phi_eb, v_best, dir_best, thetas, corrs = mms.lh_wave_analysis(
            tints, e_xyz, b_scm, b_xyz, n_e, min_freq=10.0
        )

        self.assertAlmostEqual(np.rad2deg(thetas[np.argmax(corrs)]), 30.0)
        np.testing.assert_allclose(dir_best, d_vec, atol=1e-6)
        self.assertAlmostEqual(v_best, v_ph, delta=1.0)
        self.assertGreater(np.max(corrs), 0.99)
        self.assertListEqual(list(phi_eb.comp.data), ["Ebest", "Bs"])

    def test_lh_wave_analysis_vmax(self):
        tints, e_xyz, b_scm, b_xyz, n_e, _ = _lh_wave(1200.0, np.deg2rad(30.0))
        with self.assertLogs("pyrfu", level="WARNING"):
            _, v_best, _, _, _ = mms.lh_wave_analysis(
                tints, e_xyz, b_scm, b_xyz, n_e, min_freq=10.0, vmax=500.0
            )
        self.assertEqual(v_best, 500.0)


def _brst_segments_response(rows, status_code=200):
    header = (
        "DATASEGMENTID,TAISTARTTIME (TAI seconds since 1958-01-01),"
        "TAIENDTIME (TAI seconds since 1958-01-01),PARAMETERSETID,FOM,ISPENDING,"
        "INPLAYLIST,STATUS,NUMEVALCYCLES,SOURCEID,CREATETIME,FINISHTIME,DISCUSSION"
    )
    lines = [
        f'{i},{start},{end},p,1.0,0,0,{status},1,eva,t,t,"Bow shock, maybe"'
        for i, (start, end, status) in enumerate(rows)
    ]
    response = mock.MagicMock(status_code=status_code)
    response.text = "\n".join([header, *lines]) + "\n"
    if status_code != 200:
        response.raise_for_status.side_effect = requests.HTTPError("404")
    return response


class LoadBrstSegmentsTestCase(unittest.TestCase):
    # TAI 1829005990 s is 2015-12-17T01:12:34 UTC (36 leap seconds)
    tint = ["2015-12-17T01:13:00", "2015-12-17T01:20:00"]

    def test_load_brst_segments_values(self):
        rows = [
            (1829006400, 1829006500, "COMPLETE+FINISHED"),  # unsorted, kept
            (1829005990, 1829006120, "COMPLETE+FINISHED"),  # straddles tint[0]
            (1829006200, 1829006300, "DERELICT+FINISHED"),  # incomplete
            (1829005800, 1829005900, "COMPLETE+FINISHED"),  # ends before tint
            (1829006700, 1829006800, "COMPLETE+FINISHED"),  # starts after tint
        ]
        with mock.patch.object(
            requests, "get", return_value=_brst_segments_response(rows)
        ) as get:
            result = mms.load_brst_segments(self.tint)

        # End times get 10 s more
        expected = [
            ["2015-12-17T01:12:34.000000000", "2015-12-17T01:14:54.000000000"],
            ["2015-12-17T01:19:24.000000000", "2015-12-17T01:21:14.000000000"],
        ]
        self.assertListEqual(result, expected)

        # Server-side selection of the segments overlapping tint
        url = get.call_args.args[0]
        self.assertTrue(url.startswith("https://lasp.colorado.edu/mms/sdc/"))
        self.assertIn("TAISTARTTIME<=1829006436", url)
        self.assertIn("TAIENDTIME>=1829006006", url)

    def test_load_brst_segments_empty(self):
        with mock.patch.object(
            requests, "get", return_value=_brst_segments_response([])
        ):
            self.assertListEqual(mms.load_brst_segments(self.tint), [])

    def test_load_brst_segments_http_error(self):
        response = _brst_segments_response([], status_code=404)
        with mock.patch.object(requests, "get", return_value=response):
            with self.assertRaises(requests.HTTPError):
                mms.load_brst_segments(self.tint)

    def test_load_brst_segments_deprecated_args(self):
        with mock.patch.object(
            requests, "get", return_value=_brst_segments_response([])
        ):
            with self.assertWarns(FutureWarning):
                mms.load_brst_segments(self.tint, data_path="/tmp", download=True)


@ddt
class MakeModelVDFTestCase(unittest.TestCase):
    @data(
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), False),
        (generate_vdf(64.0, 100, (32, 16, 16), species="electrons"), False),
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), True),
    )
    @unpack
    def test_make_Model_vdf_output(self, vdf, isotropic):
        result = mms.make_model_vdf(
            vdf,
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=2),
            isotropic,
        )
        self.assertIsInstance(result, xr.Dataset)


class HpcaEnergiesTestCase(unittest.TestCase):
    def test_hpca_energies_output(self):
        result = mms.hpca_energies()
        self.assertIsInstance(result, list)


@ddt
class HpcaPadTestCase(unittest.TestCase):
    N_AZ, N_PO, N_EN, D_T = 16, 16, 4, 0.625

    def _inputs(self, aze_deg, n_spins=2, centred=True, b_dir=None):
        # vdf = 1 at 16 samples per half-spin (start azimuths 0..15), all anodes
        # at 90 deg (equatorial), and aze records at the start or centre of the
        # half-spins (pyrfu.mms.get_data centres the times: 7.5 samples later)
        n_ti = n_spins * self.N_AZ
        t_0 = np.datetime64("2019-09-14T07:54:00", "ns")
        times = t_0 + (np.arange(n_ti) * self.D_T * 1e9).astype("timedelta64[ns]")
        energy = np.arange(1.0, self.N_EN + 1)
        vdf = xr.DataArray(
            np.ones((n_ti, self.N_PO, self.N_EN)),
            coords=[times, np.arange(self.N_PO), energy],
            dims=["time", "rcomp", "ccomp"],
        )
        saz = xr.DataArray(np.arange(n_ti) % self.N_AZ, coords=[times], dims=["time"])
        offset = 7.5 * self.D_T * 1e9 if centred else 0.0
        t_aze = times[:: self.N_AZ] + np.timedelta64(int(offset), "ns")
        aze = xr.DataArray(
            np.broadcast_to(
                aze_deg[..., None, None], (n_spins, self.N_AZ, self.N_PO, self.N_EN)
            ).copy(),
            coords=[t_aze, np.arange(self.N_AZ), np.arange(self.N_PO), energy],
            dims=["time", "az", "po", "en"],
        )

        # B at 128 Hz, along b_dir(t) (default +x)
        t_b = generate_timeline(
            128.0, int(n_ti * self.D_T * 128) + 256, ref_time="2019-09-14T07:53:59"
        )
        t_s = (t_b - t_0) / np.timedelta64(1, "s")
        b_data = (
            b_dir(t_s) if b_dir is not None else np.tile([1.0, 0.0, 0.0], (len(t_b), 1))
        )
        b_xyz = pyrf.ts_vec_xyz(t_b, b_data)

        return vdf, saz, aze, b_xyz

    def _pad(self, *args, **kwargs):
        return mms.hpca_pad(*args, elevation=np.full(self.N_PO, 90.0), **kwargs)

    @data(True, False)
    def test_hpca_pad_look_direction_and_pairing(self, centred):
        # Looking along -x (azimuth 180) with B along +x: particles along B,
        # pitch angle 0; looking along +x (azimuth 0): 180. The look direction
        # alternates with the azimuth step in half-spin 0 and is 0 in half-spin 1,
        # so each sample must get the azimuths of its own half-spin and step.
        aze_deg = np.zeros((2, self.N_AZ))
        aze_deg[0, ::2] = 180.0
        pad = self._pad(*self._inputs(aze_deg, centred=centred))

        is_par = np.zeros(32, dtype=bool)
        is_par[0:16:2] = True
        np.testing.assert_array_equal(pad.data[is_par, 0], 1.0)
        self.assertTrue(np.all(np.isnan(pad.data[is_par, 1:])))
        np.testing.assert_array_equal(pad.data[~is_par, -1], 1.0)
        self.assertTrue(np.all(np.isnan(pad.data[~is_par, :-1])))

    @data((0.0, 1.0, 0), (180.0, 1.0, -1), (0.0, -1.0, -1))
    @unpack
    def test_hpca_pad_polar_angle(self, polar, b_z, pa_bin):
        # The elevations are the polar angles of the particle velocities (not
        # flipped as the azimuths): polar 0 with B along +z is pitch angle 0
        def b_dir(t_s):
            return np.stack([0 * t_s, 0 * t_s, b_z + 0 * t_s], axis=1)

        inputs = self._inputs(np.full((2, self.N_AZ), 37.0), b_dir=b_dir)
        pad = mms.hpca_pad(*inputs, elevation=np.full(self.N_PO, polar))

        np.testing.assert_array_equal(pad.data[:, pa_bin], 1.0)

    def test_hpca_pad_b_time(self):
        # B along +x in half-spin 0 and along -x in half-spin 1, looking along -x:
        # pitch angle 0 then 180 (B was paired with the wrong samples)
        def b_dir(t_s):
            sign = np.where(t_s < 16 * self.D_T, 1.0, -1.0)
            return np.stack([sign, np.zeros_like(t_s), np.zeros_like(t_s)], axis=1)

        aze_deg = np.full((2, self.N_AZ), 180.0)
        pad = self._pad(*self._inputs(aze_deg, b_dir=b_dir))

        np.testing.assert_array_equal(pad.data[1:15, 0], 1.0)
        np.testing.assert_array_equal(pad.data[17:31, -1], 1.0)
        self.assertTrue(np.all(np.isnan(pad.data[1:15, -1])))

    def test_hpca_pad_energy_range(self):
        # Both ends of elim are included (the top channel was dropped)
        vdf, saz, aze, b_xyz = self._inputs(np.full((2, self.N_AZ), 180.0))
        vdf.data[...] = vdf.ccomp.data  # value = energy
        pad = self._pad(vdf, saz, aze, b_xyz, elim=[2.0, 3.0])

        np.testing.assert_allclose(pad.data[:, 0], 2.5)

    def test_hpca_pad_mismatch(self):
        vdf, saz, aze, b_xyz = self._inputs(np.full((2, self.N_AZ), 180.0))

        # aze shifted by 3 samples: not the start nor the centre of a half-spin
        aze_shifted = aze.assign_coords(
            time=aze.time.data + np.timedelta64(int(3 * self.D_T * 1e9), "ns")
        )
        with self.assertRaises(ValueError):
            self._pad(vdf, saz, aze_shifted, b_xyz)

        # start azimuths of a half-spin not 0..15
        saz_bad = saz.copy()
        saz_bad.data[20] = 7
        with self.assertRaises(ValueError):
            self._pad(vdf, saz_bad, aze, b_xyz)

    def test_hpca_pad_elevation_from_file(self):
        # mms?_hpca_centroid_elevation_angle read from the dataset of vdf
        module = importlib.import_module("pyrfu.mms.hpca_pad")
        vdf, saz, aze, b_xyz = self._inputs(np.full((2, self.N_AZ), 180.0))
        vdf.attrs["GLOBAL"] = {"Logical_source": "mms1_hpca_brst_l2_ion"}
        elevation = xr.DataArray(np.full((1, self.N_PO), 90.0), dims=["x", "y"])

        with mock.patch.object(
            module, "db_get_variable", return_value=elevation
        ) as read:
            pad = mms.hpca_pad(vdf, saz, aze, b_xyz, source="aws")

        self.assertEqual(
            read.call_args.args[:2],
            ("mms1_hpca_brst_l2_ion", "mms1_hpca_centroid_elevation_angle"),
        )
        self.assertEqual(read.call_args.kwargs["source"], "aws")
        np.testing.assert_array_equal(pad.data[:, 0], 1.0)

        del vdf.attrs["GLOBAL"]
        with self.assertRaises(ValueError):
            mms.hpca_pad(vdf, saz, aze, b_xyz)


@ddt
class MakeModelKappaTestCase(unittest.TestCase):
    @data(
        (generate_vdf(64.0, 100, (32, 16, 16), species="bazinga"), random.random()),
    )
    @unpack
    def test_make_model_kappa_input(self, vdf, kappa):
        with self.assertRaises(ValueError):
            mms.make_model_kappa(
                vdf,
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=0),
                kappa,
            )

    @data(
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), random.random()),
        (generate_vdf(64.0, 100, (32, 16, 16), species="electrons"), random.random()),
        (generate_vdf(64.0, 100, (32, 16, 16), species="ions"), random.random()),
    )
    @unpack
    def test_make_model_kappa_output(self, vdf, kappa):
        result = mms.make_model_kappa(
            vdf,
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=1),
            generate_ts(64.0, 100, tensor_order=0),
            kappa,
        )

        self.assertIsInstance(result, xr.Dataset)


class ProbeAlignTimesTestCase(unittest.TestCase):
    def test_probe_align_times_values(self):
        # 20 s spin (18 deg/s), B along x in the spin plane except for
        # 30 s <= t < 40 s where B is along z. Probe 1 (phase + 30 deg) is
        # within 25 deg of +-B for phase mod 180 in [125, 175] deg, i.e.
        # t mod 10 s in [6.94, 9.72] s, probe 3 (phase + 120 deg) for phase
        # mod 180 in [35, 85] deg, i.e. t mod 10 s in [1.94, 4.72] s.
        t_0 = np.datetime64("2019-01-01T00:00:00", "ns")

        def timeline(f_s, duration=60.0):
            t_sec = np.arange(0.0, duration, 1.0 / f_s)
            return t_sec, t_0 + (t_sec * 1e9).astype("timedelta64[ns]")

        t_pot, time_pot = timeline(32.0)
        sc_pot = xr.DataArray(
            np.tile([5.0, 4.0, 5.5, 4.5, 6.0, 6.5], (len(t_pot), 1)),
            coords=[time_pot, np.arange(1, 7)],
            dims=["time", "probe"],
        )
        t_b, time_b = timeline(16.0, 80.0)
        time_b = time_b - np.timedelta64(10, "s")
        b_data = np.tile([30.0, 0.0, 2.0], (len(t_b), 1))
        b_data[(t_b - 10 >= 30) & (t_b - 10 < 40)] = [2.0, 0.0, 30.0]
        b_xyz = pyrf.ts_vec_xyz(time_b, b_data)
        t_ph, time_ph = timeline(4.0, 80.0)
        z_phase = pyrf.ts_scalar(
            time_ph - np.timedelta64(10, "s"), np.mod(18.0 * (t_ph - 10), 360.0)
        )
        inputs = [inp.data.copy() for inp in [sc_pot, b_xyz, z_phase]]

        start1, end1, start3, end3 = mms.probe_align_times(None, b_xyz, sc_pot, z_phase)

        def seconds(times):
            return (times - t_0) / np.timedelta64(1, "s")

        spins = np.array([0, 1, 2, 4, 5]) * 10.0
        dt_pot = 1.0 / 32.0
        for start, end, (t_l, t_r) in [
            (start1, end1, (125 / 18, 175 / 18)),
            (start3, end3, (35 / 18, 85 / 18)),
        ]:
            np.testing.assert_allclose(seconds(start), spins + t_l, atol=dt_pot)
            np.testing.assert_allclose(seconds(end), spins + t_r, atol=dt_pot)

        for inp, data_ in zip([sc_pot, b_xyz, z_phase], inputs):
            np.testing.assert_array_equal(inp.data, data_)


@ddt
class Psd2DefTestCase(unittest.TestCase):
    @data(
        ("I AM GROOT!!", "s^3/cm^6"),
        ("ions", "bazinga"),
    )
    @unpack
    def test_psd2def_input(self, species, units):
        with self.assertRaises(ValueError):
            mms.psd2def(generate_vdf(64.0, 100, (32, 32, 16), False, species, units))

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["s^3/cm^6", "s^3/m^6", "s^3/km^6"],
        )
    )
    @unpack
    def test_psd2def_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.psd2def(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.psd2def(spectr)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class Psd2DpfTestCase(unittest.TestCase):
    @data(
        ("I AM GROOT!!", "s^3/cm^6"),
        ("ions", "bazinga"),
    )
    @unpack
    def test_psd2dpf_input(self, species, units):
        with self.assertRaises(ValueError):
            mms.psd2dpf(generate_vdf(64.0, 100, (32, 32, 16), False, species, units))

    @idata(
        itertools.product(
            [
                "ions",
                "ion",
                "protons",
                "proton",
                "alphas",
                "alpha",
                "helium",
                "electrons",
                "e",
            ],
            ["s^3/cm^6", "s^3/m^6", "s^3/km^6"],
        )
    )
    @unpack
    def test_psd2dpf_output(self, species, units):
        vdf = generate_vdf(64.0, 100, (32, 32, 16), False, species, units)
        result = mms.psd2dpf(vdf)
        self.assertIsInstance(result, xr.Dataset)

        spectr = generate_spectr(64.0, 100, 32, {"species": species, "UNITS": units})
        result = mms.psd2dpf(spectr)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class PsdMomentsTestCase(unittest.TestCase):
    @data(
        (generate_vdf(64.0, 100, (32, 32, 16)), "brst"),
        (generate_vdf(64.0, 100, (32, 32, 16), species="electrons"), "brst"),
        (generate_vdf(64.0, 100, (32, 32, 16), energy01=False), "brst"),
        (generate_vdf(64.0, 100, (32, 32, 16), energy01=False), "fast"),
        (generate_vdf(64.0, 100, (32, 32, 16), energy01=True), "brst"),
    )
    @unpack
    def test_psd_moments_input(self, vdf, data_rate):
        delta_theta = 0.5 * np.ones(vdf.data.shape[3])
        vdf.attrs["delta_theta_minus"] = delta_theta
        vdf.attrs["delta_theta_plus"] = delta_theta

        delta_phi = 0.5 * np.ones((vdf.data.shape[0], vdf.data.shape[2]))
        vdf.attrs["delta_phi_minus"] = delta_phi
        vdf.attrs["delta_phi_plus"] = delta_phi
        vdf.data.attrs["FIELDNAM"] = f"MMS1 FPI/DIS {data_rate}SkyMap dist"
        mms.psd_moments(vdf, generate_ts(64.0, 100, tensor_order=0))

    @data({"energy_range": [1, 1000]}, {"no_sc_pot": True})
    def test_psd_moments_options(self, options):
        vdf = generate_vdf(64.0, 100, (32, 32, 16))
        vdf.data.attrs["FIELDNAM"] = "MMS1 FPI/DIS brstSkyMap dist"
        mms.psd_moments(vdf, generate_ts(64.0, 100, tensor_order=0), **options)

    @data(
        (
            np.random.random((100, 32)),  # energy
            np.random.random((10000, 32)),  # delta_v
            random.random(),  # q_e
            np.random.random(100),  # sc_pot
            random.random(),  # p_mass
            random.choice([True, False]),  # flag_inner_electron
            random.random(),  # w_inner_electron
            np.random.random((100, 32, 16)),  # phi
            np.random.random((100, 32, 16)),  # theta
            np.arange(32),  # int_energies
            np.random.random((100, 32, 32, 16)),  # vdf
            np.random.random((100, 32, 16)),  # delta_ang
        )
    )
    def test_moms(self, value):
        result = _moms.__wrapped__(*value)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertIsInstance(result[3], np.ndarray)

    @staticmethod
    def _drifting_maxwellian(n, t_ev, v_kms, n_t=2, energy1=None):
        # Isotropic proton Maxwellian on an FPI-like burst skymap (32
        # log-spaced energies, 32 phi, 16 theta). One energy table unless
        # energy1 is given, in which case samples alternate energy0/energy1.
        m_p, q_e = constants.proton_mass, constants.elementary_charge
        ratio = (3e4 / 10.0) ** (1 / 31)
        energy0 = 10.0 * ratio ** np.arange(32)
        energy1 = energy0 if energy1 is None else energy1
        phi = 5.625 + 11.25 * np.arange(32)
        theta = 5.625 + 11.25 * np.arange(16)

        step_table = np.zeros(n_t, dtype=np.uint8)
        if energy1 is not energy0:
            step_table = np.arange(n_t, dtype=np.uint8) % 2

        energy = np.where(step_table[:, None] == 1, energy1, energy0)

        # Particle velocity is minus the FPI look direction
        speed = np.sqrt(2 * q_e * energy / m_p)[..., None, None]
        ph, th = np.deg2rad(phi)[:, None], np.deg2rad(theta)[None, :]
        v_x = -speed * np.sin(th) * np.cos(ph)
        v_y = -speed * np.sin(th) * np.sin(ph)
        v_z = -speed * np.cos(th) * np.ones_like(ph)

        v_d = np.array(v_kms) * 1e3
        v_th2 = 2 * q_e * t_ev / m_p
        dv2 = (v_x - v_d[0]) ** 2 + (v_y - v_d[1]) ** 2 + (v_z - v_d[2]) ** 2
        vdf = n * 1e-6 / (np.pi * v_th2) ** 1.5 * np.exp(-dv2 / v_th2)  # s^3/cm^6

        time = generate_timeline(1 / 0.15, n_t)
        vdf = pyrf.ts_skymap(
            time,
            vdf,
            energy,
            np.tile(phi, (n_t, 1)),
            theta,
            energy0=energy0,
            energy1=energy1,
            esteptable=step_table,
            attrs={"FIELDNAM": "MMS1 FPI/DIS brstSkyMap dist"},
            glob_attrs={
                "species": "ions",
                "delta_energy_plus": energy * (np.sqrt(ratio) - 1),
                "delta_energy_minus": energy * (1 - 1 / np.sqrt(ratio)),
            },
        )
        return vdf, pyrf.ts_scalar(time, np.zeros(n_t))

    def test_psd_moments_drifting_maxwellian(self):
        v_kms = [200.0, 150.0, -250.0]
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, v_kms)
        n, v, _, p2, t, _ = mms.psd_moments(vdf, sc_pot)

        self.assertAlmostEqual(float(n.data[0]), 1.0, delta=0.01)
        np.testing.assert_allclose(v.data[0], v_kms, atol=2.0)
        np.testing.assert_allclose(np.diag(t.data[0]), 1000.0, rtol=0.01)

        # Off-diagonal elements: T is isotropic (0) and the full second moment
        # p2 reduces to m n Vi Vj (nPa). Pxz/Pyz used the wrong angular kernel.
        i_row, i_col = [0, 0, 1], [1, 2, 2]
        np.testing.assert_allclose(t.data[0][i_row, i_col], 0.0, atol=5.0)
        v_d = np.array(v_kms) * 1e3
        p2_expected = constants.proton_mass * 1e6 * np.outer(v_d, v_d) * 1e9
        np.testing.assert_allclose(
            p2.data[0][i_row, i_col], p2_expected[i_row, i_col], rtol=0.02
        )

    def test_psd_moments_tables_differ_in_some_channels(self):
        # Tables that differ in all but one channel must use the alternating-table
        # speed widths (which don't use delta_energy_*). `all(e_tmp) == 0` flagged
        # them as a single table, so zero delta_energy widths gave n = 0.
        ratio = (3e4 / 10.0) ** (1 / 31)
        energy1 = 10.0 * ratio ** (np.arange(32) + 0.5)
        energy1[0] = 10.0
        vdf, sc_pot = self._drifting_maxwellian(
            1.0, 1000.0, [200.0, 150.0, -250.0], n_t=4, energy1=energy1
        )
        vdf.attrs["delta_energy_plus"] = np.zeros_like(vdf.attrs["delta_energy_plus"])
        vdf.attrs["delta_energy_minus"] = np.zeros_like(vdf.attrs["delta_energy_minus"])
        n, _, _, _, _, _ = mms.psd_moments(vdf, sc_pot)
        np.testing.assert_allclose(n.data, 1.0, rtol=0.01)

    def test_psd_moments_partial_moments_mask(self):
        vdf, sc_pot = self._drifting_maxwellian(1.0, 1000.0, [200.0, 150.0, -250.0])
        n_full = mms.psd_moments(vdf, sc_pot)[0].data

        # Not a 0/1 mask -> ignored, full moments (it used to be applied as weights)
        weights = 0.5 * np.ones(vdf.data.shape)
        n_weights = mms.psd_moments(vdf, sc_pot, partial_moments=weights)[0].data
        np.testing.assert_allclose(n_weights, n_full)

        # A 0/1 mask is applied: keeping all bins gives n, keeping none gives 0
        ones = np.ones(vdf.data.shape, dtype=int)
        n_ones = mms.psd_moments(vdf, sc_pot, partial_moments=ones)[0].data
        np.testing.assert_allclose(n_ones, n_full)
        mask = ones.copy()
        mask[:, :, :16, :] = 0
        n_half = mms.psd_moments(vdf, sc_pot, partial_moments=mask)[0].data
        self.assertTrue(np.all(n_half < 0.9 * n_full))


@ddt
class PsdRebinTestCase(unittest.TestCase):
    @data(generate_vdf(64.0, 100, (32, 32, 16), energy01=True, species="ions"))
    def test_psd_rebin_vdf_type(self, vdf):
        with self.assertRaises(TypeError):
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.attrs["energy0"],
                vdf.attrs["energy1"],
                vdf.attrs["esteptable"],
            )

    @data(generate_vdf(64.0, 100, (32, 32, 16), energy01=True, species="ions"))
    def test_psd_rebin_phi_type(self, vdf):
        with self.assertRaises(TypeError):
            mms.psd_rebin(
                vdf.data,
                vdf.phi,
                vdf.attrs["energy0"],
                vdf.attrs["energy1"],
                vdf.attrs["esteptable"],
            )
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.energy[0, :],
                vdf.attrs["energy1"],
                vdf.attrs["esteptable"],
            )
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.attrs["energy0"],
                vdf.energy[1, :],
                vdf.attrs["esteptable"],
            )
            mms.psd_rebin(
                vdf.data,
                vdf.phi.data,
                vdf.attrs["energy0"],
                vdf.attrs["energy1"],
                pyrf.ts_scalar(vdf.time.data, vdf.attrs["esteptable"]),
            )

    @data(generate_vdf(64.0, 100, (32, 32, 16), energy01=True, species="ions"))
    def test_psd_rebin_output(self, vdf):
        result = mms.psd_rebin(
            vdf,
            vdf.phi.data,
            vdf.attrs["energy0"],
            vdf.attrs["energy1"],
            vdf.attrs["esteptable"],
        )
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(len(result[0]), 50)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertListEqual(list(result[1].shape), [50, 64, 32, 16])
        self.assertIsInstance(result[2], np.ndarray)
        self.assertEqual(len(result[2]), 64)
        self.assertIsInstance(result[3], np.ndarray)
        self.assertListEqual(list(result[3].shape), [50, 32])

    def test_psd_rebin_values(self):
        # 30 ms DES-like burst with alternating energy tables and f(t) = t + 1,
        # phi increasing within each pair (no wrap).
        n_t = 20
        time = np.datetime64("2020-01-01", "ns") + np.arange(n_t) * np.timedelta64(
            30, "ms"
        )
        vdf = generate_vdf(1.0, n_t, (32, 32, 16), energy01=True)
        vdf = vdf.assign_coords(time=time)
        vdf.data.data[...] = (np.arange(n_t) + 1.0)[:, None, None, None]
        phi = vdf.phi.data.astype(np.float64)
        phi[1::2] += 5.625

        time_r, vdf_r, _, _ = mms.psd_rebin(
            vdf,
            phi,
            vdf.attrs["energy0"],
            vdf.attrs["energy1"],
            vdf.attrs["esteptable"],
        )

        # Time stamps at the middle of each pair (the time step used to overflow int16)
        np.testing.assert_array_equal(time_r, time[::2] + np.timedelta64(15, "ms"))

        # Every pair is filled, including the last one (used to be all zeros).
        # esteptable[2k] = 0 -> energy0 (sample 2k) on even rows, energy1 on odd.
        expected_even = np.arange(1.0, n_t, 2)[:, None, None, None]
        expected_odd = np.arange(2.0, n_t + 1, 2)[:, None, None, None]
        np.testing.assert_array_equal(
            vdf_r[:, 0:63:2, ...],
            np.broadcast_to(expected_even, vdf_r[:, 0:63:2].shape),
        )
        np.testing.assert_array_equal(
            vdf_r[:, 1:64:2, ...], np.broadcast_to(expected_odd, vdf_r[:, 1:64:2].shape)
        )

    @idata(itertools.product([False, True], [0, 1]))
    @unpack
    def test_psd_rebin_energy_table_order(self, phi_wrap, first_step):
        # The lower table (energy0) must always go to the even channels, whether
        # or not phi wraps between the two samples of a pair (the phi-wrap
        # branch ignored the step table). Each sample holds a constant value,
        # so the order does not depend on the phi shift.
        n_t = 8
        vdf = generate_vdf(64.0, n_t, (32, 32, 16), energy01=True)
        step_table = (np.arange(n_t) + first_step) % 2
        vdf.data.data[...] = (np.arange(n_t) + 1.0)[:, None, None, None]
        phi = vdf.phi.data.astype(np.float64) + 5.625
        phi[1::2] += -5.625 if phi_wrap else 5.625

        _, vdf_r, energy_r, _ = mms.psd_rebin(
            vdf, phi, vdf.attrs["energy0"], vdf.attrs["energy1"], step_table
        )

        # Samples using energy0 (step 0) in the even channels, energy1 (step 1)
        # in the odd channels
        samples = np.arange(n_t).reshape(-1, 2)
        pair_steps = step_table.reshape(-1, 2)
        from_energy0 = samples[pair_steps == 0] + 1.0
        from_energy1 = samples[pair_steps == 1] + 1.0
        np.testing.assert_array_equal(vdf_r[:, 0:63:2, 0, 0].max(axis=1), from_energy0)
        np.testing.assert_array_equal(vdf_r[:, 0:63:2, 0, 0].min(axis=1), from_energy0)
        np.testing.assert_array_equal(vdf_r[:, 1:64:2, 0, 0].max(axis=1), from_energy1)
        np.testing.assert_array_equal(vdf_r[:, 1:64:2, 0, 0].min(axis=1), from_energy1)
        np.testing.assert_array_equal(energy_r[0:63:2], vdf.attrs["energy0"])


class EstimatePhaseSpeedTestCase(unittest.TestCase):
    def test_estimate_phase_speed(self):
        # Power ridge along f = v k / (2 pi), v = 20 km/s: the speed is recovered
        # and the caller's power spectrum is unchanged (parts were set to 0)
        k = np.linspace(-0.3, 0.3, 121)
        freq = np.linspace(0.0, 1000.0, 201)
        k_mat, f_mat = np.meshgrid(k, freq)
        power = np.exp(-(((f_mat - 2e4 * k_mat / (2 * np.pi)) / 20.0) ** 2))
        power *= k_mat > 0
        power[50, 60] = np.nan
        power_in = power.copy()

        vph = mms.estimate_phase_speed(power, freq, k, 100.0)

        self.assertAlmostEqual(vph / 2e4, 1.0, delta=0.01)
        np.testing.assert_array_equal(power, power_in)


@ddt
class FeepsActiveEyesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"], ["sitl", "l2"]))
    @unpack
    def test_feeps_active_eyes_output(self, data_rate, dtype, lev):
        result = mms.feeps_active_eyes(
            {"tmmode": data_rate, "dtype": dtype, "lev": lev},
            TEST_TINT,
            random.randint(1, 4),
        )
        self.assertIsInstance(result, dict)

        result = mms.feeps_active_eyes(
            {"tmmode": data_rate, "dtype": dtype, "lev": lev},
            pyrf.iso86012datetime64(np.array(TEST_TINT)),
            str(random.randint(1, 4)),
        )
        self.assertIsInstance(result, dict)


@ddt
class FeepsCorrectEnergiesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_correct_energies_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )

        result = mms.feeps_correct_energies(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class FeepsFlatFieldCorrectionsTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_flat_field_corrections_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )

        result = mms.feeps_flat_field_corrections(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)

    @idata(range(1, 5))
    def test_feeps_flat_field_corrections_values(self, mms_id):
        feeps_alle = generate_feeps(64.0, 100, "brst", "ion", "l2", "flux", mms_id)
        feeps_ref = feeps_alle.copy(deep=True)

        result = mms.feeps_flat_field_corrections(feeps_alle)

        # Each eye is scaled by its gain (1 if not in the table)
        for k in filter(lambda x: x[:3] in ["top", "bot"], feeps_ref):
            sensor, eye = k.split("-")
            gain = g_corr.get(f"mms{mms_id}-{sensor[:3]}{int(eye)}", 1.0)
            np.testing.assert_array_equal(result[k].data, feeps_ref[k].data * gain)
            self.assertDictEqual(result[k].attrs, feeps_ref[k].attrs)

        self.assertDictEqual(result.attrs, feeps_ref.attrs)

        # The caller's data must not be changed (they used to be scaled in place)
        xr.testing.assert_identical(feeps_alle, feeps_ref)


@ddt
class FeepsOmniTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_omni_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_omni(feeps_alle)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsPadTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pad_ouput(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, tensor_order=1))
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsPadSpinAvgTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pad_spin_avg(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        feeps_pad = mms.feeps_pad(feeps_alle, generate_ts(64.0, 100, tensor_order=1))
        result = mms.feeps_pad_spinavg(feeps_pad, feeps_alle.spinsectnum)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsPitchAnglesTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_pitch_angles_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_pitch_angles(
            feeps_alle, generate_ts(64.0, 100, tensor_order=1)
        )
        self.assertIsInstance(result[0], xr.DataArray)


@ddt
class FeepsRemoveBadDataTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_remove_bad_data_output(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )

        result = mms.feeps_remove_bad_data(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class FeepsRemoveSunTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_remove_sun(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)

        result = mms.feeps_remove_sun(feeps_alle)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class FeepsSpinAvgTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_spin_avg(self, data_rate, dtype):
        # Generate fake FEEPS data
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        feeps_alle, _ = mms.feeps_split_integral_ch(feeps_alle)
        feeps_omni = mms.feeps_omni(feeps_alle)

        result = mms.feeps_spin_avg(feeps_omni, feeps_alle.spinsectnum)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class FeepsSplitIntegralChTestCase(unittest.TestCase):
    @idata(itertools.product(["srvy", "brst"], ["electron", "ion"]))
    @unpack
    def test_feeps_split_integral_ch(self, data_rate, dtype):
        feeps_alle = generate_feeps(
            64.0, 100, data_rate, dtype, "l2", "flux", random.randint(1, 4)
        )
        mms.feeps_split_integral_ch(feeps_alle)


@ddt
class FkPowerSpectrum4scTestCase(unittest.TestCase):
    @data((None, None), (random.random(), None), (None, [0.1, 1]))
    @unpack
    def test_fk_power_spectrum_4sc(self, df, f_range):
        e_mms = [generate_ts(64.0, 100, tensor_order=0) for _ in range(4)]
        r_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]
        b_mms = [generate_ts(64.0, 100, tensor_order=1) for _ in range(4)]

        result = mms.fk_power_spectrum_4sc(
            e_mms, r_mms, b_mms, TEST_TINT, df=df, f_range=f_range
        )
        self.assertIsInstance(result, xr.Dataset)


@ddt
class GetDataDownloadTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.get_data")

    def test_get_file_content_aws_error(self):
        # S3 errors are raised (they were logged, then gave UnboundLocalError)
        s3_object = mock.Mock(key="mms1/file.cdf")
        s3_object.get.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "We encountered an error"}},
            "GetObject",
        )

        with self.assertLogs("pyrfu", level="ERROR"), self.assertRaises(ClientError):
            self.module._get_file_content_sources("aws", s3_object)

    def test_get_file_content_sdc_error(self):
        session = mock.Mock()
        session.get.side_effect = requests.ConnectionError("connection reset")

        with self.assertLogs("pyrfu", level="ERROR"):
            with self.assertRaises(requests.ConnectionError):
                self.module._get_file_content_sources("sdc", "url", session, {})

    def test_get_file_content_sdc_timeout(self):
        # SDC downloads time out instead of hanging on a stalled connection
        session = mock.Mock()
        session.get.return_value = mock.Mock(content=b"cdf")

        content = self.module._get_file_content_sources("sdc", "url", session, {})

        self.assertEqual(content, b"cdf")
        timeout = session.get.call_args.kwargs["timeout"]
        self.assertIsNotNone(timeout)
        self.assertTrue(all(0 < t_ < np.inf for t_ in np.atleast_1d(timeout)))

    @data(
        [],  # no file
        ["url"],  # download error
    )
    def test_get_data_keeps_sdc_session(self, file_names):
        session = mock.Mock()
        session.get.side_effect = requests.ConnectionError("connection reset")
        sources = (file_names, session, {})
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ):
            with self.assertRaises((FileNotFoundError, requests.ConnectionError)):
                with (
                    self.assertLogs("pyrfu", level="ERROR")
                    if file_names
                    else nullcontext()
                ):
                    mms.get_data("b_gse_fgm_srvy_l2", tint, 1, source="sdc")

        session.close.assert_not_called()  # shared, cached session


def _sdc_session(chunks=(b"cdf", b"data"), status_error=None, stream_error=None):
    # Mocked SDC session whose get() returns a streamed response
    response = mock.MagicMock()
    if status_error:
        response.raise_for_status.side_effect = status_error

    def iter_content(chunk_size=1):
        for chunk in chunks:
            yield chunk
        if stream_error:
            raise stream_error

    response.iter_content.side_effect = iter_content
    response.json.return_value = {"files": []}
    session = mock.MagicMock()
    session.get.return_value = response
    response.__enter__.return_value = response
    return session


class DownloadDataTestCase(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.download_data")

    def test_download_file(self):
        with tempfile.TemporaryDirectory() as root:
            out_file = os.path.join(root, "mms1", "file.cdf")
            session = _sdc_session()
            self.module._download_file(session, "url", {}, out_file, 7)

            with open(out_file, "rb") as file:
                self.assertEqual(file.read(), b"cdfdata")
            self.assertListEqual(os.listdir(os.path.dirname(out_file)), ["file.cdf"])

        timeout = session.get.call_args.kwargs["timeout"]
        self.assertTrue(all(0 < t_ < np.inf for t_ in np.atleast_1d(timeout)))

    def test_download_file_errors(self):
        # Neither an HTTP error page nor an interrupted download leaves a file
        errors = [
            {"status_error": requests.HTTPError("404")},
            {"stream_error": requests.ConnectionError("connection reset")},
        ]
        for error in errors:
            with tempfile.TemporaryDirectory() as root:
                out_file = os.path.join(root, "file.cdf")
                with self.assertRaises(requests.RequestException):
                    self.module._download_file(
                        _sdc_session(**error), "url", {}, out_file
                    )
                self.assertListEqual(os.listdir(root), [])

    def test_download_data(self):
        files = [
            {
                "url": "url",
                "file_name": "mms1_fgm_srvy_l2_20190914_v5.211.0.cdf",
                "timetag": "2019-09-14T00:00:00",
                "size": 7,
            }
        ]
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(self.module, "list_files_sdc", return_value=files):
                with mock.patch.object(
                    self.module, "_login_lasp", return_value=(_sdc_session(), {}, "")
                ):
                    self.module.download_data(
                        "b_gse_fgm_srvy_l2",
                        ["2019-09-14T07:54:00", "2019-09-14T08:11:00"],
                        1,
                        root,
                    )

            out_file = os.path.join(
                root,
                "mms1",
                "fgm",
                "srvy",
                "l2",
                "",
                "2019",
                "09",
                files[0]["file_name"],
            )
            with open(out_file, "rb") as file:
                self.assertEqual(file.read(), b"cdfdata")

    def test_list_files_sdc_request(self):
        # The SDC listing has a timeout and raises on HTTP errors
        module = importlib.import_module("pyrfu.mms.list_files_sdc")
        var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        session = _sdc_session(status_error=requests.HTTPError("500"))
        with mock.patch.object(module, "_login_lasp", return_value=(session, {}, "x/")):
            with self.assertRaises(requests.HTTPError):
                module.list_files_sdc(tint, 1, var)

        self.assertIsNotNone(session.get.call_args.kwargs.get("timeout"))


class GetFeepsTestCase(unittest.TestCase):
    TINT = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.get_feeps_alleyes")

    def _fake_read(self, dset_name, cdf_names, tint, verbose, data_path, source):
        # (time, energy) time series for every variable
        times = generate_timeline(1.0, 5)
        energy = np.arange(1, 4, dtype=float)
        return {
            name: xr.DataArray(
                np.ones((5, 3)), coords=[times, energy], dims=["time", "Epoch_E"]
            )
            for name in cdf_names
        }

    def test_get_feeps_alleyes_source(self):
        # All the eyes are read at once (one download per file), from the source
        with mock.patch.object(
            self.module, "_db_get_ts_dict", side_effect=self._fake_read
        ) as read:
            out = mms.get_feeps_alleyes("fluxe_brst_l2", self.TINT, 2, source="aws")

        read.assert_called_once()
        dset_name, cdf_names = read.call_args.args[:2]
        self.assertEqual(dset_name, "mms2_feeps_brst_l2_electron")
        self.assertEqual(read.call_args.args[5], "aws")

        eyes = [k for k in out.data_vars if k not in ["spinsectnum", "pitch_angle"]]
        self.assertEqual(len(cdf_names), len(eyes) + 2)
        self.assertIn(
            "mms2_epd_feeps_brst_l2_electron_top_intensity_sensorid_3", cdf_names
        )
        self.assertIn("energy_top-3", out["top-3"].dims)
        self.assertEqual(out["top-3"].attrs["species"], "electrons")

    def test_get_feeps_omni_source(self):
        module = importlib.import_module("pyrfu.mms.get_feeps_omni")

        with mock.patch.object(
            module, "get_feeps_alleyes", side_effect=RuntimeError("stop")
        ) as alleyes:
            with self.assertRaises(RuntimeError):
                mms.get_feeps_omni("fluxe_brst_l2", self.TINT, 2, source="sdc")

        self.assertEqual(alleyes.call_args.kwargs["source"], "sdc")

    @unittest.skipUnless(
        os.environ.get("PYRFU_NETWORK_TESTS"), "set PYRFU_NETWORK_TESTS=1 to run"
    )
    def test_get_feeps_alleyes_aws(self):
        out = mms.get_feeps_alleyes(
            "fluxe_brst_l2", self.TINT, 2, verbose=False, source="aws"
        )
        self.assertEqual(len(out.data_vars), 20)
        self.assertGreater(out.sizes["time"], 1000)


def _cdf_four_columns(t_start="2019-09-14T07:54:00"):
    # In-memory CDF with (N, 4) variables labelled as in the MMS files: FGM
    # (x, y, z, r representation), MEC quaternions (qx, qy, qz, qw) and MEC
    # B fields (labels only, last one the magnitude), with padded labels
    time = np.datetime64(t_start, "ns") + np.arange(5) * np.timedelta64(1, "s")
    data = np.tile([1.0, 2.0, 3.0, 4.0], (5, 1))

    cdf = pycdfpp.CDF()
    cdf.add_variable("Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000)
    labels = {
        "rep_vec_tot": ["x", "y", "z", "r"],
        "rep_quat": ["qx  ", "qy  ", "qz  ", "qw  "],
        "label_b_gsm": ["b_gsm_x ", "b_gsm_y ", "b_gsm_z ", "b_gsm_mag"],
        "label_other": ["a", "b", "c", "d"],
    }
    for name, values in labels.items():
        cdf.add_variable(name, values=np.array([values]))

    for name, attrs in {
        "mms1_fgm_b_gse_srvy_l2": {"REPRESENTATION_1": "rep_vec_tot"},
        "mms1_mec_quat_eci_to_gse": {"REPRESENTATION_1": "rep_quat"},
        "mms1_mec_bsc_gsm": {"LABL_PTR_1": "label_b_gsm"},
        "mms1_mec_other": {"LABL_PTR_1": "label_other"},
    }.items():
        cdf.add_variable(name, values=data)
        cdf[name].add_attribute("DEPEND_0", "Epoch")
        for key, value in attrs.items():
            cdf[name].add_attribute(key, value)

    return bytes(pycdfpp.save(cdf))


@ddt
class GetTsTestCase(unittest.TestCase):
    @data(
        ("mms1_fgm_b_gse_srvy_l2", ["x", "y", "z"]),
        ("mms1_mec_quat_eci_to_gse", ["qx", "qy", "qz", "qw"]),
        ("mms1_mec_bsc_gsm", [0, 1, 2]),
        ("mms1_mec_other", [0, 1, 2, 3]),
    )
    @unpack
    def test_get_ts_four_columns(self, cdf_name, comps):
        # Only the magnitude of (x, y, z, magnitude) variables is removed
        result = get_ts(_cdf_four_columns(), cdf_name)
        np.testing.assert_array_equal(
            result.data[0], [1.0, 2.0, 3.0, 4.0][: len(comps)]
        )
        self.assertListEqual(list(result[result.dims[1]].data), comps)

    @data(
        (1, 0.0, 30.0, 15),  # start times, single record (used to be NaT)
        (3, 0.0, 30.0, 15),  # start times
        (1, 15.0, 15.0, 0),  # centred times, single record
        (3, 15.0, 15.0, 0),  # centred times
        (3, 0.0, 20.0, 15),  # deltas do not match the 30 ms sampling period
    )
    @unpack
    def test_get_ts_fpi_epoch_shift(self, n_records, d_minus, d_plus, shift):
        # FPI epochs are moved to the centre of [t - delta_minus, t + delta_plus]
        # (in ms), or by half the sampling period if the accumulation does not
        # match it, as mms.variable2ts in irfu-matlab
        time = np.datetime64("2019-09-14T07:54:00", "ns")
        time = time + np.arange(n_records) * np.timedelta64(30, "ms")

        cdf = pycdfpp.CDF()
        cdf.add_variable(
            "Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000
        )
        cdf["Epoch"].add_attribute("DELTA_MINUS_VAR", "Epoch_minus_var")
        cdf["Epoch"].add_attribute("DELTA_PLUS_VAR", "Epoch_plus_var")
        for name, value in [("Epoch_minus_var", d_minus), ("Epoch_plus_var", d_plus)]:
            cdf.add_variable(name, values=np.array([value]), is_nrv=True)
            cdf[name].add_attribute("UNITS", "ms")
        cdf.add_variable("mms1_des_numberdensity_brst", values=np.ones(n_records))
        cdf["mms1_des_numberdensity_brst"].add_attribute("DEPEND_0", "Epoch")
        content = bytes(pycdfpp.save(cdf))

        if d_minus + d_plus == 30.0 or n_records == 1:
            result = get_ts(content, "mms1_des_numberdensity_brst")
        else:
            with self.assertLogs("pyrfu", level="WARNING"):
                result = get_ts(content, "mms1_des_numberdensity_brst")

        np.testing.assert_array_equal(
            result.time.data, time + np.timedelta64(shift, "ms")
        )


def _cdf_fpi_brst_dist(n_records, t_start):
    # In-memory FPI burst ion distribution laid out as in the L2 files: record
    # varying phi, energy, steptable parity and energy deltas, with alternating
    # energy tables (no energy0 and energy1 variables)
    time = np.datetime64(t_start, "ns") + np.arange(n_records) * np.timedelta64(
        150, "ms"
    )
    parity = np.arange(n_records) % 2
    energies = np.array(
        [[10.0, 20.0, 40.0, 80.0, 160.0], [12.0, 24.0, 48.0, 96.0, 192.0]]
    )

    cdf = pycdfpp.CDF()
    cdf.add_variable("Epoch", values=time, data_type=pycdfpp.DataType.CDF_TIME_TT2000)
    cdf.add_variable(
        "mms1_dis_phi_brst", values=np.tile(np.arange(4.0), (n_records, 1))
    )
    cdf.add_variable("mms1_dis_theta_brst", values=np.arange(3.0)[None], is_nrv=True)
    cdf.add_variable("mms1_dis_energy_brst", values=energies[parity].reshape(-1, 5))
    cdf.add_variable("mms1_dis_steptable_parity_brst", values=parity.astype(np.uint8))
    cdf.add_variable("mms1_dis_energy_delta_brst", values=np.ones((n_records, 5)))
    cdf.add_variable(
        "mms1_dis_dist_brst",
        values=np.arange(n_records * 60, dtype=float).reshape(n_records, 4, 3, 5),
    )
    for i, dep in enumerate(["Epoch", "mms1_dis_phi_brst", "mms1_dis_theta_brst"]):
        cdf["mms1_dis_dist_brst"].add_attribute(f"DEPEND_{i:d}", dep)
    cdf["mms1_dis_dist_brst"].add_attribute("DEPEND_3", "mms1_dis_energy_brst")

    return bytes(pycdfpp.save(cdf))


@ddt
class GetDataEmptyFilesTestCase(unittest.TestCase):
    # Files with no record, or no record in tint, do not change the result
    tint = ["2019-09-14T07:54:00", "2019-09-14T07:55:00"]

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.get_data")

    def _get_data(self, var_str, contents):
        sources = ([f"file_{i:d}" for i in range(len(contents))], None, {})
        with mock.patch.object(
            self.module, "_list_files_sources", return_value=sources
        ):
            with mock.patch.object(
                self.module, "_get_file_content_sources", side_effect=contents
            ):
                return mms.get_data(var_str, self.tint, 1, verbose=False, source="sdc")

    @data((0, 3), (3, 0), (0, 0))
    @unpack
    def test_get_data_dist_empty_files(self, n_first, n_second):
        empty = _cdf_fpi_brst_dist(0, "2019-09-14T07:54:00")
        full = _cdf_fpi_brst_dist(3, "2019-09-14T07:54:10")
        outside = _cdf_fpi_brst_dist(2, "2019-09-14T08:30:00")
        contents = [full if n_first else empty, full if n_second else empty, outside]

        result = self._get_data("pdi_fpi_brst_l2", contents)

        n_records = 3 if n_first or n_second else 0
        self.assertEqual(result.sizes["time"], n_records)
        self.assertEqual(len(result.attrs["esteptable"]), n_records)
        if n_records:
            expected = self._get_data("pdi_fpi_brst_l2", [full])
            np.testing.assert_array_equal(result.data.data, expected.data.data)
            np.testing.assert_array_equal(
                result.energy0, [10.0, 20.0, 40.0, 80.0, 160.0]
            )
            np.testing.assert_array_equal(
                result.energy1, [12.0, 24.0, 48.0, 96.0, 192.0]
            )

    def test_get_data_ts_empty_files(self):
        contents = [
            _cdf_four_columns("2019-09-14T08:30:00"),
            _cdf_four_columns("2019-09-14T07:54:10"),
        ]
        result = self._get_data("b_gse_fgm_srvy_l2", contents)
        self.assertEqual(result.sizes["time"], 5)
        self.assertListEqual(list(result.shape), [5, 3])


class _FakeS3Bucket:
    # Bucket listing the given keys with bucket.objects.filter(Prefix=...), and
    # recording the prefixes it was asked for.
    def __init__(self, keys, error=None):
        self.keys, self.error, self.prefixes = keys, error, []
        self.objects = self

    def filter(self, Prefix):  # pylint: disable=invalid-name
        self.prefixes.append(Prefix)

        if self.error is not None:
            raise self.error

        return [
            mock.Mock(key=key, size=len(key))
            for key in self.keys
            if key.startswith(Prefix)
        ]


@ddt
class ListFilesAwsTestCase(unittest.TestCase):
    BUCKET = "gov-nasa-hdrl-data1"
    HELIO = "spdf/cdaweb/data/mms"  # key prefix in the bucket
    FGM_BRST = {"inst": "fgm", "tmmode": "brst", "lev": "l2", "dtype": ""}
    FPI_FAST = {"inst": "fpi", "tmmode": "fast", "lev": "l2", "dtype": "des-moms"}

    def _list(self, keys, tint, var, bucket_prefix="", error=None):
        bucket = _FakeS3Bucket(keys, error)
        resource = mock.Mock()
        resource.Bucket.return_value = bucket

        with mock.patch.object(
            list_files_aws_module, "_s3_resource", return_value=resource
        ):
            out = mms.list_files_aws(tint, 1, var, bucket_prefix=bucket_prefix)

        return out, resource, bucket

    def test_list_files_aws_brst_month_directory(self):
        # HelioCloud: burst files directly in the month directory. The file
        # starting before the time interval covers its start; earlier ones and
        # those starting at or after its end are not needed.
        directory = f"{self.HELIO}/mms1/fgm/brst/l2/2019/09"
        times = ["051233", "075043", "075403", "080703", "081100", "090000"]
        keys = [f"{directory}/mms1_fgm_brst_l2_20190914{t}_v5.207.0.cdf" for t in times]
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        bucket_prefix = f"{self.BUCKET}/{self.HELIO}"
        out, resource, _ = self._list(keys, tint, self.FGM_BRST, bucket_prefix)

        resource.Bucket.assert_called_once_with(self.BUCKET)
        self.assertListEqual([f["full_name"] for f in out], [keys[1], keys[2], keys[3]])
        self.assertEqual(out[0]["timetag"], "2019-09-14T07:50:43")
        self.assertEqual(out[0]["file_size"], len(keys[1]))

    def test_list_files_aws_brst_day_directory(self):
        # SDC layout: burst files in a day directory, looked at only if there is
        # no file in the month directory. Keys are built with "/" on all systems.
        prefix = "my-bucket/mms"
        directory = "mms/mms1/fgm/brst/l2/2019/09/14"
        keys = [f"{directory}/mms1_fgm_brst_l2_20190914075403_v5.207.0.cdf"]
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        out, resource, bucket = self._list(keys, tint, self.FGM_BRST, prefix)

        resource.Bucket.assert_called_once_with("my-bucket")
        self.assertListEqual([f["full_name"] for f in out], keys)
        self.assertIn(
            "mms/mms1/fgm/brst/l2/2019/09/14/mms1_fgm_brst_l2_20190914", bucket.prefixes
        )
        self.assertTrue(all("\\" not in p for p in bucket.prefixes))

    def test_list_files_aws_latest_version_and_previous_day(self):
        # Only the latest version of each file; a file of the previous day covers
        # the start of the time interval.
        directory = f"{self.HELIO}/mms1/fpi/fast/l2/des-moms/2019/09"
        stem = "mms1_fpi_fast_l2_des-moms"
        keys = [
            f"{directory}/{stem}_20190913220000_v3.4.0.cdf",
            f"{directory}/{stem}_20190914000000_v3.3.0.cdf",
            f"{directory}/{stem}_20190914000000_v3.4.0.cdf",
            f"{directory}/{stem}_20190914000000_v3.10.0.cdf",
            f"{directory}/{stem}_20190914020000_v3.4.0.cdf",
            f"{directory}/mms1_fpi_fast_l2_dis-moms_20190914000000_v3.4.0.cdf",
        ]
        tint = ["2019-09-13T23:30:00", "2019-09-14T01:00:00"]

        out, _, _ = self._list(keys, tint, self.FPI_FAST, f"{self.BUCKET}/{self.HELIO}")

        self.assertListEqual([f["full_name"] for f in out], [keys[0], keys[3]])

    def test_list_files_aws_listing_error(self):
        error = ClientError({"Error": {"Code": "NoSuchBucket"}}, "ListObjectsV2")
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

        with self.assertRaises(FileNotFoundError):
            self._list([], tint, self.FGM_BRST, "missing/mms", error=error)

    @data(
        (["2019-09-14T07:54:00", "2019-09-14T08:11:00"], FGM_BRST),
    )
    @unpack
    def test_list_files_aws_input(self, tint, var):
        with self.assertRaises(TypeError):
            mms.list_files_aws(tuple(tint), 1, var)

        with self.assertRaises(TypeError):
            mms.list_files_aws(list(pyrf.iso86012datetime64(np.array(tint))), 1, var)

    @data(
        ("s3://my-bucket/data/mms/", ("my-bucket", "data/mms")),
        ("my-bucket", ("my-bucket", "")),
        ("", ("gov-nasa-hdrl-data1", "spdf/cdaweb/data/mms")),  # "aws" not set
    )
    @unpack
    def test_bucket_and_prefix(self, config_aws, expected):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")

            with open(config_path, "w", encoding="utf-8") as fs:
                json.dump({"aws": config_aws}, fs)

            with mock.patch.object(list_files_aws_module, "MMS_CFG_PATH", config_path):
                result = list_files_aws_module._bucket_and_prefix()

        self.assertTupleEqual(result, expected)
        self.assertTupleEqual(
            list_files_aws_module._bucket_and_prefix("s3://other/x"), ("other", "x")
        )

    @data(True, False)
    def test_s3_resource_credentials(self, has_credentials):
        # Anonymous requests if there are no AWS credentials (public buckets)
        credentials = mock.Mock() if has_credentials else None

        with mock.patch(
            "boto3.session.Session.get_credentials", return_value=credentials
        ):
            resource = list_files_aws_module._s3_resource()

        signature = resource.meta.client.meta.config.signature_version
        self.assertEqual(signature is UNSIGNED, not has_credentials)

    @unittest.skipUnless(
        os.environ.get("PYRFU_NETWORK_TESTS"), "set PYRFU_NETWORK_TESTS=1 to run"
    )
    def test_list_files_aws_heliocloud(self):
        # Real listing of the public HelioCloud bucket
        tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
        out = mms.list_files_aws(tint, 1, self.FGM_BRST, bucket_prefix="")
        times = [f["full_name"].rsplit("_", 2)[-2] for f in out]
        self.assertListEqual(
            times,
            [
                "20190914075043",
                "20190914075403",
                "20190914075823",
                "20190914080243",
                "20190914080703",
            ],
        )


@ddt
class ReduceTestCase(unittest.TestCase):
    @data("s^3/cm^6", "s^3/m^6", "s^3/km^6")
    def test_reduce_units(self, value):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species="ions")
        vdf.data.attrs["UNITS"] = value
        result = mms.reduce(vdf, np.eye(3), "1d", "pol")
        self.assertIsInstance(result, xr.DataArray)

    @data(
        (False, "ions", np.eye(3), "1d", "pol"),
        (False, "electrons", np.eye(3), "1d", "pol"),
        (True, "ions", np.eye(3), "1d", "pol"),
        (True, "electrons", np.eye(3), "1d", "pol"),
        (False, "electrons", generate_ts(64.0, 42, tensor_order=2), "1d", "pol"),
        (False, "ions", np.eye(3), "1d", "pol"),
        # (False, "ions", np.eye(3), "2d", "pol"), mc_pol_2d NotImplementedError
    )
    @unpack
    def test_reduce_output(self, energy01, species, xyz, dim, base):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01, species)
        result = mms.reduce(vdf, xyz, dim, base)
        self.assertIsInstance(result, xr.DataArray)

    @data(
        ("2d", "cart", {}),
        ("1d", "pol", {"vg": np.linspace(-1, 1, 42)}),
        ("1d", "pol", {"lower_e_lim": generate_ts(64.0, 42)}),
        ("1d", "pol", {"vg_edges": np.linspace(-1.01, 1.01, 102)}),
    )
    @unpack
    def test_reduce_options(self, dim, base, options):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=False, species="ions")
        xyz = np.eye(3)
        result = mms.reduce(vdf, xyz, dim, base, **options)
        self.assertIsInstance(result, xr.DataArray)

    @data(
        ("ions", "s^3/m^6", np.array([1, 0, 0]), "1d", "pol", {}),
        ("I AM GROOT", "s^3/m^6", np.eye(3), "1d", "pol", {}),
        ("ions", "bazinga", np.eye(3), "1d", "pol", {}),
        ("ions", "s^3/m^6", np.eye(3), "2d", "pol", {}),
        ("ions", "s^3/m^6", np.eye(3), "1d", "pol", {"lower_e_lim": generate_data(42)}),
    )
    @unpack
    def test_reduce_input(self, species, units, xyz, dim, base, options):
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species=species)
        vdf.data.attrs["UNITS"] = units
        with self.assertRaises((TypeError, ValueError, NotImplementedError)):
            mms.reduce(vdf, xyz, dim, base, **options)

    def test_reduce_default_phi_grid(self):
        # Default azimuthal grid has one point per instrument azimuth. phi is
        # (time, phi), so len(phi) gave one point per time step (5 here).
        vdf = generate_vdf(64.0, 5, [32, 32, 16], energy01=True, species="ions")
        vdf.data.attrs["UNITS"] = "s^3/cm^6"

        reduce_module = importlib.import_module("pyrfu.mms.reduce")
        with mock.patch.object(
            reduce_module, "int_sph_dist", wraps=reduce_module.int_sph_dist
        ) as isd:
            mms.reduce(vdf, np.eye(3), "2d", "cart", n_mc=1)

        phi_grid = isd.call_args.args[5]
        np.testing.assert_allclose(phi_grid, np.deg2rad(5.625 + 11.25 * np.arange(32)))

    @data(True, False)
    def test_reduce_drifting_maxwellian(self, energy_widths):
        # 1 keV proton Maxwellian (n = 1 cm^-3) drifting at (-300, 200, 100) km/s,
        # reduced along x. Uses the channel widths from delta_energy_* or, if
        # missing, the default speed bin edges. The reduced distribution used to
        # be shifted to lower speeds (n 6 %, V 7 % and T 12 % too low).
        v_d = [-300.0, 200.0, 100.0]
        vdf, sc_pot = PsdMomentsTestCase._drifting_maxwellian(1.0, 1000.0, v_d)
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        if not energy_widths:
            del vdf.attrs["delta_energy_minus"]

        v_grid = np.linspace(-2500.0, 2500.0, 251) * 1e3
        result = mms.reduce(vdf, np.eye(3), "1d", "pol", vg=v_grid, n_mc=50)

        v_x, f_x = result.vx.data * 1e3, result.data[0]
        d_v = np.median(np.diff(v_x))
        n = np.sum(f_x) * d_v
        v_bulk = np.sum(v_x * f_x) * d_v / n
        t_x = constants.proton_mass * np.sum((v_x - v_bulk) ** 2 * f_x) * d_v / n
        t_x /= constants.elementary_charge

        self.assertAlmostEqual(n / 1e6, 1.0, delta=0.01)
        self.assertAlmostEqual(v_bulk / 1e3, v_d[0], delta=3.0)
        self.assertAlmostEqual(t_x, 1000.0, delta=40.0)


@ddt
class RemoveEdistBackgroundTestCase(unittest.TestCase):
    MODEL = "mms_fpi_brst_l2_des-bgdist_v1.1.0_p0-2.cdf"

    def setUp(self):
        self.module = importlib.import_module("pyrfu.mms.remove_edist_background")

    def test_load_bgdist_model_local(self):
        with tempfile.TemporaryDirectory() as data_path:
            os.makedirs(os.path.join(data_path, "models", "fpi"))
            file_path = os.path.join(data_path, "models", "fpi", self.MODEL)
            open(file_path, "wb").close()

            with mock.patch.object(self.module.pycdfpp, "load") as load:
                self.module._load_bgdist_model(self.MODEL, "local", data_path)

            load.assert_called_once_with(file_path)

            # Missing model file: clear error with the SDC URL
            with self.assertRaises(FileNotFoundError) as context:
                self.module._load_bgdist_model("missing.cdf", "local", data_path)

            url = "https://lasp.colorado.edu/mms/sdc/public/data/models/fpi/missing.cdf"
            self.assertIn(url, str(context.exception))

    def _load_from_sdc(self, source, s3_resource=None):
        # Load the model with the SDC (and S3) mocked; returns the SDC session
        session = mock.MagicMock()
        session.get.return_value.content = b"sdc cdf bytes"
        login = (session, {"User-Agent": "pyrfu"}, self.module.LASP_PUBL)
        bucket = ("gov-nasa-hdrl-data1", "spdf/cdaweb/data/mms")

        with mock.patch.object(self.module, "_login_lasp", return_value=login):
            with mock.patch.object(
                self.module, "_s3_resource", return_value=s3_resource
            ):
                with mock.patch.object(
                    self.module, "_bucket_and_prefix", return_value=bucket
                ):
                    with mock.patch.object(self.module.pycdfpp, "load") as load:
                        self.module._load_bgdist_model(self.MODEL, source, "")

        return session, load

    def test_load_bgdist_model_sdc(self):
        # Read from the SDC into memory
        session, load = self._load_from_sdc("sdc")

        url = f"https://lasp.colorado.edu/mms/sdc/public/data/models/fpi/{self.MODEL}"
        self.assertEqual(session.get.call_args.args[0], url)
        load.assert_called_once_with(b"sdc cdf bytes")
        session.close.assert_not_called()  # shared, cached session

    def test_load_bgdist_model_aws(self):
        # Read from models/fpi/ in the MMS bucket, without the SDC
        s3_resource = mock.MagicMock()
        s3_object = s3_resource.Object.return_value
        s3_object.get.return_value = {"Body": mock.Mock(read=lambda: b"s3 cdf bytes")}

        session, load = self._load_from_sdc("aws", s3_resource)

        s3_resource.Object.assert_called_once_with(
            "gov-nasa-hdrl-data1", f"spdf/cdaweb/data/mms/models/fpi/{self.MODEL}"
        )
        load.assert_called_once_with(b"s3 cdf bytes")
        session.get.assert_not_called()

    def test_load_bgdist_model_aws_fallback(self):
        # Not in the bucket: read from the SDC
        s3_resource = mock.MagicMock()
        s3_resource.Object.return_value.get.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        with self.assertLogs("pyrfu", level="WARNING"):
            session, load = self._load_from_sdc("aws", s3_resource)

        load.assert_called_once_with(b"sdc cdf bytes")
        session.close.assert_not_called()  # shared, cached session

    def test_models_url(self):
        self.assertEqual(
            self.module._models_url(
                "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"
            ),
            "https://lasp.colorado.edu/mms/sdc/sitl/data/models/fpi/",
        )

    def _inputs(self, n_t=4):
        # Burst DES skymap with alternating energy tables (parity 0, 1, 0, ...),
        # at spin phases of model records 0, 1, 2, ... Model record k is k for
        # parity 0 and 1000 + k for parity 1, so the subtracted background
        # shows which record and table were used.
        vdf = generate_vdf(64.0, n_t, [32, 32, 16], energy01=True, species="electrons")
        vdf.data.data[...] = 1e6
        vdf.data.attrs["CATDESC"] = "MMS1 DES burst distribution"
        vdf.data.attrs["FIELDNAM"] = "mms1_des_dist_brst"

        n_e = pyrf.ts_scalar(vdf.time.data, np.ones(n_t))
        n_e.attrs["GLOBAL"] = {
            "Photoelectron_model_scaling_factor": "0.5",
            "Photoelectron_model_filenames": self.MODEL,
        }
        startdelphi = pyrf.ts_scalar(vdf.time.data, 16 * np.arange(n_t) + 3)
        records = np.arange(360.0)[:, None, None, None] * np.ones((1, 32, 16, 32))
        model = {
            "mms_des_bgdist_p0_brst": mock.Mock(values=records),
            "mms_des_bgdist_p1_brst": mock.Mock(values=records + 1000.0),
            "mms_des_startdelphi_counts_brst": mock.Mock(
                values=16 * np.arange(360) + 8
            ),
        }
        return vdf, n_e, startdelphi, model

    def _run(self, vdf, n_e, startdelphi, model, **kwargs):
        with (
            mock.patch.object(self.module, "get_data", return_value=n_e) as get_data,
            mock.patch.object(
                self.module, "db_get_ts", return_value=startdelphi
            ) as db_get_ts,
            mock.patch.object(
                self.module, "_load_bgdist_model", return_value=model
            ) as load_model,
        ):
            result = mms.remove_edist_background(vdf, **kwargs)

        return result, (get_data, db_get_ts, load_model)

    def test_remove_edist_background_source(self):
        # source is passed to get_data, db_get_ts and the model loader
        vdf, n_e, startdelphi, model = self._inputs()
        (_, _, scale), (get_data, db_get_ts, load_model) = self._run(
            vdf, n_e, startdelphi, model, source="SDC"
        )

        self.assertEqual(get_data.call_args.kwargs["source"], "sdc")
        self.assertEqual(db_get_ts.call_args.kwargs["source"], "sdc")
        self.assertEqual(load_model.call_args.args[:2], (self.MODEL, "sdc"))
        self.assertEqual(scale, 0.5)

        with self.assertRaises(ValueError):
            mms.remove_edist_background(vdf, source="bazinga")

    def test_remove_edist_background_model(self):
        # Parity 1 samples used the parity 0 model; each sample uses the model
        # record of its spin phase, scaled by the photoelectron factor
        vdf, n_e, startdelphi, model = self._inputs()
        (vdf_new, vdf_bkg, _), _ = self._run(vdf, n_e, startdelphi, model)

        expected = 0.5 * np.array([0.0, 1001.0, 2.0, 1003.0])
        np.testing.assert_allclose(vdf_bkg.data.data[:, 0, 0, 0], expected)
        np.testing.assert_allclose(vdf_new.data.data[:, 0, 0, 0], 1e6 - expected)

    def test_remove_edist_background_spin_phase_time(self):
        # startdelphi_count is read separately: an extra sample before the
        # distributions used to shift every sample to the next spin phase
        vdf, n_e, startdelphi, model = self._inputs()
        dt = vdf.time.data[1] - vdf.time.data[0]
        extra = pyrf.ts_scalar(
            np.hstack([vdf.time.data[0] - dt, vdf.time.data]),
            np.hstack([16 * 100 + 3, startdelphi.data]),
        )
        (_, vdf_bkg, _), _ = self._run(vdf, n_e, extra, model)

        expected = 0.5 * np.array([0.0, 1001.0, 2.0, 1003.0])
        np.testing.assert_allclose(vdf_bkg.data.data[:, 0, 0, 0], expected)

    def test_remove_edist_background_n_art(self):
        # n_art = 0 removes no photoelectrons (it used to mean "default")
        vdf, n_e, startdelphi, model = self._inputs()
        (vdf_new, vdf_bkg, _), _ = self._run(vdf, n_e, startdelphi, model, n_art=0.0)

        np.testing.assert_allclose(vdf_bkg.data.data, 0.0)
        np.testing.assert_allclose(vdf_new.data.data, vdf.data.data)

        # The attributes of the outputs and of the input are independent
        (vdf_new, vdf_bkg, _), _ = self._run(vdf, n_e, startdelphi, model)
        vdf_new.attrs["test"] = 1
        self.assertNotIn("test", vdf.attrs)
        self.assertNotIn("test", vdf_bkg.attrs)


class RemoveIdistBackgroundTestCase(unittest.TestCase):
    def test_remove_idist_background(self):
        # The background (omni differential energy flux, 1/(cm^2 s sr)) converted
        # to phase-space density is removed (negative values set to 0), and the
        # caller's VDF is unchanged (it was a shallow copy)
        vdf = generate_vdf(64.0, 10, [32, 32, 16], species="ions")
        vdf.data.data[...] = 1e-23
        vdf_in = vdf.data.data.copy()
        def_bg = pyrf.ts_scalar(vdf.time.data, np.full(10, 1e3))

        result = mms.remove_idist_background(vdf, def_bg)

        with np.errstate(divide="ignore"):  # the first energy channel is 0
            coeff = constants.proton_mass / (
                constants.elementary_charge * vdf.energy.data.astype(np.float64)
            )
        psd_bg = 1e3 * 1e4 / 2 * coeff**2 / 1e12  # (time, energy)
        expected = np.clip(1e-23 - psd_bg[:, :, None, None], 0.0, None)
        expected = np.broadcast_to(expected, vdf_in.shape)

        self.assertTrue(np.any(expected > 0) and np.any(expected == 0))
        np.testing.assert_allclose(result.data.data, expected, rtol=1e-5, atol=1e-30)
        np.testing.assert_array_equal(vdf.data.data, vdf_in)


class RemoveImomsBackgroundTestCase(unittest.TestCase):
    @staticmethod
    def _measured(n_t=5):
        # Moments measured with a background at rest (n_bg = 0.5 cm^-3,
        # p_bg = 0.01 nPa) added to a known plasma (n = 0.5 cm^-3, V, P)
        n_true, v_true = 0.5, np.array([-800.0, 50.0, 20.0])
        p_true = np.array([[0.2, 0.01, 0.02], [0.01, 0.15, 0.03], [0.02, 0.03, 0.1]])
        n_bg, p_bg = 0.5, 0.01

        n_meas = n_true + n_bg
        v_meas = n_true * v_true / n_meas
        fact = constants.proton_mass * 1e21  # m n v v in nPa (cm^-3, km/s)
        p_meas = p_true + fact * n_true * np.outer(v_true, v_true)
        p_meas += p_bg * np.eye(3) - fact * n_meas * np.outer(v_meas, v_meas)

        time = generate_timeline(1.0, n_t)
        measured = (
            pyrf.ts_scalar(time, np.full(n_t, n_meas)),
            pyrf.ts_vec_xyz(time, np.tile(v_meas, (n_t, 1))),
            pyrf.ts_tensor_xyz(
                time, np.tile(p_meas, (n_t, 1, 1)), attrs={"UNITS": "nPa"}
            ),
        )
        return time, measured, (n_true, v_true, p_true), (n_bg, p_bg)

    def test_remove_imoms_background_values(self):
        # The dynamic pressure terms lacked a 1e21 unit factor, so only p_bg
        # was removed from the pressure tensor
        time, measured, expected, (n_bg, p_bg) = self._measured()
        n_i, v_i, p_i = mms.remove_imoms_background(
            *measured,
            pyrf.ts_scalar(time, np.full(len(time), n_bg)),
            pyrf.ts_scalar(time, np.full(len(time), p_bg)),
        )

        np.testing.assert_allclose(n_i.data, expected[0])
        np.testing.assert_allclose(v_i.data, np.tile(expected[1], (len(time), 1)))
        np.testing.assert_allclose(
            p_i.data, np.tile(expected[2], (len(time), 1, 1)), atol=1e-12
        )
        self.assertEqual(p_i.attrs["UNITS"], "nPa")

    def test_remove_imoms_background_time_alignment(self):
        # Background moments on another time line are resampled first (they
        # used to be subtracted sample by sample, or raise a shape error)
        time, measured, expected, (n_bg, p_bg) = self._measured(10)
        time_bg = time[0] + (time[-1] - time[0]) * np.linspace(0, 1, 4)
        _, _, p_i = mms.remove_imoms_background(
            *measured,
            pyrf.ts_scalar(time_bg, np.full(4, n_bg)),
            pyrf.ts_scalar(time_bg, np.full(4, p_bg)),
        )
        np.testing.assert_allclose(
            p_i.data, np.tile(expected[2], (10, 1, 1)), atol=1e-12
        )


@ddt
class RotateTensorTestCase(unittest.TestCase):
    @data(
        (
            generate_data(100, tensor_order=1),
            "fac",
            generate_ts(64.0, 100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            42,
            generate_ts(64.0, 100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            "bazinga!!",
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            generate_data(100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "rot",
            generate_ts(64.0, 100, tensor_order=2),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "gse",
            generate_ts(64.0, 100, tensor_order=2),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            generate_ts(64.0, 100, tensor_order=1),
            42,
        ),
    )
    @unpack
    def test_rotate_tensor_input_type(self, inp, rot_flag, vec, perp):
        with self.assertRaises(TypeError):
            mms.rotate_tensor(inp, rot_flag, vec, perp)

    @data(
        (
            generate_ts(64.0, 100, tensor_order=2),
            "bazinga!!",
            generate_ts(64.0, 100, tensor_order=1),
            "pp",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "rot",
            generate_data(100, tensor_order=2),
            "",
        ),
        (
            generate_ts(64.0, 100, tensor_order=2),
            "fac",
            generate_ts(64.0, 100, tensor_order=1),
            "bazinga!!",
        ),
    )
    @unpack
    def test_rotate_tensor_input_method(self, inp, flag, vec, perp):
        with self.assertRaises(NotImplementedError):
            mms.rotate_tensor(inp, flag, vec, perp)

    @data(
        ("fac", generate_ts(64.0, 100, tensor_order=1), "pp"),
        ("fac", generate_ts(64.0, 100, tensor_order=1), "qq"),
        ("rot", np.random.random(3), "pp"),
        ("rot", np.random.random((3, 3)), "pp"),
        ("gse", generate_defatt(64.0, 100), "pp"),
    )
    @unpack
    def test_rotate_tensor_output(self, rot_flag, vec, perp):
        result = mms.rotate_tensor(
            generate_ts(64.0, 100, tensor_order=2), rot_flag, vec, perp
        )
        self.assertIsInstance(result, xr.DataArray)

    @data("gse", "gsm")
    def test_rotate_tensor_dsl_consistent_with_dsl2gse(self, frame):
        # T = a I + v v^T in DSL must rotate to a I + v' v'^T with v' = dsl2gse(v).
        # The spin-axis angles were converted from degrees to radians twice.
        n_t = 50
        time = generate_timeline(1.0, n_t)
        defatt = xr.Dataset(
            {
                "z_ra": pyrf.ts_scalar(time, np.linspace(268.0, 272.0, n_t)),
                "z_dec": pyrf.ts_scalar(time, np.linspace(65.0, 67.0, n_t)),
            }
        )
        v_dsl = pyrf.ts_vec_xyz(time, np.random.default_rng(0).normal(size=(n_t, 3)))
        t_dsl = pyrf.ts_tensor_xyz(
            time, 2.0 * np.eye(3) + np.einsum("ti,tj->tij", v_dsl.data, v_dsl.data)
        )

        v_new = mms.dsl2gse(v_dsl, defatt)
        if frame == "gsm":
            v_new = pyrf.cotrans(v_new, "gse>gsm")

        expected = 2.0 * np.eye(3) + np.einsum("ti,tj->tij", v_new.data, v_new.data)
        result = mms.rotate_tensor(t_dsl, frame, defatt)
        np.testing.assert_allclose(result.data, expected, atol=1e-10)


@ddt
class SpectrToDatasetTestCase(unittest.TestCase):
    @data(generate_spectr(64.0, 100, 10))
    def test_spectr_to_dataset_output(self, spectr):
        result = mms.spectr_to_dataset(spectr)
        self.assertIsInstance(result, xr.Dataset)


@ddt
class Scpot2NeTestCase(unittest.TestCase):
    @data(None, generate_ts(64.0, 100, tensor_order=0))
    def test_scpot2ne_output(self, i_aspoc):
        result = mms.scpot2ne(
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=0),
            generate_ts(64.0, 100, tensor_order=2),
            i_aspoc,
        )
        self.assertIsInstance(result[0], xr.DataArray)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)


class VdfReduceDeprecationTestCase(unittest.TestCase):
    # Deprecated in 2.5.0: FutureWarning (shown to users), pointing at the caller
    def setUp(self):
        self.vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True, species="ions")

    def test_vdf_frame_transformation_deprecated(self):
        v_gse = generate_ts(64.0, 42, tensor_order=1)

        with self.assertWarnsRegex(FutureWarning, "deprecated") as context:
            mms.vdf_frame_transformation(self.vdf, v_gse)

        self.assertEqual(context.filename, __file__)

    def test_vdf_reduce_deprecated(self):
        tint = list(pyrf.datetime642iso8601(self.vdf.time.data[[0, -1]]))

        with self.assertWarnsRegex(FutureWarning, "pyrfu.mms.reduce") as context:
            mms.vdf_reduce(self.vdf, tint, "1d", [1, 0, 0], n_vpt=10)

        self.assertEqual(context.filename, __file__)


class VdfToE64TestCase(unittest.TestCase):
    def test_vdf_to_e64_delta_energy(self):
        # The input attrs are unchanged; the 64-channel widths are the same at
        # every time, the top/bottom edges from the 32-channel widths (they were
        # written into the last/first time step)
        vdf = generate_vdf(64.0, 10, [32, 32, 16], energy01=True)
        attrs = {k: np.array(v, copy=True) for k, v in vdf.attrs.items()}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # zero energy channel
            result = mms.vdf_to_e64(vdf)

        for key, value in attrs.items():
            np.testing.assert_array_equal(vdf.attrs[key], value)

        d_plus = result.attrs["delta_energy_plus"]
        d_minus = result.attrs["delta_energy_minus"]
        for d_e in [d_plus, d_minus]:
            np.testing.assert_array_equal(d_e, np.broadcast_to(d_e[0], d_e.shape))
        self.assertEqual(d_plus[0, -1], np.max(attrs["delta_energy_plus"][:, -1]))
        self.assertEqual(d_minus[0, 0], np.min(attrs["delta_energy_minus"][:, 0]))


@ddt
class VdfElimTestCase(unittest.TestCase):
    @data(
        random.randint(0, 15),
        random.randint(0, 15) + 0.4,
        [random.randint(0, 15), random.randint(16, 31)],
    )
    def test_vdf_elim_output(self, e_int):
        result = mms.vdf_elim(
            generate_vdf(64.0, 42, [32, 32, 16], energy01=True), e_int
        )
        self.assertIsInstance(result, xr.Dataset)

    def setUp(self):
        # Energies 0, ..., 31 and distinct widths for each channel
        self.vdf = generate_vdf(64.0, 10, [32, 32, 16])
        self.vdf.attrs["delta_energy_plus"] = np.tile(np.arange(32) * 0.1, (10, 1))
        self.vdf.attrs["delta_energy_minus"] = np.arange(32) * 0.2

    def test_vdf_elim_interval(self):
        result = mms.vdf_elim(self.vdf, [5.0, 10.0])

        # Channels strictly within the interval, energy widths clipped too
        np.testing.assert_array_equal(result.energy.data[0], [6.0, 7.0, 8.0, 9.0])
        np.testing.assert_array_equal(result.data.data, self.vdf.data.data[:, 6:10])
        np.testing.assert_allclose(
            result.attrs["delta_energy_plus"],
            self.vdf.attrs["delta_energy_plus"][:, 6:10],
        )
        np.testing.assert_allclose(
            result.attrs["delta_energy_minus"], [1.2, 1.4, 1.6, 1.8]
        )

        # The caller's attributes are unchanged
        self.assertEqual(self.vdf.attrs["delta_energy_plus"].shape, (10, 32))

    def test_vdf_elim_two_tables(self):
        # Union of the channels of the two tables (0, 1, ... and 0.5, 1.5, ...)
        vdf = generate_vdf(64.0, 10, [32, 32, 16], energy01=True)
        result = mms.vdf_elim(vdf, [5.0, 10.0])
        self.assertEqual(result.energy.shape, (10, 5))
        self.assertEqual(result.attrs["delta_energy_plus"].shape, (10, 5))

    def test_vdf_elim_single_energy(self):
        # One time step: the closest channel of the energy table
        result = mms.vdf_elim(self.vdf.isel(time=[0]), 5.2)
        np.testing.assert_array_equal(result.energy.data, [[5.0]])
        np.testing.assert_allclose(result.attrs["delta_energy_minus"], [1.0])

    def test_vdf_elim_empty(self):
        with self.assertRaisesRegex(ValueError, "No energy channel"):
            mms.vdf_elim(self.vdf, [100.0, 200.0])


@ddt
class VdfOmniTestCase(unittest.TestCase):
    @data("mean", "sum")
    def test_vdf_omni_output(self, method):
        result = mms.vdf_omni(generate_vdf(64.0, 100, (32, 32, 16)), method)
        self.assertIsInstance(result, xr.DataArray)


@ddt
class TokenizeTestCase(unittest.TestCase):
    @data(*random.choices(_mms_keys(), k=10))
    def test_tokenize(self, var_str):
        result = mms.tokenize(var_str)
        self.assertIsInstance(result, dict)

    def test_tokenize_values(self):
        result = mms.tokenize("b_gse_fgm_brst_l2")
        self.assertEqual(result["inst"], "fgm")
        self.assertEqual(result["cdf_name"], "fgm_b_gse_brst_l2")

    @data(
        "tsi_fpi_brst_l2",
        "qe_gse_fpi_brst_l2",
        "phplus_dbcs_hpca_brst_l2",
        "e_dsl_edi_srvy_l2",
        "energye_fpi_fast_ql",
    )
    def test_tokenize_unsupported(self, var_str):
        # Valid parts, but not in mms_keys.json: a clear ValueError, not KeyError
        with self.assertRaisesRegex(ValueError, "not supported"):
            mms.tokenize(var_str)


@ddt
class ListFilesTestCase(unittest.TestCase):
    @data(*random.choices(_mms_keys(), k=10))
    def test_list_files(self, var_str):
        # Empty data directory, independent of the user's configuration
        with tempfile.TemporaryDirectory() as root:
            result = mms.list_files(
                TEST_TINT, random.randint(1, 4), mms.tokenize(var_str), root
            )

        self.assertListEqual(result, [])

    @staticmethod
    def _touch(root, rel_dir, names):
        path = os.path.join(root, *rel_dir.split("/"))
        os.makedirs(path, exist_ok=True)
        for name in names:
            with open(os.path.join(path, name), "wb"):
                pass

    @data("2019-09-14T07:54:00", "2019-09-14T00:00:00")
    def test_list_files_latest_version_srvy(self, t_start):
        # One file per day, several versions of one of them: only the latest
        # (the old code kept the last listed one, or all from midnight)
        with tempfile.TemporaryDirectory() as root:
            self._touch(
                root,
                "mms1/fgm/srvy/l2/2019/09",
                [
                    "mms1_fgm_srvy_l2_20190913_v5.211.0.cdf",
                    "mms1_fgm_srvy_l2_20190914_v5.86.0.cdf",
                    "mms1_fgm_srvy_l2_20190914_v5.211.0.cdf",
                    "mms1_fgm_srvy_l2_20190914_v5.9.0.cdf",
                    "mms1_fgm_srvy_l2_20190915_v5.211.0.cdf",
                ],
            )
            var = {"inst": "fgm", "tmmode": "srvy", "lev": "l2", "dtype": ""}
            tint = [t_start, "2019-09-14T08:11:00"]
            result = mms.list_files(tint, 1, var, root)

        self.assertListEqual(
            [os.path.basename(f) for f in result],
            ["mms1_fgm_srvy_l2_20190914_v5.211.0.cdf"],
        )

    def test_list_files_brst_segments(self):
        # Segments starting in the interval and the one covering its start,
        # latest versions only, in time order
        with tempfile.TemporaryDirectory() as root:
            self._touch(
                root,
                "mms1/fpi/brst/l2/des-moms/2019/09/14",
                [
                    "mms1_fpi_brst_l2_des-moms_20190914074923_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914075223_v3.3.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914075223_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914075523_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914080023_v3.4.0.cdf",
                    "mms1_fpi_brst_l2_des-moms_20190914081223_v3.4.0.cdf",
                ],
            )
            var = {"inst": "fpi", "tmmode": "brst", "lev": "l2", "dtype": "des-moms"}
            tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]
            result = mms.list_files(tint, 1, var, root)

        self.assertListEqual(
            [os.path.basename(f) for f in result],
            [
                "mms1_fpi_brst_l2_des-moms_20190914075223_v3.4.0.cdf",
                "mms1_fpi_brst_l2_des-moms_20190914075523_v3.4.0.cdf",
                "mms1_fpi_brst_l2_des-moms_20190914080023_v3.4.0.cdf",
            ],
        )


@ddt
class ListFilesSdcTestCase(unittest.TestCase):
    @data(*random.choices(_mms_keys(), k=10))
    def test_list_files_sdc(self, var_str):
        try:
            mms.list_files_sdc(TEST_TINT, random.randint(1, 4), mms.tokenize(var_str))
        except requests.exceptions.ReadTimeout:
            pass


# DEFEPH layout (MMS1_DEFEPH_2019257_2019258.V01): 14 header lines, no footer
_DEFEPH_HEADER = [
    "Definitive Orbit Ephemeris",
    "Spacecraft = MMS1",
    "StartTime = 2019-257/00:05:52.907 UTC  (MMS TAI: 22536.004512812)",
    "StopTime = 2019-258/00:14:22.907 UTC  (MMS TAI: 22537.010415590)",
    "MMS TAI Reference Epoch = 1958-001/00:00:00 UTC",
    "CentralBody = Earth",
    "ReferenceFrame = Mean of J2000",
    "PrincipalPlane = Equatorial",
    "Project = DefEphemReFormat.m",
    "Source = FDGSS OPS",
    "FileCreationDate = Sep 16 2019 09:45:07.912 UTC",
    "",
    "Epoch (UTC)   Epoch MMS TAI   X   Y   Z   VX   VY   VZ   Mass",
    "                              Km  Km  Km  Km/Sec  Km/Sec  Km/Sec  Kg",
]

_DEFEPH_ROWS = [
    "2019-257/00:05:52.907   22536.004512812   -10539.488308885   2267.641774990"
    "   2762.979162558   -3.379702093   -7.150985755   2.233656130",
    "2019-257/00:06:22.907   22536.004860035   -10639.511569375   2052.828069429"
    "   2829.628149301   -3.288657503   -7.169561589   2.209589514",
    "2019-257/00:06:52.907   22536.005207257   -10736.816612409   1837.489793072"
    "   2895.553467478   -3.198497883   -7.185972813   2.185417405",
]

# DEFATT layout (MMS1_DEFATT_2019257_2019258.V00): 49 header lines (up to the
# COMMENT line with the column names) and a DATA_STOP footer
_DEFATT_ROWS = [
    "2019-257T00:00:00.012 1947110437.012  0.09800 -0.18260  0.80379  0.55765 "
    "-0.026  0.026 18.302 300.435 263.470  66.080 300.430 263.384  65.97 "
    "300.435 263.384 65.970 300.435 0.005  EKF",
    "2019-257T00:00:00.562 1947110437.562  0.08149 -0.19048  0.84953  0.48516 "
    "-0.025  0.027 18.301 310.481 263.431  66.086 310.476 263.386  65.97 "
    "310.481 263.386 65.970 310.481 0.005  EKF",
]


class LoadAncillaryTestCase(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = root.name
        self.tint = ["2019-09-14T00:00:00", "2019-09-14T01:00:00"]

    def _write(self, product, file_name, lines):
        path = os.path.join(self.root, "ancillary", "mms1", product)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, file_name), "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    def test_load_ancillary_defeph(self):
        # Without footer, the last sample is data (it was dropped)
        self._write(
            "defeph",
            "MMS1_DEFEPH_2019257_2019258.V01",
            _DEFEPH_HEADER + _DEFEPH_ROWS,
        )
        result = mms.load_ancillary("defeph", self.tint, 1, False, self.root)

        self.assertEqual(len(result.time), 3)
        self.assertEqual(
            result.time.data[-1], np.datetime64("2019-09-14T00:06:52.907", "ns")
        )
        np.testing.assert_allclose(
            result.x.data, [-10539.488308885, -10639.511569375, -10736.816612409]
        )

    def test_load_ancillary_defatt(self):
        # The DATA_STOP footer is not data
        header = ["META_START"] * 48 + ["COMMENT   Time (UTC)    Elapsed Sec"]
        self._write(
            "defatt",
            "MMS1_DEFATT_2019257_2019258.V00",
            header + _DEFATT_ROWS + ["DATA_STOP"],
        )
        result = mms.load_ancillary("defatt", self.tint, 1, False, self.root)

        self.assertEqual(len(result.time), 2)
        np.testing.assert_allclose(result.z_dec.data, [66.080, 66.086])

    def test_load_ancillary_no_file(self):
        with self.assertRaises(FileNotFoundError):
            mms.load_ancillary("defeph", self.tint, 1, False, self.root)

    def test_load_ancillary_unsupported(self):
        for product in ["predatt", "predeph", "bazinga"]:
            with self.assertRaisesRegex(ValueError, "not supported"):
                mms.load_ancillary(product, self.tint, 1, False, self.root)


@ddt
class ListFilesAncillaryTestCase(unittest.TestCase):
    @data("predatt", "predeph", "defatt", "defeph")
    def test_list_files_ancillary(self, product):
        # Empty data directory, independent of the user's configuration
        with tempfile.TemporaryDirectory() as root:
            result = mms.list_files_ancillary(
                TEST_TINT, random.randint(1, 4), product, root
            )

        self.assertListEqual(result, [])


@ddt
class ListFilesAncillarySdcTestCase(unittest.TestCase):
    @data("predatt", "predeph", "defatt", "defeph")
    def test_list_files_ancillary_sdc(self, product):
        try:
            mms.list_files_ancillary_sdc(TEST_TINT, random.randint(1, 4), product)
        except requests.exceptions.ReadTimeout:
            pass


@ddt
class VdfProjectionTestCase(unittest.TestCase):
    def test_vdf_projection_output(self):
        result = mms.vdf_projection(
            generate_vdf(64.0, 42, [32, 32, 16], energy01=True), TEST_TINT
        )
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)

    @staticmethod
    def _vdf():
        # Alternating energy tables (positive energies) and a phi table that
        # changes with time, so that each sample's phi is identifiable
        vdf = generate_vdf(64.0, 42, [32, 32, 16], energy01=True)
        energy0 = 10.0 * 1.3 ** np.arange(32)
        energy1 = energy0 * np.sqrt(1.3)
        vdf.attrs["energy0"], vdf.attrs["energy1"] = energy0, energy1
        vdf["energy"] = (
            ("time", "idx0"),
            np.where(vdf.attrs["esteptable"][:, None] == 1, energy1, energy0),
        )
        vdf["phi"] = (
            ("time", "idx1"),
            (np.arange(32) * 11.25)[None, :] + 0.1 * np.arange(42)[:, None],
        )
        return vdf

    def test_vdf_projection_phi_clipped(self):
        # With a tint that doesn't start at the first sample, psd_rebin used to
        # get the phi of the first samples instead of those in tint
        vdf = self._vdf()
        tint = list(pyrf.datetime642iso8601(vdf.time.data[[10, 21]]))
        module = importlib.import_module("pyrfu.mms.vdf_projection")
        with mock.patch.object(
            module, "psd_rebin", wraps=module.psd_rebin
        ) as psd_rebin:
            module._init(vdf, tint)

        expected = pyrf.time_clip(vdf.phi, tint).data
        np.testing.assert_allclose(psd_rebin.call_args.args[1], expected)

    @data(0, 1)
    def test_vdf_projection_single_time_energy_table(self, step):
        # A single time used the energy1 edges for both step table values
        vdf = self._vdf()
        t_id = int(np.flatnonzero(vdf.attrs["esteptable"] == step)[0])
        tint = [pyrf.datetime642iso8601(vdf.time.data[t_id])]
        energy_edges = importlib.import_module("pyrfu.mms.vdf_projection")._init(
            vdf, tint
        )[3]

        energy = vdf.attrs[f"energy{step}"]
        # Log-centred edges: the geometric mean of consecutive edges is the energy
        np.testing.assert_allclose(
            np.sqrt(energy_edges[:-1] * energy_edges[1:]), energy, rtol=1e-6
        )

    @staticmethod
    def _maxwellian(v_kms, n_t=2, alternating=False):
        # 100 eV proton Maxwellian on FPI-like bins
        energy1 = None
        if alternating:
            energy1 = 10.0 * 3000.0 ** ((np.arange(32) + 0.5) / 31)

        vdf, _ = PsdMomentsTestCase._drifting_maxwellian(
            1.0, 100.0, v_kms, n_t=n_t, energy1=energy1
        )
        vdf.data.attrs["UNITS"] = "s^3/cm^6"
        return vdf

    @staticmethod
    def _peak(v_x, v_y, f_mat):
        # Speed (km/s) and angle (deg) of the centre of the maximum bin
        v_x_c = (v_x[:-1, :-1] + v_x[1:, :-1] + v_x[:-1, 1:] + v_x[1:, 1:]) / 4
        v_y_c = (v_y[:-1, :-1] + v_y[1:, :-1] + v_y[:-1, 1:] + v_y[1:, 1:]) / 4
        idx = np.unravel_index(np.nanargmax(f_mat.T), f_mat.T.shape)
        speed = np.hypot(v_x_c[idx], v_y_c[idx])
        return speed, np.rad2deg(np.arctan2(v_y_c[idx], v_x_c[idx]))

    @data(
        # (in-plane angle of the drift, frame with x, y, z as rows)
        (5.625, np.eye(3)),
        (5.625, np.vstack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
        (95.625, np.vstack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
    )
    @unpack
    def test_vdf_projection_peak(self, angle, coord_sys):
        # 400 km/s drift in the (x, y) plane of coord_sys, at an azimuthal bin
        # centre: the projection peaks there. An orthonormal frame gives no
        # warning (the check compared rows to columns and always warned).
        coord_sys = coord_sys.astype(float)
        rad = np.deg2rad(angle)
        v_d = 400.0 * (np.cos(rad) * coord_sys[0] + np.sin(rad) * coord_sys[1])
        vdf = self._maxwellian(list(v_d))
        tint = [pyrf.datetime642iso8601(vdf.time.data[0])]

        with self.assertNoLogs(level="WARNING"):
            result = mms.vdf_projection(vdf, tint, coord_sys)

        speed, peak_angle = self._peak(*result)
        self.assertAlmostEqual(peak_angle, angle, delta=0.1)
        self.assertAlmostEqual(speed, 400.0, delta=30.0)

    @data(False, True)
    def test_vdf_projection_interval(self, alternating):
        # Intervals with one energy table used to crash (.data on an array);
        # with alternating tables (rebinned to 64 energies) phi was used in
        # degrees as radians, so the peak was misplaced
        rot = np.deg2rad(5.625)
        vdf = self._maxwellian(
            [400.0 * np.cos(rot), 400.0 * np.sin(rot), 0.0], 20, alternating
        )
        tint = list(pyrf.datetime642iso8601(vdf.time.data[[0, -1]]))
        v_x, v_y, f_mat = mms.vdf_projection(vdf, tint)

        self.assertEqual(f_mat.shape[1], 64 if alternating else 32)
        speed, peak_angle = self._peak(v_x, v_y, f_mat)
        self.assertAlmostEqual(peak_angle, 5.625, delta=0.1)
        self.assertAlmostEqual(speed, 400.0, delta=30.0)

    def test_vdf_projection_sc_pot(self):
        # Energy edges below the spacecraft potential gave NaN speeds
        vdf = self._maxwellian([400.0, 0.0, 0.0])
        vdf.attrs["species"] = "electrons"
        sc_pot = pyrf.ts_scalar(vdf.time.data, np.full(2, 20.0))
        tint = [pyrf.datetime642iso8601(vdf.time.data[0])]
        v_x, v_y, _ = mms.vdf_projection(vdf, tint, sc_pot=sc_pot)

        self.assertTrue(np.isfinite(v_x).all() and np.isfinite(v_y).all())
        self.assertEqual(np.hypot(v_x, v_y)[0].max(), 0.0)


@ddt
class FftBandpassTestCase(unittest.TestCase):
    @data(
        (generate_data(100, tensor_order=0), 0.1, 1.0),
        (generate_ts(64.0, 100, tensor_order=0), "bazinga!!", 1.0),
        (generate_ts(64.0, 100, tensor_order=0), 0.1, "bazinga!!"),
    )
    @unpack
    def test_fft_bandpass_input_type(self, inp, f_min, f_max):
        with self.assertRaises(TypeError):
            mms.fft_bandpass(inp, f_min, f_max)

    @data(
        (generate_ts(64.0, 100, tensor_order=0), 1.0, 0.1),
        (generate_ts(64.0, 100, tensor_order=2), 0.1, 1.0),
    )
    @unpack
    def test_fft_bandpass_input_value(self, inp, f_min, f_max):
        with self.assertRaises(ValueError):
            mms.fft_bandpass(inp, f_min, f_max)

    @data(
        (generate_ts(64.0, 101, tensor_order=0), random.random(), random.random()),
        (generate_ts(64.0, 101, tensor_order=1), random.random(), random.random()),
    )
    @unpack
    def test_fft_bandpass_output(self, inp, f_min, f_max):
        result = mms.fft_bandpass(inp, *sorted([f_min, f_max]))
        self.assertIsInstance(result, xr.DataArray)

    @data(0, 1)
    def test_fft_bandpass_nan_input_unchanged(self, tensor_order):
        # NaNs are set to 0 for the FFT in a copy: the caller's data keep them, and
        # the output is NaN there. 5 and 20 Hz sines, band 10-30 Hz.
        times = generate_timeline(128.0, 1024)
        t_s = np.arange(1024) / 128.0
        sig = np.sin(2 * np.pi * 5 * t_s) + np.sin(2 * np.pi * 20 * t_s)
        if tensor_order:
            inp = pyrf.ts_vec_xyz(times, np.tile(sig[:, None], (1, 3)))
        else:
            inp = pyrf.ts_scalar(times, sig)
        inp.data[100] = np.nan
        inp_data = inp.data.copy()

        out = mms.fft_bandpass(inp, 10.0, 30.0)

        np.testing.assert_array_equal(inp.data, inp_data)
        self.assertTrue(np.all(np.isnan(out.data[100])))
        expected = np.sin(2 * np.pi * 20 * t_s)
        if tensor_order:
            expected = np.tile(expected[:, None], (1, 3))
        np.testing.assert_allclose(out.data[300:700], expected[300:700], atol=0.05)


class WhistlerB2ETestCase(unittest.TestCase):
    b_mag, n_e = 50.0, 5.0  # nT, cm^-3
    f_ce = constants.e * 1e-9 * b_mag / constants.m_e / (2 * np.pi)
    f_pe = np.sqrt(1e6 * n_e * constants.e**2 / (constants.m_e * constants.epsilon_0))
    f_pe /= 2 * np.pi

    def test_whistler_b2e_parallel(self):
        # theta_k = 0: n^2 = R and E^2 = (c^2 / R) B^2, with 1e-12 from nT^2 to
        # T^2 (1e-18) and from (V/m)^2 to (mV/m)^2 (1e6)
        freq = np.array([0.1, 0.3, 0.5]) * self.f_ce
        b2 = np.array([1.0, 2.0, 3.0])
        result = mms.whistler_b2e(b2, freq, 0.0, self.b_mag, self.n_e)

        r_cold = 1 - self.f_pe**2 / (freq * (freq - self.f_ce))
        expected = constants.c**2 / r_cold * b2 * 1e-12
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_whistler_b2e_spectrogram(self):
        # A spectrogram gives, at each time, the spectrum for the local B and n
        time = generate_timeline(1.0, 3)
        freq = np.array([0.1, 0.3, 0.5]) * self.f_ce
        b2_data = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [1.0, 1.0, 1.0]])
        b2 = xr.DataArray(b2_data, coords=[time, freq], dims=["time", "frequency"])
        b_mag = pyrf.ts_scalar(time, np.array([50.0, 60.0, 70.0]))
        n_e = pyrf.ts_scalar(time, np.array([5.0, 4.0, 6.0]))
        theta_k = np.deg2rad(30.0)

        result = mms.whistler_b2e(b2, freq, theta_k, b_mag, n_e)

        self.assertIsInstance(result, xr.DataArray)
        self.assertTupleEqual(result.dims, ("time", "frequency"))
        for i in range(3):
            expected = mms.whistler_b2e(
                b2_data[i], freq, theta_k, b_mag.data[i], n_e.data[i]
            )
            np.testing.assert_allclose(result.data[i], expected, rtol=1e-12)

    def test_whistler_b2e_length(self):
        with self.assertRaises(IndexError):
            mms.whistler_b2e(np.ones(3), np.ones(4), 0.0, 50.0, 5.0)


if __name__ == "__main__":
    unittest.main()
