#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools
import os
import random
import subprocess
import sys
import unittest
from unittest import mock

import matplotlib as mpl
import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np
import xarray as xr
from ddt import data, ddt, unpack
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.contour import ContourSet
from matplotlib.dates import date2num
from matplotlib.image import AxesImage
from scipy import constants

# Local imports
from .. import plot, pyrf
from ..constants import R_E
from ..models.ion_anisotropy_thresh import ion_anisotropy_thresh
from . import generate_data, generate_timeline, generate_ts, generate_vdf


def _close_new_figures(test_case):
    r"""Close the figures created during the test, and only those (not the
    figures the caller may have open)."""
    before = set(plt.get_fignums())

    def close():
        for num in set(plt.get_fignums()) - before:
            plt.close(num)

    test_case.addCleanup(close)


# Axes named in @data, created when the test runs: the decorators run at
# import, where plt.subplots opened 20 figures
_AXES = {
    "axis": lambda: plt.subplots(1)[1],
    "axes": lambda: plt.subplots(3)[1],
    "axes[0]": lambda: plt.subplots(3)[1][0],
}


def _axis(spec):
    r"""Axis named by `spec` in _AXES, or `spec` itself (None, a number)."""
    return _AXES[spec]() if isinstance(spec, str) else spec


@ddt
class PlotLineTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    @data((0.0, generate_ts(64.0, 100)), ("axes", generate_ts(64.0, 100)))
    @unpack
    def test_plot_line_axis_type(self, axis, inp):
        with self.assertRaises(TypeError):
            plot.plot_line(_axis(axis), inp)

    @data(("axis", generate_data(100)))
    @unpack
    def test_plot_line_inp_type(self, axis, inp):
        with self.assertRaises(TypeError):
            plot.plot_line(_axis(axis), inp)

    @data(("axis", generate_ts(64.0, 100, tensor_order=random.randint(3, 10))))
    @unpack
    def test_plot_line_inp_shape(self, axis, inp):
        with self.assertRaises(NotImplementedError):
            plot.plot_line(_axis(axis), inp)

    @data(
        (None, generate_ts(64.0, 100, tensor_order=random.randint(0, 2))),
        ("axis", generate_ts(64.0, 100, tensor_order=random.randint(0, 2))),
        ("axes[0]", generate_ts(64.0, 100, tensor_order=random.randint(0, 2))),
    )
    @unpack
    def test_plot_line_output(self, axis, inp):
        result = plot.plot_line(_axis(axis), inp)
        self.assertIsInstance(result, Axes)


@ddt
class PlotClinesTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    @data("jet", plt.get_cmap("viridis"))
    def test_plot_clines_output(self, cmap):
        # get_cmap(name=cmap) raised a TypeError on every call
        energy = np.array([10.0, 30.0, 100.0, 1000.0, 3000.0])
        inp = xr.DataArray(
            np.random.rand(100, len(energy)),
            coords=[generate_ts(64.0, 100).time.data, energy],
            dims=["time", "energy"],
        )
        _, axis = plt.subplots(1)
        result = plot.plot_clines(axis, inp, cmap=cmap)
        self.assertIs(result[0], axis)
        self.assertIsInstance(result[1], Axes)
        self.assertEqual(len(axis.lines), len(energy))
        self.assertEqual(axis.get_yscale(), "log")

        # The colors follow the energies on the log colorbar (not the index)
        c_map = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        expected = c_map(LogNorm(vmin=10.0, vmax=3000.0)(energy))
        colors = [to_rgba(line.get_color()) for line in axis.lines]
        np.testing.assert_allclose(colors, expected)

    def test_plot_clines_cscale(self):
        inp = xr.DataArray(
            np.random.rand(10, 3),
            coords=[generate_ts(64.0, 10).time.data, [1.0, 10.0, 100.0]],
            dims=["time", "energy"],
        )
        with self.assertRaises(NotImplementedError):
            plot.plot_clines(plt.subplots(1)[1], inp, cscale="lin")

    def test_plot_clines_other_figure(self):
        # The colour bar goes in the figure of the axis, not the current figure
        inp = xr.DataArray(
            np.random.rand(10, 3),
            coords=[generate_ts(64.0, 10).time.data, [1.0, 10.0, 100.0]],
            dims=["time", "energy"],
        )
        f, ax = plt.subplots(1)
        f_current = plt.figure()
        _, cax = plot.plot_clines(ax, inp)

        self.assertIn(cax, f.axes)
        self.assertEqual(len(f_current.axes), 0)


@ddt
class AddPositionTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    @data(generate_ts(64.0, 100, tensor_order=1))
    def test_add_position_output(self, value):
        result = plot.add_position(plt.subplots(1)[1], value)
        self.assertIsInstance(result, Axes)

    def test_add_position_values(self):
        # Position every minute, x = seconds since the first sample
        t_0 = np.datetime64("2019-09-14T07:54:00", "ns")
        time = t_0 + np.arange(10) * np.timedelta64(60, "s")
        x_pos = np.arange(10) * 60.0
        r_xyz = pyrf.ts_vec_xyz(time, np.stack([x_pos, 2 * x_pos, -x_pos], axis=1))

        # Ticks between the samples, the last two after the time series
        t_ticks = t_0 + np.arange(15, 660, 60) * np.timedelta64(1, "s")
        _, ax = plt.subplots(1)
        ax.plot(time, x_pos)
        ax.set_xticks(date2num(t_ticks))
        ax.set_xlim(date2num(t_ticks[[0, -1]]))

        result = plot.add_position(ax, r_xyz, units="km")
        labels = [label.get_text() for label in result.get_xticklabels()]
        self.assertEqual(labels[0], "15.00\n30.00\n-15.00\n36.74")
        self.assertEqual(labels[1], "75.00\n150.00\n-75.00\n183.71")
        self.assertEqual(labels[8], "495.00\n990.00\n-495.00\n1212.50")
        self.assertListEqual(labels[9:], ["", ""])
        texts = [text_.get_text() for text_ in result.texts]
        self.assertListEqual(texts, ["X [km]\nY [km]\nZ [km]\nR [km]"])

    @data("top", "bottom")
    def test_add_position_earth_radii(self, position):
        # Position at (3, 4, 12) R_E, |R| = 13 R_E
        time = np.datetime64("2019-09-14T07:54:00", "ns")
        time = time + np.arange(10) * np.timedelta64(60, "s")
        r_xyz = pyrf.ts_vec_xyz(time, np.tile([3.0, 4.0, 12.0], (10, 1)) * R_E)

        _, ax = plt.subplots(1)
        ax.plot(time, np.arange(10))
        ax.set_xticks(date2num(time[2:8]))
        result = plot.add_position(ax, r_xyz, position=position)
        labels = [label.get_text() for label in result.get_xticklabels()]
        self.assertListEqual(labels, ["3.00\n4.00\n12.00\n13.00"] * 6)
        texts = [text_.get_text() for text_ in result.texts]
        self.assertListEqual(texts, ["\n".join(f"{c} [$R_E$]" for c in "XYZR")])

    def test_add_position_units(self):
        with self.assertRaises(ValueError):
            plot.add_position(
                plt.subplots(1)[1], generate_ts(64.0, 100, tensor_order=1), units="m"
            )


@ddt
class PlTxTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    @data(
        ([generate_ts(64.0, 100, tensor_order=3) for _ in range(4)], "cluster"),
        ([generate_ts(64.0, 100, tensor_order=0) for _ in range(4)], "bazinga"),
    )
    @unpack
    def test_pl_tx_input(self, value, colors):
        with self.assertRaises(NotImplementedError):
            plot.pl_tx(plt.subplots(1)[1], value, colors=colors)

    @data(
        (
            None,
            [
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
            ],
        ),
        (
            "axis",
            [
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
            ],
        ),
        (
            "axis",
            [
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
            ],
        ),
        (
            "axis",
            [
                generate_ts(64.0, 100, tensor_order=2),
                generate_ts(64.0, 100, tensor_order=2),
                generate_ts(64.0, 100, tensor_order=2),
                generate_ts(64.0, 100, tensor_order=2),
            ],
        ),
    )
    @unpack
    def test_pl_tx_output(self, ax, value):
        result = plot.pl_tx(_axis(ax), value, 0)
        self.assertIsInstance(result, Axes)


@ddt
class ZoomTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    @data((None, "axis"), ("axis", None))
    @unpack
    def test_zoom_input(self, ax1, ax2):
        with self.assertRaises(TypeError):
            plot.zoom(_axis(ax1), _axis(ax2))

    def test_zoom_output(self):
        _, (ax1, ax2) = plt.subplots(2)
        result = plot.zoom(ax1, ax2)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], Axes)
        self.assertIsInstance(result[1], Axes)


@ddt
class SetColorCycleTestCase(unittest.TestCase):
    @data(
        ("pyrfu", "Spectral_r"),
        ("Oceanic", "ocean"),
        ("tab", "Spectral_r"),
        ("Tableau", "Spectral_r"),
        ("mpl", "Spectral_r"),
        ("", "jet"),
    )
    @unpack
    def test_set_color_cycle_input(self, value, cmap):
        with mpl.rc_context():
            colors, result = plot.set_color_cycle(value)

            self.assertEqual(result, cmap)
            self.assertEqual(mpl.rcParams["image.cmap"], cmap)
            self.assertListEqual(
                mpl.rcParams["axes.prop_cycle"].by_key()["color"], colors
            )


class UsePyrfuStyleTestCase(unittest.TestCase):
    # The style of the example notebooks
    STYLE = {
        "image.cmap": "Spectral_r",
        "legend.loc": "upper right",
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "mathtext.fontset": "dejavusans",
        "text.usetex": False,
    }
    COLORS = ["#005f87", "#008700", "#af0000", "#444444", "#d75f00"]

    @staticmethod
    def _which(missing=()):
        return lambda cmd: None if cmd in missing else f"/usr/bin/{cmd}"

    def _assert_pyrfu_style(self):
        for key, value in self.STYLE.items():
            self.assertEqual(mpl.rcParams[key], value, key)

        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        self.assertListEqual(
            [mpl.colors.to_hex(c) for c in colors[: len(self.COLORS)]], self.COLORS
        )

    def test_use_pyrfu_style_default(self):
        with mpl.rc_context():
            # The pyrfu sheet is applied on matplotlib's defaults
            mpl.rcParams["lines.linewidth"] = 5.0
            plot.use_pyrfu_style()
            self._assert_pyrfu_style()
            self.assertEqual(mpl.rcParams["lines.linewidth"], 1.5)

    def test_use_pyrfu_style_sheet(self):
        # plt.style.use("pyrfu") gives the same style
        with mpl.rc_context():
            plt.style.use(["default", "pyrfu"])
            self._assert_pyrfu_style()

    def test_use_pyrfu_style_other_sheet(self):
        with mpl.rc_context():
            plot.use_pyrfu_style("classic", color_cycle="tab")
            self.assertEqual(mpl.rcParams["image.cmap"], "Spectral_r")
            self.assertTrue(mpl.rcParams["xtick.minor.visible"])

    def test_use_pyrfu_style_usetex(self):
        with mpl.rc_context(), mock.patch("pyrfu.plot.shutil.which", self._which()):
            plot.use_pyrfu_style(usetex=True)
            self.assertTrue(mpl.rcParams["text.usetex"])
            self.assertIn(r"\usepackage{amsmath}", mpl.rcParams["text.latex.preamble"])

    def test_use_pyrfu_style_no_usetex(self):
        with mpl.rc_context():
            mpl.rcParams["text.usetex"] = True
            plot.use_pyrfu_style(usetex=False)
            self.assertFalse(mpl.rcParams["text.usetex"])

    def test_use_pyrfu_style_usetex_fallback(self):
        # matplotlib runs latex (not pdflatex), dvipng and gs
        for missing in ["latex", "dvipng", "gs"]:
            with (
                mpl.rc_context(),
                mock.patch("pyrfu.plot.shutil.which", self._which([missing])),
            ):
                with self.assertWarns(UserWarning):
                    plot.use_pyrfu_style(usetex=True)

                self.assertFalse(mpl.rcParams["text.usetex"])


class PlotAngAngTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        # Energies 0, 1, ..., 31 and data (E + 1) (phi + 1) (theta + 1), constant
        # in time, so the averaged map is <E + 1> (phi + 1) (theta + 1)
        self.vdf = generate_vdf(64.0, 10, [32, 32, 16], units="s^3/m^6")
        e_idx, p_idx, t_idx = np.meshgrid(
            np.arange(32), np.arange(32), np.arange(16), indexing="ij"
        )
        self.vdf.data.data[:] = (e_idx + 1.0) * (p_idx + 1.0) * (t_idx + 1.0)
        self.angles = np.outer(np.arange(1.0, 33.0), np.arange(1.0, 17.0))
        self.tint = list(pyrf.datetime642iso8601(self.vdf.time.data[[2, 7]]))

    @staticmethod
    def _map(ax):
        # Plotted (phi, theta) map
        return np.asarray(ax.collections[0].get_array()).reshape(16, 32).T

    def test_plot_ang_ang_tint(self):
        for tint in [self.tint, self.tint[:1]]:
            f, ax, cax = plot.plot_ang_ang(self.vdf, tint, [5.0, 10.0])

            self.assertIsInstance(f, plt.Figure)
            self.assertIsInstance(cax, Axes)
            np.testing.assert_allclose(self._map(ax), 8.5 * self.angles)
            self.assertEqual(cax.get_ylabel(), "PSD [s$^3$ m$^{-6}$]")
            self.assertEqual(ax.get_title(), "5 keV $\\leq E \\leq$ 10 keV")

    def test_plot_ang_ang_defaults(self):
        # Whole interval and all the energies, including the end channels
        with self.assertWarns(UserWarning):
            _, ax, _ = plot.plot_ang_ang(self.vdf)

        np.testing.assert_allclose(self._map(ax), 16.5 * self.angles)
        self.assertEqual(ax.get_title(), "0 keV $\\leq E \\leq$ 31 keV")

    def test_plot_ang_ang_en_range(self):
        # Clamped to the instrument range, without changing the caller's list
        en_range = [-10.0, 1e3]
        _, ax, _ = plot.plot_ang_ang(self.vdf, self.tint, en_range)

        np.testing.assert_allclose(self._map(ax), 16.5 * self.angles)
        self.assertListEqual(en_range, [-10.0, 1e3])

        with self.assertRaises(ValueError):
            plot.plot_ang_ang(self.vdf, self.tint, [100.0, 200.0])

    def test_plot_ang_ang_phi_1d(self):
        # phi as (phi,) (ValueError): the same map
        vdf = self.vdf.copy()
        vdf["phi"] = ("idx1", self.vdf.phi.data[0])

        for tint in [self.tint, self.tint[:1]]:
            _, ax, _ = plot.plot_ang_ang(vdf, tint, [5.0, 10.0])
            _, expected, _ = plot.plot_ang_ang(self.vdf, tint, [5.0, 10.0])

            np.testing.assert_allclose(self._map(ax), 8.5 * self.angles)
            np.testing.assert_allclose(
                ax.collections[0].get_coordinates(),
                expected.collections[0].get_coordinates(),
            )

    def test_plot_ang_ang_tint_type(self):
        with self.assertRaises(TypeError):
            plot.plot_ang_ang(self.vdf, self.tint + self.tint[:1])

    def test_plot_ang_ang_units(self):
        for units, label in [
            ("s^3/cm^6", "PSD [s$^3$ cm$^{-6}$]"),
            ("1/(cm^2 s sr keV)", "Intensity [(cm$^2$ s sr keV)$^{-1}$]"),
            ("keV/(cm^2 s sr keV)", "DEF [keV (cm$^2$ s sr keV)$^{-1}$]"),
            ("counts", "counts"),
        ]:
            self.vdf.data.attrs["UNITS"] = units
            _, _, cax = plot.plot_ang_ang(self.vdf, self.tint, [5.0, 10.0])
            self.assertEqual(cax.get_ylabel(), label)


class PlotPadPolarTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        # 8 contiguous energy channels from 10 eV to 10 keV, 6 pitch-angle bins
        # of 30 degrees, and data (t + 1) (E + 1) (alpha + 1) at 4 times
        edges = np.logspace(1, 4, 9)
        self.energy_edges = edges
        energy = np.sqrt(edges[:-1] * edges[1:])
        time = pyrf.iso86012datetime64(
            np.array([f"2017-07-11T22:34:0{i}.000" for i in range(4)])
        )
        t_idx, e_idx, a_idx = np.meshgrid(
            np.arange(4), np.arange(8), np.arange(6), indexing="ij"
        )
        self.pad = xr.Dataset(
            {
                "data": (
                    ["time", "idx0", "idx1"],
                    (t_idx + 1.0) * (e_idx + 1.0) * (a_idx + 1.0),
                    {"UNITS": "s^3/m^6"},
                ),
                "energy": (["time", "idx0"], np.tile(energy, (4, 1))),
                "theta": (
                    ["time", "idx1"],
                    np.tile(np.arange(15.0, 180.0, 30.0), (4, 1)),
                ),
            },
            coords={"time": time, "idx0": np.arange(8), "idx1": np.arange(6)},
            attrs={
                "species": "ions",
                "delta_energy_minus": np.tile(energy - edges[:-1], (4, 1)),
                "delta_energy_plus": np.tile(edges[1:] - energy, (4, 1)),
                "delta_pitchangle_minus": np.full(6, 15.0),
                "delta_pitchangle_plus": np.full(6, 15.0),
            },
        )

    @staticmethod
    def _speed(energy, mass=constants.proton_mass):
        # Speed (km/s) of energy (eV)
        return np.sqrt(2 * constants.elementary_charge * energy / mass) / 1e3

    @staticmethod
    def _mesh(ax, i=0):
        # Radii and pitch angles (degrees) of the edges, and the values drawn
        coords = ax.collections[i].get_coordinates()
        radius = np.hypot(coords[..., 0], coords[..., 1])[:, 0]
        angle = np.rad2deg(np.arctan2(coords[0, :, 1], coords[0, :, 0]))
        values = np.ma.filled(ax.collections[i].get_array(), np.nan)
        return radius, angle, values.reshape(len(radius) - 1, len(angle) - 1)

    def test_plot_pad_polar_output(self):
        f, ax = plt.subplots(1)
        ax, cax = plot.plot_pad_polar(ax, self.pad)
        self.assertIsInstance(ax, Axes)
        self.assertIsInstance(cax, Axes)
        self.assertEqual(cax.get_ylabel(), "PSD [s$^3$ m$^{-6}$]")
        self.assertEqual(ax.get_xlabel(), "$v_\\parallel$ [km/s]")
        self.assertEqual(ax.get_ylabel(), "$v_\\perp$ [km/s]")
        r_max = self._speed(1e4)
        np.testing.assert_allclose(ax.get_xlim(), [-r_max, r_max])
        np.testing.assert_allclose(ax.get_ylim(), [0.0, r_max])

        _, cax = plot.plot_pad_polar(ax, self.pad, colorbar="top")
        self.assertEqual(cax.get_xlabel(), "PSD [s$^3$ m$^{-6}$]")
        _, cax = plot.plot_pad_polar(ax, self.pad, colorbar="none")
        self.assertIsNone(cax)

    def test_plot_pad_polar_values(self):
        f, ax = plt.subplots(1)
        plot.plot_pad_polar(ax, self.pad)
        radius, angle, values = self._mesh(ax)
        # Radii: the speeds of the energy edges (ions, km/s)
        np.testing.assert_allclose(radius, self._speed(self.energy_edges))
        # Pitch angles from 0 to 180 degrees as arcs of at most 2 degrees
        np.testing.assert_allclose(angle[[0, -1]], [0.0, 180.0], atol=1e-12)
        self.assertLessEqual(np.max(np.diff(angle)), 2.0 + 1e-9)
        # Time average 2.5 (E + 1) (alpha + 1) in each bin
        a_idx = np.floor(((angle[:-1] + angle[1:]) / 2) / 30.0)
        expected = 2.5 * np.outer(np.arange(1.0, 9.0), a_idx + 1.0)
        np.testing.assert_allclose(values, expected)

    def test_plot_pad_polar_tint(self):
        f, ax = plt.subplots(1)
        tint = ["2017-07-11T22:34:01.200"]
        plot.plot_pad_polar(ax, self.pad, tint)
        values = self._mesh(ax)[2]
        self.assertAlmostEqual(values[0, 0], 2.0)

        tint = ["2017-07-11T22:34:01.500", "2017-07-11T22:34:03.000"]
        plot.plot_pad_polar(ax, self.pad, tint)
        values = self._mesh(ax, 1)[2]
        self.assertAlmostEqual(values[0, 0], 3.5)

        with self.assertRaises(ValueError):
            plot.plot_pad_polar(ax, self.pad, ["2017-07-11T22:35:00.000"] * 2)

        with self.assertRaises(ValueError):
            plot.plot_pad_polar(ax, self.pad, ["2017-07-11T22:34:00.000"] * 3)

    def test_plot_pad_polar_sc_pot(self):
        f, ax = plt.subplots(1)
        # Ions gain q V
        plot.plot_pad_polar(ax, self.pad, sc_pot=5.0)
        radius = self._mesh(ax)[0]
        np.testing.assert_allclose(radius, self._speed(self.energy_edges + 5.0))

        # Electrons lose e V: the channels below the potential are not drawn
        pad = self.pad.assign_attrs(species="electrons")
        sc_pot = xr.DataArray(
            np.full(4, 30.0), coords=[self.pad.time.data], dims=["time"]
        )
        plot.plot_pad_polar(ax, pad, sc_pot=sc_pot)
        radius, _, values = self._mesh(ax, 1)
        speed = self._speed(self.energy_edges - 30.0, constants.electron_mass) / 1e3
        np.testing.assert_allclose(radius, np.hstack([0.0, speed[2:]]))
        np.testing.assert_allclose(values[:, 0], 2.5 * np.arange(2.0, 9.0))
        self.assertEqual(ax.get_xlabel(), "$v_\\parallel$ [10$^3$ km/s]")

    def test_plot_pad_polar_edges(self):
        f, ax = plt.subplots(1)
        # A gap between 90 and 100 degrees is blank
        pad = self.pad.copy()
        pad.attrs["delta_pitchangle_minus"] = np.array([15.0] * 3 + [5.0] + [15.0] * 2)
        pad.attrs["delta_pitchangle_plus"] = np.full(6, 15.0)
        plot.plot_pad_polar(ax, pad)
        _, angle, values = self._mesh(ax)
        middle = (angle[:-1] + angle[1:]) / 2
        gap = (middle > 90.0) & (middle < 100.0)
        self.assertTrue(np.all(np.isnan(values[:, gap])))
        self.assertFalse(np.any(np.isnan(values[:, ~gap])))

        # Rounded half widths (FPI) leave no gaps nor overlaps
        pad = self.pad.copy()
        pad.attrs["delta_energy_minus"] = self.pad.attrs["delta_energy_minus"] * 0.999
        plot.plot_pad_polar(ax, pad)
        radius, _, values = self._mesh(ax, 1)
        self.assertEqual(len(radius), 9)
        self.assertFalse(np.any(np.isnan(values)))

        # Zeros are not drawn
        pad = self.pad.copy(deep=True)
        pad.data.data[:, 0, 0] = 0.0
        plot.plot_pad_polar(ax, pad)
        values = self._mesh(ax, 2)[2]
        self.assertTrue(np.isnan(values[0, 0]))

    def test_plot_pad_polar_contour(self):
        f, ax = plt.subplots(1)
        plot.plot_pad_polar(ax, self.pad, contour=3)
        contours = [c for c in ax.collections if isinstance(c, ContourSet)]
        self.assertEqual(len(contours), 1)

        f, ax = plt.subplots(1)
        plot.plot_pad_polar(ax, self.pad, contour=[10.0, 20.0], clim=[1.0, 100.0])
        contours = [c for c in ax.collections if isinstance(c, ContourSet)]
        np.testing.assert_allclose(contours[0].levels, [1.0, np.log10(20.0)])
        np.testing.assert_allclose(ax.collections[0].norm.vmin, 1.0)
        np.testing.assert_allclose(ax.collections[0].norm.vmax, 100.0)

    def _monte_carlo(self, pad, **kwargs):
        # Edges (m/s) and values of the Monte-Carlo grid, and the density
        f, ax = plt.subplots(1)
        plot.plot_pad_polar(ax, pad, monte_carlo=True, random_state=0, **kwargs)
        coords = ax.collections[0].get_coordinates() * 1e3
        par, perp = coords[:, 0, 0], coords[0, :, 1]
        values = np.ma.filled(ax.collections[0].get_array(), np.nan)
        values = values.reshape(len(par) - 1, len(perp) - 1)
        volume = np.outer(np.diff(par), np.pi * np.diff(perp**2))
        return ax, par, perp, values, np.nansum(values * volume)

    def test_plot_pad_polar_monte_carlo(self):
        ax, par, perp, values, density = self._monte_carlo(self.pad, n_grid=40)
        # Grid of 40 x 20 cells up to the largest speed
        v_max = self._speed(1e4) * 1e3
        np.testing.assert_allclose(par[[0, -1]], [-v_max, v_max])
        np.testing.assert_allclose(perp[[0, -1]], [0.0, v_max])
        self.assertEqual(values.shape, (40, 20))
        np.testing.assert_allclose(ax.get_ylim(), [0.0, v_max / 1e3])

        # The density (time average 2.5 (E + 1) (alpha + 1)) is conserved
        speed = self._speed(self.energy_edges) * 1e3
        cos_alpha = np.cos(np.deg2rad(np.arange(0.0, 181.0, 30.0)))
        d3v = np.outer(2 * np.pi * np.diff(speed**3) / 3, -np.diff(cos_alpha))
        f_avg = 2.5 * np.outer(np.arange(1.0, 9.0), np.arange(1.0, 7.0))
        self.assertAlmostEqual(density / np.sum(f_avg * d3v), 1.0, places=12)

        # Same particles for the same random_state
        values_2 = self._monte_carlo(self.pad, n_grid=40)[3]
        np.testing.assert_array_equal(values, values_2)

    def test_plot_pad_polar_monte_carlo_tables(self):
        # Constant f = 1 with energy tables that alternate in time (FPI burst):
        # f = 1 where both tables have data, and the contours are drawn
        pad = self.pad.copy(deep=True)
        pad.data.data[:] = 1.0
        pad.energy.data[1::2] *= 10 ** (3 / 16)
        pad.attrs["delta_energy_minus"] = None
        pad.attrs["delta_energy_plus"] = None
        ax, par, perp, values, _ = self._monte_carlo(
            pad, n_mc=2000, n_grid=20, contour=[0.5]
        )
        centre = np.hypot(*np.meshgrid(par[:-1] + par[1:], perp[:-1] + perp[1:]))
        inside = (centre.T / 2 > 0.3 * perp[-1]) & (centre.T / 2 < 0.8 * perp[-1])
        self.assertAlmostEqual(np.median(values[inside]), 1.0, delta=0.02)
        contours = [c for c in ax.collections if isinstance(c, ContourSet)]
        self.assertEqual(len(contours), 1)

    def test_plot_pad_polar_invalid(self):
        f, ax = plt.subplots(1)

        with self.assertRaises(ValueError):
            plot.plot_pad_polar(ax, self.pad, colorbar="left")


class PlScatterMatrixTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        # Components linear in time, so that resampling to another grid is exact
        time = generate_timeline(100.0, 200)
        t_sec = np.arange(200) / 100.0
        self.inp1 = pyrf.ts_vec_xyz(time, np.outer(t_sec, [1.0, 2.0, 3.0]))
        self.inp2 = pyrf.ts_vec_xyz(
            time + np.timedelta64(3, "ms"), np.outer(t_sec, [-1.0, 4.0, 0.5]) + 1.0
        )
        # inp2 at the times of inp1
        self.inp2_1 = np.outer(t_sec - 0.003, [-1.0, 4.0, 0.5]) + 1.0

    def test_pl_scatter_matrix_scatter(self):
        fig, axs = plot.pl_scatter_matrix(self.inp1, self.inp2)

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(axs.shape, (3, 3))

        for i, j in itertools.product(range(3), range(3)):
            offsets = axs[j, i].collections[0].get_offsets()
            np.testing.assert_allclose(offsets[:, 0], self.inp1.data[:, i])
            np.testing.assert_allclose(offsets[:, 1], self.inp2_1[:, j], atol=1e-12)

    def test_pl_scatter_matrix_pdf(self):
        _, axs, caxs = plot.pl_scatter_matrix(self.inp1, self.inp2, pdf=True)

        self.assertEqual(axs.shape, (3, 3))
        caxs = [cax for row in caxs for cax in row]
        self.assertTrue(all(isinstance(cax, Axes) for cax in caxs))
        self.assertEqual(len({id(cax) for cax in caxs}), 9)

    def test_pl_scatter_matrix_no_inp2(self):
        with self.assertWarns(UserWarning):
            _, axs = plot.pl_scatter_matrix(self.inp1)

        offsets = axs[1, 0].collections[0].get_offsets()
        np.testing.assert_allclose(offsets[:, 0], self.inp1.data[:, 0])
        np.testing.assert_allclose(offsets[:, 1], self.inp1.data[:, 1])

    def test_pl_scatter_matrix_input_type(self):
        with self.assertRaises(TypeError):
            plot.pl_scatter_matrix(self.inp1.data, self.inp2)


class PlotMagnetosphereTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        # Offline: fake geopack, and models without OMNI data
        trace = mock.Mock(return_value=(0, 0, 0, np.zeros(3), 0, np.zeros(3)))
        self.geopack = mock.Mock(recalc=mock.Mock(), trace=trace)
        self.module = sys.modules["pyrfu.plot.plot_magnetosphere"]
        self.tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

        patches = [
            mock.patch.dict(sys.modules, {"geopack": mock.Mock(geopack=self.geopack)}),
            mock.patch.object(
                self.module,
                "magnetosphere",
                lambda model, tint: pyrf.magnetosphere(model),
            ),
        ]

        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_import_pyrfu_skips_geopack(self):
        # geopack prints and requests the IGRF coefficients online when imported
        code = "import sys, pyrfu; print('geopack' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            text=True,
        )
        self.assertEqual(result.stdout, "False\n")

    def test_plot_magnetosphere_field_lines(self):
        _, ax = plt.subplots(1)
        result = plot.plot_magnetosphere(ax, self.tint)

        self.assertIs(result, ax)
        # Magnetopause, bow shock and 2 x 190 field lines
        self.assertEqual(len(ax.lines), 2 + 2 * 190)
        self.geopack.recalc.assert_called_once()
        self.assertEqual(self.geopack.trace.call_count, 2 * 190)

    def test_plot_magnetosphere_no_field_lines(self):
        _, ax = plt.subplots(1)
        plot.plot_magnetosphere(ax, self.tint, field_lines=False)

        self.assertEqual(len(ax.lines), 2)
        self.geopack.trace.assert_not_called()


@ddt
class PlotHeatmapTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    @data(
        ("axis", "bazinga", np.random.rand(10), np.random.rand(10)),
        ("axis", np.random.rand(10, 10), "bazinga", np.random.rand(10)),
        ("axis", np.random.rand(10, 10), np.random.rand(10), "bazinga"),
    )
    @unpack
    def test_plot_heatmap_input_types(self, ax, data, x, y):
        with self.assertRaises(TypeError):
            plot.plot_heatmap(_axis(ax), data, x, y)

    @data(
        (
            "axis",
            np.random.rand(10, 10),
            np.random.rand(9),
            np.random.rand(10),
        ),
        (
            "axis",
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(9),
        ),
    )
    @unpack
    def test_plot_heatmap_input_shape(self, ax, data, x, y):
        with self.assertRaises(ValueError):
            plot.plot_heatmap(_axis(ax), data, x, y)

    @data(
        (None, np.random.rand(10, 10), np.random.rand(10), np.random.rand(10)),
        (
            "axis",
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(10),
        ),
    )
    @unpack
    def test_plot_heatmap_output(self, ax, data, x, y):
        result = plot.plot_heatmap(_axis(ax), data, x, y)
        self.assertIsInstance(result[0], AxesImage)
        self.assertIsInstance(result[1], Colorbar)


@ddt
class AnnotateHeatmapTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    def test_annotate_heatmap_input(self):
        _, ax = plt.subplots(1)
        # Create image
        im, _ = plot.plot_heatmap(
            ax, np.random.rand(10, 10), np.random.rand(10), np.random.rand(10)
        )
        with self.assertRaises(TypeError):
            plot.annotate_heatmap(im, "bazinga")

    @data(
        (
            "axis",
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(10),
            random.random(),
        ),
        (
            "axis",
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(10),
            None,
        ),
    )
    @unpack
    def test_annotate_heatmap_output(self, ax, data, x, y, threshold):
        # Create image
        im, _ = plot.plot_heatmap(_axis(ax), data, x, y)

        # Test with data provided
        plot.annotate_heatmap(im, data, threshold=threshold)

        # Test with no data provided
        plot.annotate_heatmap(im)


class MmsPlConfigTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        # Tetrahedron of ~20 km around (60000, 10000, 5000) km, with a small
        # motion averaged out
        time = generate_timeline(1.0, 5)
        self.r_mean = np.array([60000.0, 10000.0, 5000.0]) + np.array(
            [[0, 0, 0], [20, 0, 0], [10, 17, 0], [10, 6, 16]], dtype=float
        )
        motion = np.outer(np.arange(5) - 2.0, [1.0, -1.0, 0.5])
        self.r_mms = [pyrf.ts_vec_xyz(time, r + motion) for r in self.r_mean]

    def test_mms_pl_config_positions(self):
        fig, axs = plot.mms_pl_config(self.r_mms)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axs), 4)

        # X-Z, Y-Z and X-Y panels in Earth radii
        for ax, (i_x, i_y) in zip(axs[:3], [(0, 2), (1, 2), (0, 1)]):
            offsets = np.vstack([c.get_offsets()[0] for c in ax.collections])
            expected = self.r_mean[:, [i_x, i_y]] / R_E
            np.testing.assert_allclose(offsets, expected, rtol=1e-12)

        # Relative positions in km, inside the axis limits
        delta_r = self.r_mean - np.mean(self.r_mean, axis=0)
        points = np.array(
            [np.ravel(c._offsets3d) for c in axs[3].collections[:4]]  # noqa
        )
        np.testing.assert_allclose(points, delta_r, atol=1e-9)
        for lim in [axs[3].get_xlim(), axs[3].get_zlim()]:
            self.assertGreaterEqual(lim[1], np.max(np.abs(delta_r)))

    def test_mms_pl_config_far(self):
        # Positions beyond 20 R_E widen the 2d panels
        r_mms = [r + 30 * R_E for r in self.r_mms]
        _, axs = plot.mms_pl_config(r_mms)
        self.assertGreater(axs[0].get_xlim()[0], 30 + np.max(self.r_mean) / R_E)


class ColorbarTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    def test_colorbar(self):
        f, ax = plt.subplots(1)
        image = ax.pcolormesh(
            np.arange(4.0), np.arange(3.0), np.arange(12.0).reshape(3, 4)
        )
        cax = plot.colorbar(image, ax, pad=0.02, width=0.05)

        self.assertIn(cax, f.axes)
        pos, cpos = ax.get_position(), cax.get_position()
        self.assertAlmostEqual(cpos.x0, pos.x1 + 0.02)
        self.assertAlmostEqual(cpos.width, 0.05)
        self.assertAlmostEqual(cpos.y0, pos.y0)
        self.assertAlmostEqual(cpos.height, pos.height)
        np.testing.assert_allclose(cax.get_ylim(), [0.0, 11.0])


class IonBrazilPlotThreshTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    def test_ion_brazil_plot_thresh(self):
        f, ax = plt.subplots(1)
        plot.ion_brazil_plot_thresh(ax)

        self.assertEqual(len(ax.lines), 4)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        np.testing.assert_allclose(ax.get_xlim(), [1e-2, 5e3])
        np.testing.assert_allclose(ax.get_ylim(), [0.3, 1e1])

        beta = np.logspace(-2, 4)
        for line, instability in zip(
            ax.lines,
            ["proton-cyclotron", "mirror", "parallel-firehose", "oblique-firehose"],
        ):
            np.testing.assert_allclose(line.get_xdata(), beta)
            np.testing.assert_allclose(
                line.get_ydata(), ion_anisotropy_thresh(beta, instability, "10^-2")
            )

        self.assertEqual(len(f.legends), 1)
        self.assertListEqual(
            [text.get_text() for text in f.legends[0].get_texts()],
            [
                "Proton cyclotron",
                "Mirror Mode",
                "Parallel Firehose",
                "Oblique Firehose",
            ],
        )

    def test_ion_brazil_plot_thresh_options(self):
        f, ax = plt.subplots(1)
        colors = ["k", "r", "g", "y"]
        plot.ion_brazil_plot_thresh(ax, legend=False, colors=colors)

        self.assertEqual(len(f.legends), 0)
        for line, color in zip(ax.lines, colors):
            self.assertEqual(to_rgba(line.get_color()), to_rgba(color))

    def test_ion_brazil_plot_thresh_other_figure(self):
        # The legend goes in the figure of the axis, not the current figure
        f, ax = plt.subplots(1)
        f_current = plt.figure()
        plot.ion_brazil_plot_thresh(ax)

        self.assertEqual(len(f.legends), 1)
        self.assertEqual(len(f_current.legends), 0)


class MakeLabelsTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    def test_make_labels(self):
        _, axs = plt.subplots(3)
        result = plot.make_labels(axs, [0.02, 0.9], fontsize=7)

        self.assertIs(result, axs)
        for axis, label in zip(axs, ["(a)", "(b)", "(c)"]):
            self.assertEqual(len(axis.texts), 1)
            self.assertEqual(axis.texts[0].get_text(), label)
            self.assertEqual(axis.texts[0].get_position(), (0.02, 0.9))
            self.assertIs(axis.texts[0].get_transform(), axis.transAxes)
            self.assertEqual(axis.texts[0].get_fontsize(), 7)

    def test_make_labels_pref_suff_pad(self):
        _, axs = plt.subplots(2)
        plot.make_labels(axs, [0.1, 0.1], pref="1", suff="'", pad=2)

        self.assertListEqual(
            [axis.texts[0].get_text() for axis in axs], ["(1c')", "(1d')"]
        )

    def test_make_labels_3d(self):
        f = plt.figure()
        axs = [f.add_subplot(1, 2, 1), f.add_subplot(1, 2, 2, projection="3d")]
        plot.make_labels(axs, [0.1, 0.9])

        # text2D on the 3d axis, at the same axes position as on the 2d one
        for axis, label in zip(axs, ["(a)", "(b)"]):
            self.assertEqual(axis.texts[0].get_text(), label)
            self.assertEqual(axis.texts[0].get_position(), (0.1, 0.9))
            self.assertIs(axis.texts[0].get_transform(), axis.transAxes)

    def test_make_labels_too_many(self):
        _, axs = plt.subplots(3)

        with self.assertRaises(ValueError):
            plot.make_labels(axs, [0.1, 0.9], pad=24)


class PlotContourTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        x, y = np.linspace(-2.0, 2.0, 21), np.linspace(-1.0, 1.0, 11)
        self.inp = xr.DataArray(
            np.add.outer(x**2, y**2), coords=[x, y], dims=["x", "y"]
        )

    def test_plot_contour(self):
        _, ax = plt.subplots(1)
        axis, clines = plot.plot_contour(ax, self.inp, levels=[0.5, 1.0, 2.0])

        self.assertIs(axis, ax)
        self.assertIsInstance(clines, ContourSet)
        np.testing.assert_allclose(clines.levels, [0.5, 1.0, 2.0])
        # x along the horizontal axis
        np.testing.assert_allclose(ax.dataLim.intervalx, [-2.0, 2.0], atol=0.21)

    def test_plot_contour_input_type(self):
        _, ax = plt.subplots(1)

        with self.assertRaises(TypeError):
            plot.plot_contour(None, self.inp)

        with self.assertRaises(TypeError):
            plot.plot_contour(ax, self.inp.data)


class PlotProjectionTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        # Velocities in km/s, plotted in 10^3 km/s
        self.v_x = np.linspace(-2e3, 2e3, 20)
        self.v_y = np.linspace(-1e3, 1e3, 10)
        self.f_mat = np.outer(np.arange(1.0, 21.0), np.arange(1.0, 11.0)) * 1e-15

    def test_plot_projection(self):
        f, ax = plt.subplots(1)
        axis, caxis = plot.plot_projection(
            ax, self.v_x, self.v_y, self.f_mat, vlim=1.5e3, clim=[-15, -12]
        )

        self.assertIs(axis, ax)
        self.assertIn(caxis, f.axes)
        image = ax.collections[0]
        np.testing.assert_allclose(
            np.asarray(image.get_array()).reshape(10, 20), np.log10(self.f_mat.T)
        )
        self.assertEqual((image.norm.vmin, image.norm.vmax), (-15, -12))
        np.testing.assert_allclose(ax.get_xlim(), [-1.5, 1.5])
        np.testing.assert_allclose(ax.get_ylim(), [-1.5, 1.5])
        self.assertEqual(ax.get_aspect(), 1.0)
        # On the right
        self.assertGreater(caxis.get_position().x0, ax.get_position().x1)

    def test_plot_projection_colorbar(self):
        f, ax = plt.subplots(1)
        _, caxis = plot.plot_projection(
            ax, self.v_x, self.v_y, self.f_mat, colorbar="top"
        )
        self.assertGreater(caxis.get_position().y0, ax.get_position().y1)
        self.assertEqual(caxis.xaxis.get_label_position(), "top")

        f, ax = plt.subplots(1)
        _, caxis = plot.plot_projection(
            ax, self.v_x, self.v_y, self.f_mat, colorbar="none"
        )
        self.assertIsNone(caxis)
        self.assertEqual(len(f.axes), 1)

        with self.assertRaises(NotImplementedError):
            plot.plot_projection(ax, self.v_x, self.v_y, self.f_mat, colorbar="left")

    def test_plot_projection_cmap(self):
        _, ax = plt.subplots(1)
        plot.plot_projection(ax, self.v_x, self.v_y, self.f_mat, cmap="viridis")
        self.assertEqual(ax.collections[0].get_cmap().name, "viridis")

        with self.assertRaises(TypeError):
            plot.plot_projection(
                ax, self.v_x, self.v_y, self.f_mat, cmap=mpl.colormaps["viridis"]
            )

    def test_plot_projection_other_figure(self):
        # The colour bar goes in the figure of the axis, not the current figure
        f, ax = plt.subplots(1)
        f_current = plt.figure()
        _, caxis = plot.plot_projection(ax, self.v_x, self.v_y, self.f_mat)

        self.assertIn(caxis, f.axes)
        self.assertEqual(len(f_current.axes), 0)


class PlotReduced2dTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        v_x, v_y = np.linspace(-3e3, 3e3, 31), np.linspace(-2e3, 2e3, 21)
        self.f2d = xr.DataArray(
            1e-9 * np.exp(-np.add.outer(v_x**2, v_y**2) / 1e6) + 1e-14,
            coords=[v_x, v_y],
            dims=["vx", "vy"],
        )

    def test_plot_reduced_2d(self):
        f, ax = plt.subplots(1)
        axis, cax = plot.plot_reduced_2d(ax, self.f2d)

        self.assertIs(axis, ax)
        self.assertIn(cax, f.axes)
        image = ax.collections[0]
        np.testing.assert_allclose(
            np.asarray(image.get_array()).reshape(21, 31), self.f2d.data.T
        )
        self.assertIsInstance(image.norm, LogNorm)
        self.assertEqual((image.norm.vmin, image.norm.vmax), (1e-13, 1e-8))
        # One contour per decade of the colour range
        clines = [c for c in ax.collections if isinstance(c, ContourSet)][0]
        np.testing.assert_allclose(clines.levels, np.logspace(-13, -8, 6))
        np.testing.assert_allclose(ax.get_xlim(), [-3e3, 3e3])
        np.testing.assert_allclose(ax.get_ylim(), [-2e3, 2e3])
        # Colour bar on top, with the contour lines
        self.assertGreater(cax.get_position().y0, ax.get_position().y1)
        self.assertEqual(cax.xaxis.get_label_position(), "top")
        self.assertGreater(len(cax.collections), 0)

    def test_plot_reduced_2d_clim(self):
        _, ax = plt.subplots(1)
        plot.plot_reduced_2d(ax, self.f2d, clim=[1e-12, 1e-9])

        image = ax.collections[0]
        self.assertEqual((image.norm.vmin, image.norm.vmax), (1e-12, 1e-9))
        clines = [c for c in ax.collections if isinstance(c, ContourSet)][0]
        np.testing.assert_allclose(clines.levels, np.logspace(-12, -9, 4))

    def test_plot_reduced_2d_other_figure(self):
        # The colour bar goes in the figure of the axis, not the current figure
        f, ax = plt.subplots(1)
        f_current = plt.figure()
        _, cax = plot.plot_reduced_2d(ax, self.f2d)

        self.assertIn(cax, f.axes)
        self.assertEqual(len(f_current.axes), 0)


class PlotSpectrTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        self.time = generate_timeline(64.0, 50)
        self.energy = np.logspace(1, 4, 8)
        self.data = np.outer(np.arange(1.0, 51.0), np.arange(1.0, 9.0))
        self.inp = pyrf.ts_spectr(self.time, self.energy, self.data)

    @staticmethod
    def _image(ax):
        return ax.collections[0]

    def test_plot_spectr_dataarray(self):
        f, ax = plt.subplots(1)
        axis, cax = plot.plot_spectr(ax, self.inp)

        self.assertIs(axis, ax)
        self.assertIn(cax, f.axes)
        image = self._image(ax)
        np.testing.assert_allclose(
            np.asarray(image.get_array()).reshape(8, 50), self.data.T
        )
        self.assertNotIsInstance(image.norm, LogNorm)
        np.testing.assert_allclose(ax.get_ylim(), [10.0, 1e4])
        self.assertEqual(ax.get_yscale(), "linear")
        self.assertIsInstance(
            ax.xaxis.get_major_formatter(), mpl.dates.ConciseDateFormatter
        )
        # Colour bar on the right, 4 ticks at most
        self.assertAlmostEqual(cax.get_position().x0, ax.get_position().x1 + 0.01)
        self.assertIsInstance(cax.yaxis.get_major_locator(), mpl.ticker.MaxNLocator)

    def test_plot_spectr_new_axis(self):
        axis, _ = plot.plot_spectr(None, self.inp)

        self.assertIsInstance(axis, Axes)
        self.assertIn(axis.get_figure().number, plt.get_fignums())

    def test_plot_spectr_not_time(self):
        # A spectrum of another quantity than time (no date formatter)
        inp = xr.DataArray(
            self.data, coords=[np.arange(50.0), self.energy], dims=["x", "energy"]
        )
        _, ax = plt.subplots(1)
        plot.plot_spectr(ax, inp, colorbar="none")

        self.assertNotIsInstance(
            ax.xaxis.get_major_formatter(), mpl.dates.ConciseDateFormatter
        )

    def test_plot_spectr_scales(self):
        _, ax = plt.subplots(1)
        _, cax = plot.plot_spectr(ax, self.inp, yscale="log", cscale="log")

        self.assertEqual(ax.get_yscale(), "log")
        self.assertIsInstance(ax.yaxis.get_major_locator(), mpl.ticker.LogLocator)
        self.assertIsInstance(self._image(ax).norm, LogNorm)
        self.assertEqual(self._image(ax).norm.vmin, 1.0)
        self.assertEqual(self._image(ax).norm.vmax, 400.0)
        self.assertIsInstance(cax.yaxis.get_major_locator(), mpl.ticker.LogLocator)

    def test_plot_spectr_clim(self):
        for cscale, norm_type in [("log", LogNorm), ("linear", mpl.colors.Normalize)]:
            _, ax = plt.subplots(1)
            plot.plot_spectr(ax, self.inp, cscale=cscale, clim=[2.0, 100.0])

            norm = self._image(ax).norm
            self.assertIsInstance(norm, norm_type)
            self.assertEqual((norm.vmin, norm.vmax), (2.0, 100.0))

    def test_plot_spectr_clim_sequence(self):
        # A tuple or an array as clim (was ignored: only lists were used)
        for clim in [(2.0, 100.0), np.array([2.0, 100.0])]:
            _, ax = plt.subplots(1)
            plot.plot_spectr(ax, self.inp, cscale="log", clim=clim)

            norm = self._image(ax).norm
            self.assertEqual((norm.vmin, norm.vmax), (2.0, 100.0))

    def test_plot_spectr_microseconds(self):
        # Times in another unit than ns (xarray keeps it): concise dates
        inp = self.inp.assign_coords(time=self.time.astype("datetime64[us]"))
        _, ax = plt.subplots(1)
        plot.plot_spectr(ax, inp)

        self.assertIsInstance(
            ax.xaxis.get_major_formatter(), mpl.dates.ConciseDateFormatter
        )

    def test_plot_spectr_norm(self):
        # Any matplotlib norm as cscale
        norm = mpl.colors.SymLogNorm(linthresh=1.0, vmin=-10.0, vmax=10.0)
        _, ax = plt.subplots(1)
        plot.plot_spectr(ax, self.inp, cscale=norm, cmap="viridis")

        self.assertIs(self._image(ax).norm, norm)
        self.assertEqual(self._image(ax).get_cmap().name, "viridis")

    def test_plot_spectr_colorbar(self):
        _, ax = plt.subplots(1)
        _, cax = plot.plot_spectr(ax, self.inp, colorbar="right", pad=0.05)
        self.assertAlmostEqual(cax.get_position().x0, ax.get_position().x1 + 0.05)

        # Default and given pad
        for cscale, locator, pad in [
            ("linear", mpl.ticker.MaxNLocator, {}),
            ("log", mpl.ticker.LogLocator, {"pad": 0.05}),
        ]:
            _, ax = plt.subplots(1)
            _, cax = plot.plot_spectr(
                ax, self.inp, cscale=cscale, colorbar="top", **pad
            )

            self.assertAlmostEqual(
                cax.get_position().y0,
                ax.get_position().y1 + pad.get("pad", 0.01),
            )
            self.assertEqual(cax.xaxis.get_label_position(), "top")
            self.assertIsInstance(cax.xaxis.get_major_locator(), locator)

        f, ax = plt.subplots(1)
        self.assertIs(plot.plot_spectr(ax, self.inp, colorbar="none"), ax)
        self.assertEqual(len(f.axes), 1)

        with self.assertRaises(NotImplementedError):
            plot.plot_spectr(ax, self.inp, colorbar="left")

    def test_plot_spectr_pad_zero(self):
        # pad=0 puts the colour bar against the axis (it was taken as 0.01)
        _, ax = plt.subplots(1)
        _, cax = plot.plot_spectr(ax, self.inp, colorbar="right", pad=0)
        self.assertAlmostEqual(cax.get_position().x0, ax.get_position().x1)

        _, ax = plt.subplots(1)
        _, cax = plot.plot_spectr(ax, self.inp, colorbar="top", pad=0)
        self.assertAlmostEqual(cax.get_position().y0, ax.get_position().y1)

    def test_plot_spectr_dataset(self):
        # Time-dependent energy table: as a coordinate (named energy or not) or
        # as a second data variable
        energy = np.tile(self.energy, (50, 1))
        energy[25:] *= 2.0
        datasets = [
            xr.Dataset(
                {"data": (["time", "idx"], self.data)},
                coords={
                    "time": self.time,
                    "idx": np.arange(8),
                    "energy": (["time", "idx"], energy),
                },
            ),
            xr.Dataset(
                {"data": (["time", "idx"], self.data)},
                coords={
                    "time": self.time,
                    "idx": np.arange(8),
                    "en": (["time", "idx"], energy),
                },
            ),
            xr.Dataset(
                {
                    "data": (["time", "idx"], self.data),
                    "energy": (["time", "idx"], energy),
                },
                coords={"time": self.time, "idx": np.arange(8)},
            ),
        ]

        for dataset in datasets:
            _, ax = plt.subplots(1)
            plot.plot_spectr(ax, dataset, yscale="log")

            image = self._image(ax)
            np.testing.assert_allclose(
                np.asarray(image.get_array()).reshape(8, 50), self.data.T
            )
            np.testing.assert_allclose(ax.get_ylim(), [10.0, 2e4])

    def test_plot_spectr_input_type(self):
        _, ax = plt.subplots(1)

        with self.assertRaises(TypeError):
            plot.plot_spectr(ax, self.data)


class PlotSurfTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)
        phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 9), np.linspace(0, np.pi, 5))
        self.xyz = [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
        self.c = np.cos(theta)
        self.cmap = mpl.colormaps["viridis"]
        self.norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

    def _plot(self, cax_pos):
        f = plt.figure()
        ax = f.add_subplot(projection="3d")
        axis, caxis = plot.plot_surf(
            ax, *self.xyz, self.c, self.cmap, self.norm, cax_pos
        )
        self.assertIs(axis, ax)
        self.assertIn(caxis, f.axes)
        return ax.get_position(), caxis

    def test_plot_surf(self):
        f = plt.figure()
        ax = f.add_subplot(projection="3d")
        plot.plot_surf(ax, *self.xyz, self.c, self.cmap, self.norm)

        self.assertEqual(len(ax.collections), 1)
        # Bottom colour bar with the norm's range
        caxis = f.axes[1]
        self.assertLess(caxis.get_position().y0, ax.get_position().y0)
        np.testing.assert_allclose(caxis.get_xlim(), [-1.0, 1.0])

    def test_plot_surf_cax_pos(self):
        pos, caxis = self._plot("top")
        self.assertGreater(caxis.get_position().y0, pos.y1)
        self.assertEqual(caxis.xaxis.get_label_position(), "top")

        pos, caxis = self._plot("left")
        self.assertLess(caxis.get_position().x0, pos.x0)
        self.assertEqual(caxis.yaxis.get_label_position(), "left")

        pos, caxis = self._plot("right")
        self.assertGreater(caxis.get_position().x0, pos.x1)

        with self.assertRaises(ValueError):
            self._plot("center")

    def test_plot_surf_other_figure(self):
        # The colour bar goes in the figure of the axis, not the current figure
        f = plt.figure()
        ax = f.add_subplot(projection="3d")
        f_current = plt.figure()
        _, caxis = plot.plot_surf(ax, *self.xyz, self.c, self.cmap, self.norm)

        self.assertIn(caxis, f.axes)
        self.assertEqual(len(f_current.axes), 0)


class SpanTintTestCase(unittest.TestCase):
    def setUp(self):
        _close_new_figures(self)

    def test_span_tint(self):
        tint = ["2019-01-01T00:00:01.000000000", "2019-01-01T00:00:02.500000000"]
        _, axs = plt.subplots(2)
        result = plot.span_tint(axs, tint, ymin=0.2, ymax=0.7, color="r")

        self.assertIs(result, axs)
        t_lim = date2num(np.array(tint, dtype="datetime64[ns]"))
        for axis in axs:
            self.assertEqual(len(axis.patches), 1)
            verts = axis.patches[0].get_verts()
            x_span = axis.transData.inverted().transform(verts)[:, 0]
            y_span = axis.transAxes.inverted().transform(verts)[:, 1]
            np.testing.assert_allclose([x_span.min(), x_span.max()], t_lim, atol=1e-9)
            np.testing.assert_allclose([y_span.min(), y_span.max()], [0.2, 0.7])
            self.assertEqual(axis.patches[0].get_facecolor()[:3], (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
