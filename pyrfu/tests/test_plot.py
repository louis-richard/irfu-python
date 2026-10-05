#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
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
from matplotlib.dates import date2num
from matplotlib.image import AxesImage

# Local imports
from .. import plot, pyrf
from ..constants import R_E
from . import generate_data, generate_timeline, generate_ts


@ddt
class PlotLineTestCase(unittest.TestCase):
    @data((0.0, generate_ts(64.0, 100)), (plt.subplots(3)[1], generate_ts(64.0, 100)))
    @unpack
    def test_plot_line_axis_type(self, axis, inp):
        with self.assertRaises(TypeError):
            plot.plot_line(axis, inp)

    @data((plt.subplots(1)[1], generate_data(100)))
    @unpack
    def test_plot_line_inp_type(self, axis, inp):
        with self.assertRaises(TypeError):
            plot.plot_line(axis, inp)

    @data(
        (plt.subplots(1)[1], generate_ts(64.0, 100, tensor_order=random.randint(3, 10)))
    )
    @unpack
    def test_plot_line_inp_shape(self, axis, inp):
        with self.assertRaises(NotImplementedError):
            plot.plot_line(axis, inp)

    @data(
        (None, generate_ts(64.0, 100, tensor_order=random.randint(0, 2))),
        (plt.subplots(1)[1], generate_ts(64.0, 100, tensor_order=random.randint(0, 2))),
        (
            plt.subplots(3)[1][0],
            generate_ts(64.0, 100, tensor_order=random.randint(0, 2)),
        ),
    )
    @unpack
    def test_plot_line_output(self, axis, inp):
        result = plot.plot_line(axis, inp)
        self.assertIsInstance(result, Axes)


@ddt
class PlotClinesTestCase(unittest.TestCase):
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
        plt.close("all")

    def test_plot_clines_cscale(self):
        inp = xr.DataArray(
            np.random.rand(10, 3),
            coords=[generate_ts(64.0, 10).time.data, [1.0, 10.0, 100.0]],
            dims=["time", "energy"],
        )
        with self.assertRaises(NotImplementedError):
            plot.plot_clines(plt.subplots(1)[1], inp, cscale="lin")
        plt.close("all")


@ddt
class AddPositionTestCase(unittest.TestCase):
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
        plt.close("all")

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
        plt.close("all")

    def test_add_position_units(self):
        with self.assertRaises(ValueError):
            plot.add_position(
                plt.subplots(1)[1], generate_ts(64.0, 100, tensor_order=1), units="m"
            )
        plt.close("all")


@ddt
class PlTxTestCase(unittest.TestCase):
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
            plt.subplots(1)[1],
            [
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
                generate_ts(64.0, 100, tensor_order=0),
            ],
        ),
        (
            plt.subplots(1)[1],
            [
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
                generate_ts(64.0, 100, tensor_order=1),
            ],
        ),
        (
            plt.subplots(1)[1],
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
        result = plot.pl_tx(ax, value, 0)
        self.assertIsInstance(result, Axes)


@ddt
class ZoomTestCase(unittest.TestCase):
    @data((None, plt.subplots(2)[1][1]), (plt.subplots(2)[1][0], None))
    @unpack
    def test_zoom_input(self, ax1, ax2):
        with self.assertRaises(TypeError):
            plot.zoom(ax1, ax2)

    @data((plt.subplots(2)[1][0], plt.subplots(2)[1][1]))
    @unpack
    def test_zoom_output(self, ax1, ax2):
        result = plot.zoom(ax1, ax2)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], Axes)
        self.assertIsInstance(result[1], Axes)


@ddt
class SetColorCycleTestCase(unittest.TestCase):
    @data("pyrfu", "oceanic", "tab", "")
    def set_color_cycle_input(self, value):
        result = plot.set_color_cycle(value)
        self.asssertIsInstance(result[0], list)
        self.asssertIsInstance(result[1], str)


class UsePyrfuStyleTestCase(unittest.TestCase):
    @staticmethod
    def _which(missing=()):
        return lambda cmd: None if cmd in missing else f"/usr/bin/{cmd}"

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


@ddt
class PlotHeatmapTestCase(unittest.TestCase):
    @data(
        (plt.subplots(1)[1], "bazinga", np.random.rand(10), np.random.rand(10)),
        (plt.subplots(1)[1], np.random.rand(10, 10), "bazinga", np.random.rand(10)),
        (plt.subplots(1)[1], np.random.rand(10, 10), np.random.rand(10), "bazinga"),
    )
    @unpack
    def test_plot_heatmap_input_types(self, ax, data, x, y):
        with self.assertRaises(TypeError):
            plot.plot_heatmap(ax, data, x, y)

    @data(
        (
            plt.subplots(1)[1],
            np.random.rand(10, 10),
            np.random.rand(9),
            np.random.rand(10),
        ),
        (
            plt.subplots(1)[1],
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(9),
        ),
    )
    @unpack
    def test_plot_heatmap_input_shape(self, ax, data, x, y):
        with self.assertRaises(ValueError):
            plot.plot_heatmap(ax, data, x, y)

    @data(
        (None, np.random.rand(10, 10), np.random.rand(10), np.random.rand(10)),
        (
            plt.subplots(1)[1],
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(10),
        ),
    )
    @unpack
    def test_plot_heatmap_output(self, ax, data, x, y):
        result = plot.plot_heatmap(ax, data, x, y)
        self.assertIsInstance(result[0], AxesImage)
        self.assertIsInstance(result[1], Colorbar)


@ddt
class AnnotateHeatmapTestCase(unittest.TestCase):
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
            plt.subplots(1)[1],
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(10),
            random.random(),
        ),
        (
            plt.subplots(1)[1],
            np.random.rand(10, 10),
            np.random.rand(10),
            np.random.rand(10),
            None,
        ),
    )
    @unpack
    def test_annotate_heatmap_output(self, ax, data, x, y, threshold):
        # Create image
        im, _ = plot.plot_heatmap(ax, data, x, y)

        # Test with data provided
        plot.annotate_heatmap(im, data, threshold=threshold)

        # Test with no data provided
        plot.annotate_heatmap(im)


class MmsPlConfigTestCase(unittest.TestCase):
    def setUp(self):
        # Tetrahedron of ~20 km around (60000, 10000, 5000) km, with a small
        # motion averaged out
        time = generate_timeline(1.0, 5)
        self.r_mean = np.array([60000.0, 10000.0, 5000.0]) + np.array(
            [[0, 0, 0], [20, 0, 0], [10, 17, 0], [10, 6, 16]], dtype=float
        )
        motion = np.outer(np.arange(5) - 2.0, [1.0, -1.0, 0.5])
        self.r_mms = [pyrf.ts_vec_xyz(time, r + motion) for r in self.r_mean]

    def tearDown(self):
        plt.close("all")

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


if __name__ == "__main__":
    unittest.main()
