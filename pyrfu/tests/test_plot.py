#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import random
import unittest

import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np
from ddt import data, ddt, unpack
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage

# Local imports
from .. import plot
from . import generate_data, generate_ts


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
class AddPositionTestCase(unittest.TestCase):
    @data(generate_ts(64.0, 100, tensor_order=1))
    def test_add_position_output(self, value):
        result = plot.add_position(plt.subplots(1)[1], value)
        self.assertIsInstance(result, Axes)


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


if __name__ == "__main__":
    unittest.main()
