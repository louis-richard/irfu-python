#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates, gridspec
from matplotlib.widgets import Button, SpanSelector

# Local imports
from ..plot import plot_line
from .mva import mva
from .new_xyz import new_xyz
from .norm import norm
from .time_clip import time_clip

__author__ = "Atlas Silverhult"
__email__ = "atlas.silverhult.9977@student.uu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = ["mva_gui"]


def mva_gui(inp):
    r"""GUI to interactively perform minimum variance analysis (MVA) on
    time series data by selecting the time interval to apply MVA on.
    The return of this function is a callback to the GUI object
    and class attributes like the minimum variance direction vector are accesable
    through this callback by the method get_minvar().


    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the quantity to load into GUI and perform MVA on.

    Returns
    -------
    mva_callback :
        Returns MvaGui object to access attributes. In order to keep
        GUI responsive and interactive, a reference to this object is needed.

    """

    mva_callback = MvaGui(inp)

    return mva_callback


class MvaGui:
    r"""Class to display and update GUI elements of MVA."""

    def __init__(self, b):
        # Time series data
        self.b = b
        self.t = self.b.time.data

        # Figure and GUI
        self.fig = self.init_fig(self.b)

        axreset = self.fig.add_axes([0.91, 0.66, 0.04, 0.03])
        self.resetbutton = Button(axreset, "Reset", hovercolor="0.75")
        self.resetbutton.on_clicked(self.reset_selection)

        self.span = SpanSelector(
            self.fig.axes[0],
            self.update_onselect,
            "horizontal",
            useblit=True,
            props={"alpha": 0, "facecolor": "black"},
            interactive=True,
            minspan=0,
            drag_from_anywhere=True,
        )

        # Normal vec - to be calculated
        self.minvar = None
        self.errors = None

    @staticmethod
    def init_fig(b_xyz):
        r"""Initialize figure window with time series. Returns figure object
        to access the axes.
        """

        fig = plt.figure(figsize=(10, 8))
        legend_options = {
            "ncol": 1,
            "loc": "center left",
            "frameon": True,
            "framealpha": 1,
            "bbox_to_anchor": (1, 0.5),
        }
        gs = gridspec.GridSpec(
            3, 3, top=0.9, left=0.05, right=0.9, hspace=0.3, wspace=0.35
        )
        # Time series data
        ax1 = fig.add_subplot(gs[0, :])

        # MVA transformed frame
        fig.add_subplot(gs[1, :])

        # Min/Max hodogram
        fig.add_subplot(gs[2, 0])

        # Interm/Max hodogram
        fig.add_subplot(gs[2, 1])

        # Text display
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis("off")

        # Plot b_xyz in the first axis
        plot_line(ax1, b_xyz)
        plot_line(ax1, norm(b_xyz), color="black")
        ax1.set_ylabel("$B$ [nT]")

        ax1.set_title("Time series for magnetic field")
        b_labels = ["$B_x$", "$B_y$", "$B_z$", "$|B|$"]
        ax1.legend(b_labels, **legend_options)
        return fig

    def reset_selection(self):
        r"""Resets MVA selection. Event is passed on from button click but has
        no meaning or use here.
        """

        ax3 = self.fig.axes[2]
        ax4 = self.fig.axes[3]
        ax5 = self.fig.axes[4]
        self.span.set_visible(False)
        self.update_fig(self.b)
        ax3.clear()
        ax4.clear()
        for txt in ax5.texts:
            txt.set_visible(False)
        plt.draw()

    def update_onselect(self, tmin, tmax):
        r"""This function is called when a span is selected with spanselector."""

        indmin, indmax = np.searchsorted(dates.date2num(self.t), (tmin, tmax))
        indmax = min(len(self.t) - 1, indmax)

        region_t = self.t[indmin:indmax]

        # convert times to datetime64
        tmin = np.datetime64(dates.num2date(tmin).replace(tzinfo=None))
        tmax = np.datetime64(dates.num2date(tmax).replace(tzinfo=None))

        b_xyz_clip = time_clip(self.b, [tmin, tmax])
        if len(region_t) >= 2:
            self.update_fig(b_xyz_clip)
            self.fig.canvas.draw_idle()

    @staticmethod
    def force_positive(mva_frame, b_xyz):
        r"""Force maximum variance direction to be positive."""

        frame = mva_frame[2]
        frame[:, 0] *= np.sign(max(frame[:, 0], key=abs))

        # Keep frame right-handed
        frame[:, 2] = np.cross(frame[:, 0], frame[:, 1])
        b_lmn = new_xyz(b_xyz, frame)
        return b_lmn, mva_frame[1], frame

    def update_fig(self, b_xyz_clip):
        r"""Update figure based on the clipped time series"""

        legend_options = {
            "ncol": 1,
            "loc": "center left",
            "frameon": True,
            "framealpha": 1,
            "bbox_to_anchor": (1, 0.5),
        }

        # Perform MVA on the clipped time series
        b_lmn_clip, lamb_clip, frame_clip = self.force_positive(
            mva(b_xyz_clip), b_xyz_clip
        )

        # Pick out time series for each component of the new magnetic field
        b_1 = b_lmn_clip[:, 0]
        b_2 = b_lmn_clip[:, 1]
        b_3 = b_lmn_clip[:, 2]

        # Pick out each eigenvector as the columns of frame_clip
        v1 = frame_clip[:, 0]
        v2 = frame_clip[:, 1]
        v3 = frame_clip[:, 2]

        # Define axse of figure
        ax2 = self.fig.axes[1]
        ax3 = self.fig.axes[2]
        ax4 = self.fig.axes[3]
        ax5 = self.fig.axes[4]

        # Clear axis graphs from previous selection
        ax2.clear()
        ax3.clear()
        ax4.clear()

        # Plot b in MVA frame given by frame_clip
        b_new = new_xyz(self.b, frame_clip)
        plot_line(ax2, b_new)
        plot_line(ax2, norm(b_new), color="black")
        b2_labels = ["max", "interm", "min", "abs"]
        ax2.legend(b2_labels, **legend_options)
        ax2.set_ylabel("$B$ [nT]")
        ax2.set_title("MVA frame")

        # Update B3/B1 hodogram
        ax3.axis("equal")
        ax3.plot(b_3, b_1, c="black")
        ax3.set_ylabel("max")
        ax3.set_xlabel("min")

        # Update B2/B1 hodogram
        ax4.axis("equal")
        ax4.plot(b_2, b_1, c="black")
        ax4.set_ylabel("max")
        ax4.set_xlabel("interm")

        # Text
        for txt in ax5.texts:
            txt.set_visible(False)

        val_textstring = (
            f"$\\lambda_1$ = {np.round(lamb_clip[0], 2)}\n"
            f"$\\lambda_2$ = {np.round(lamb_clip[1], 2)}\n"
            f"$\\lambda_3$ = {np.round(lamb_clip[2], 2)}"
        )
        ratio_textstring = (
            f"\n"
            f"$\\lambda_1 / \\lambda_2$ ="
            f" {np.round(lamb_clip[0] / lamb_clip[1], 1)}\n"
            f"$\\lambda_2 / \\lambda_3$ = {np.round(lamb_clip[1] / lamb_clip[2], 1)}"
        )
        vec_textstring = (
            f"\n"
            f"$x_1$ = {np.round(v1, 2)}\n"
            f"$x_2$ = {np.round(v2, 2)}\n"
            f"$x_3$ = {np.round(v3, 2)}"
        )
        fontsize = "large"
        ax5.text(
            0,
            1,
            val_textstring,
            fontsize=fontsize,
            ha="left",
            va="top",
            transform=ax5.transAxes,
        )
        ax5.text(
            0,
            0.5,
            ratio_textstring,
            fontsize=fontsize,
            ha="left",
            va="center",
            transform=ax5.transAxes,
        )
        ax5.text(
            0,
            0,
            vec_textstring,
            fontsize=fontsize,
            ha="left",
            va="center",
            transform=ax5.transAxes,
        )

        # Set minimum variance vector
        self.minvar = v3

        return b_lmn_clip, lamb_clip, frame_clip

    def get_minvar(self):
        r"""Access the minimum varience direction vector from mva_gui_class object
        via this method.
        """

        if self.minvar is not None:
            out = self.minvar
        else:
            raise RuntimeError("Normal has not been calculated yet")

        return out
