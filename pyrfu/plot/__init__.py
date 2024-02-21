#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import shutil
import warnings

# 3rd party imports
import cycler
import matplotlib as mpl
from matplotlib import style

# Local imports
from .add_position import add_position
from .annotate_heatmap import annotate_heatmap
from .colorbar import colorbar
from .make_labels import make_labels
from .mms_pl_config import mms_pl_config
from .pl_scatter_matrix import pl_scatter_matrix
from .pl_tx import pl_tx
from .plot_ang_ang import plot_ang_ang
from .plot_clines import plot_clines
from .plot_heatmap import plot_heatmap
from .plot_line import plot_line
from .plot_magnetosphere import plot_magnetosphere
from .plot_projection import plot_projection
from .plot_reduced_2d import plot_reduced_2d
from .plot_spectr import plot_spectr
from .plot_surf import plot_surf
from .span_tint import span_tint
from .zoom import zoom

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


__all__ = [
    "add_position",
    "annotate_heatmap",
    "colorbar",
    "make_labels",
    "mms_pl_config",
    "pl_scatter_matrix",
    "pl_tx",
    "plot_ang_ang",
    "plot_clines",
    "plot_heatmap",
    "plot_line",
    "plot_magnetosphere",
    "plot_projection",
    "plot_reduced_2d",
    "plot_spectr",
    "plot_surf",
    "set_color_cycle",
    "span_tint",
    "use_pyrfu_style",
    "zoom",
]

stylesheets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stylesheets")
style.core.USER_LIBRARY_PATHS.append(stylesheets)
style.core.reload_library()

EXTRA_COLORS = {
    "pyrfu:bg": "#eeeeee",
    "pyrfu:fg": "#444444",
    "pyrfu:bgAlt": "#e4e4e4",
    "pyrfu:red": "#af0000",
    "pyrfu:green": "#008700",
    "pyrfu:blue": "#005f87",
    "pyrfu:yellow": "#afaf00",
    "pyrfu:orange": "#d75f00",
    "pyrfu:pink": "#d70087",
    "pyrfu:purple": "#8700af",
    "pyrfu:lightblue": "#0087af",
    "pyrfu:olive": "#5f7800",
    "on:bg": "#1b2b34",
    "on:fg": "#cdd3de",
    "on:bgAlt": "#343d46",
    "on:fgAlt": "#d8dee9",
    "on:red": "#ec5f67",
    "on:orange": "#f99157",
    "on:yellow": "#fac863",
    "on:green": "#99c794",
    "on:cyan": "#5fb3b3",
    "on:blue": "#6699cc",
    "on:pink": "#c594c5",
    "on:brown": "#ab7967",
    "series:cyan": "#54c9d1",
    "series:orange": "#eca89a",
    "series:blue": "#95bced",
    "series:olive": "#ceb776",
    "series:purple": "#d3a9ea",
    "series:green": "#9bc57f",
    "series:pink": "#f0a1ca",
    "series:turquoise": "#5fcbaa",
    "transparent": "#ffffff00",
}

mpl.colors.EXTRA_COLORS = EXTRA_COLORS
mpl.colors.colorConverter.colors.update(EXTRA_COLORS)
mpl.colormaps.register(
    name="bird",
    cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "bird",
        [
            (0.0592, 0.3599, 0.8684),
            (0.078, 0.5041, 0.8385),
            (0.0232, 0.6419, 0.7914),
            (0.1802, 0.7178, 0.6425),
            (0.5301, 0.7492, 0.4662),
            (0.8186, 0.7328, 0.3499),
            (0.9956, 0.7862, 0.1968),
            (0.9764, 0.9832, 0.0539),
        ],
        N=2560,
    ),
)


def set_color_cycle(pal=None):
    r"""Sets color cycle.

    Parameters
    ----------
    pal : {"Pyrfu", "Oceanic", "Tab", None}. Optional
      The palette to use. "Tab" provides the default matplotlib palette. Default is
      None (resets to default palette).
    """

    if pal.lower() == "pyrfu":
        colors = [
            "pyrfu:blue",
            "pyrfu:green",
            "pyrfu:red",
            "pyrfu:fg",
            "pyrfu:orange",
            "pyrfu:purple",
            "pyrfu:yellow",
            "pyrfu:lightblue",
            "pyrfu:olive",
        ]
    elif pal.lower() == "oceanic":
        colors = [
            "on:green",
            "on:red",
            "on:blue",
            "on:cyan",
            "on:orange",
            "on:pink",
            "on:yellow",
        ]
    elif pal.lower() == "tab" or pal.lower() == "tableau" or pal.lower() == "mpl":
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
    else:
        colors = [
            "series:cyan",
            "series:orange",
            "series:blue",
            "series:olive",
            "series:purple",
            "series:green",
            "series:pink",
        ]

    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)


def use_pyrfu_style(
    name="classic", color_cycle="pyrfu", fancy_legend=False, usetex=False
):
    r"""Setup plot style.

    Parameters
    ----------
    name : str, Optional
       Name of the style sheet. Default is "classic" (see also
       https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
    color_cycle : str, Optional
       Name of the color cycle to use. Default is "pyrfu".
    fancy_legend : bool, Optional
       Use matplotlib's fancy legend frame. Default is False.
    usetex : bool, Optional
       Use LaTeX installation to set text. Default is False.
       If no LaTeX installation is found, this package will fallback to usetex=False.
    """

    if usetex:
        if (
            shutil.which("pdflatex") is None
            or shutil.which("dvipng") is None
            or shutil.which("gs") is None
        ):
            warnings.warn(
                "No LaTeX installation found pyrfu.plot is falling back to usetex=False"
            )
            usetex = False

    style.use(name)
    set_color_cycle(color_cycle)
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    if not usetex:
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["mathtext.sf"] = "sans"
        mpl.rcParams["mathtext.fontset"] = "dejavusans"
    else:
        mpl.rcParams["text.latex.preamble"] = "\n".join(
            [
                r"\usepackage{amsmath}",
                r"\usepackage{physics}",
                r"\usepackage{siunitx}",
                r"\usepackage{bm}",
            ]
        )
    if not fancy_legend:
        mpl.rcParams["legend.frameon"] = False
        mpl.rcParams["legend.fancybox"] = False
        mpl.rcParams["legend.framealpha"] = 0.75
