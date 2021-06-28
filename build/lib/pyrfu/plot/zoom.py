#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from matplotlib.transforms import TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import (BboxPatch, BboxConnector,
                                                   BboxConnectorPatch)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _connect_bbox(bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, prop_lines,
                  prop_patches: dict = None):
    if prop_patches is None:
        prop_patches = {**prop_lines, "alpha": prop_lines.get("alpha", 1) * 0}

    connector_a = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a,
                                **prop_lines)
    connector_a.set_clip_on(False)
    connector_b = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b,
                                **prop_lines)
    connector_b.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    connector_patch = BboxConnectorPatch(bbox1, bbox2, loc1a=loc1a,
                                         loc2a=loc2a, loc1b=loc1b,
                                         loc2b=loc2b, **prop_patches)

    connector_patch.set_clip_on(False)

    return connector_a, connector_b, bbox_patch1, bbox_patch2, connector_patch


def zoom(ax1, ax2, **kwargs):
    r"""Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.

    Parameters
    ----------
    ax1 : matplotlib.pyplot.subplotsaxes
        Reference axes.
    ax2 : matplotlib.pyplot.subplotsaxes
        Connected axes.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments control the BboxPatch.

    Returns
    -------
    ax1 : matplotlib.pyplot.subplotsaxes
        Reference axis.
    ax2 : matplotlib.pyplot.subplotsaxes
        Connected axis.

    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    bbox_1 = ax1.bbox
    bbox_2 = TransformedBbox(ax1.viewLim, trans)

    c1, c2, p1, p2, p = _connect_bbox(bbox_1, bbox_2, loc1a=2, loc2a=3,
                                      loc1b=1, loc2b=4, prop_lines=kwargs)

    ax1.add_patch(p1)
    ax2.add_patch(p2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return ax1, ax2
