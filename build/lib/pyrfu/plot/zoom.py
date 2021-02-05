#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

from matplotlib.transforms import TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def connect_bbox(bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, prop_lines, prop_patches=None):
    """to fill
    """
    if prop_patches is None:
        prop_patches = {**prop_lines, "alpha": prop_lines.get("alpha", 1) * 0}

    connector_a = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    connector_a.set_clip_on(False)
    connector_b = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    connector_b.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    connector_patch = BboxConnectorPatch(bbox1, bbox2, loc1a=loc1a, loc2a=loc2a, loc1b=loc1b,
                                         loc2b=loc2b, **prop_patches)

    connector_patch.set_clip_on(False)

    return connector_a, connector_b, bbox_patch1, bbox_patch2, connector_patch


def zoom(ax1, ax2, **kwargs):
    """Similar to zoom_effect01.  The xmin & xmax will be taken from the ax1.viewLim.

    Parameters
    ----------
    ax1 : axs
        The main axes.

    ax2 : axs
        The zoomed axes.

    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(mybbox1, mybbox2, loc1a=2, loc2a=3,
                                                       loc1b=1, loc2b=4, prop_lines=kwargs)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return ax1, ax2
