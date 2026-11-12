#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Rectangle of the colorbar (figure coordinates) on each side of an axis at
# position p, pad away and w thick
_RECTS = {
    "right": lambda p, pad, w: [p.x0 + p.width + pad, p.y0, w, p.height],
    "left": lambda p, pad, w: [p.x0 - pad - w, p.y0, w, p.height],
    "top": lambda p, pad, w: [p.x0, p.y0 + p.height + pad, p.width, w],
    "bottom": lambda p, pad, w: [p.x0, p.y0 - pad - w, p.width, w],
}


def colorbar(
    mappable, axis, pad: float = 0.01, width: float = 0.03, where: str = "right"
):
    r"""Add a colorbar of `mappable` on a side of `axis` (the colorbar of the
    plot routines).

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The mappable described by this colorbar, e.g. the
        matplotlib.collections.QuadMesh returned by pcolormesh.
    axis : matplotlib.axes.Axes
        Axis of the plot.
    pad : float, Optional
        Gap between the axis and the colorbar, in figure coordinates. Default
        is 0.01.
    width : float, Optional
        Width of the colorbar, in figure coordinates. Default is 0.03.
    where : {"right", "left", "top", "bottom"}, Optional
        Side of `axis`, with the ticks and label of the colorbar on the outer
        side. Default is "right".

    Returns
    -------
    cax : matplotlib.axes.Axes
        Axis of the colorbar, added to the figure of `axis` (the
        matplotlib.colorbar.Colorbar is ``mappable.colorbar``).

    Raises
    ------
    ValueError
        If `where` is not "right", "left", "top" or "bottom".

    """
    if where not in _RECTS:
        raise ValueError(f"where must be right, left, top or bottom, not {where!r}")

    fig = axis.get_figure()
    cax = fig.add_axes(_RECTS[where](axis.get_position(), pad, width))
    vertical = where in ("right", "left")
    fig.colorbar(
        mappable, cax=cax, orientation="vertical" if vertical else "horizontal"
    )

    side = cax.yaxis if vertical else cax.xaxis
    side.set_ticks_position(where)
    side.set_label_position(where)

    return cax
