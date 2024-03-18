#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools
from typing import List, Union

# 3rd party imports
import numpy as np
from matplotlib import ticker

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def annotate_heatmap(
    im,
    data: Union[np.ndarray, List[List[float]]] = None,
    valfmt: str = "{x:.2f}",
    textcolors: tuple = ("black", "white"),
    threshold: float = None,
    **textkw,
):
    r"""Annotate a heatmap.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        The AxesImage to be labeled.
    data : array_like, Optional
        Data used to annotate.  If None, the image's data is used.
    valfmt : str or matplotlib.ticker.Formatter, Optional
        The format of the annotations inside the heatmap..
    textcolors : dict, Optional
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.
    threshold : float, Optional
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.
    **textkw
        All other arguments are forwarded to each call to `text` used to create
        the text labels.

    Returns
    -------
    texts : list
        Cells labels

    """

    if data is None:
        data = im.get_array()

    if not isinstance(data, (np.ndarray, list)):
        raise TypeError("data must be a numpy array")

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = {"horizontalalignment": "center", "verticalalignment": "center"}
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
        kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
        text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
        texts.append(text)

    return texts
