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

import os
import shutil
import requests

from PIL import Image
from dateutil import parser as date_parser


def current_location(time, view: str = "xy", path: str = ".",
                     show: bool = False):
    """Download MMS location plots from MMS SDC.

    Parameters
    ----------
    time : str
        Time at which the location is wanted.

    view : str
        View mode xy or xz.

    path : str
        Path where to save images.

    show : bool
        Flag to show orbit plot.

    """

    # URL of the MMS Science Data Center (SDC)
    sdc_url = "https://lasp.colorado.edu/mms/sdc/public/data/sdc"

    # Date string format of time (needs only up to hours)
    str_time = date_parser.parse(time).strftime("%Y%m%d%H0000")

    # Name of the directory and the image corresponding to the view mode
    if view.lower() == "xy":
        dir_path = "mms_orbit_plots"
        img_name = "mms_orbit_plot_{}.png".format(str_time)
    elif view.lower() == "xz":
        dir_path = "mms_orbit_plots"
        img_name = "mms_orbit_plot_{}_{}.png".format(view, str_time)
    else:
        raise ValueError("Invalid view mode")

    # Create directory in the target path
    os.makedirs(os.path.join(path, dir_path), exist_ok=True)

    # Full url and path of the orbit plot
    img_url, img_path = [os.path.join(root, dir_path, img_name)
                         for root in [sdc_url, path]]

    # Download image
    response = requests.get(img_url, stream=True)

    with open(img_path, "wb") as file:
        shutil.copyfileobj(response.raw, file)

    del response

    # Show the plot
    if show:
        img = Image.open(img_path)
        img.show()
