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

import os

from .mms_config import CONFIG
from .list_files import list_files


def copy_files(var, tint, mms_id, target_dir="./data/"):
    """Copy files from NAS24 to the target path

    Parameters
    ----------
    var : dict
        Dictionary containing 4 keys
            * var["inst"] : name of the instrument
            * var["tmmode"] : data rate
            * var["lev"] : data level
            * var["dtype"] : data type

    tint : list of str
        Time interval

    mms_id : str or int
        Index of the spacecraft

    target_dir : str, optional
        Target path. Default is './data/'

    """

    mms_path = CONFIG["local_data_dir"] + "/"

    files = list_files(tint, mms_id, var)

    for file in files:
        relative_path = os.path.split(file)[0].replace(mms_path, "")
        path = os.path.join(target_dir, relative_path)
        target_file = os.path.join(path, os.path.split(file)[1])

        if not os.path.exists(path):
            os.makedirs(path)

        os.popen('cp {} {}'.format(file, target_file))
