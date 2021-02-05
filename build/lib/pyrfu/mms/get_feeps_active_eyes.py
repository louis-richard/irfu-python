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

import datetime

from dateutil import parser


def get_feeps_active_eyes(var, tint, mms_id):
    """This function returns the FEEPS active eyes, based on date/mms_id/species/rate.
    
    Parameters
    ----------
    var : dict
        Hash table containing parameters.

    tint : list of str
        Time range.

    mms_id : int or str
        mms_id e.g., '4' for MMS4.

    Returns
    -------
    sensors : dict
        Hash table containing 2 keys :
            * out["top"] : maps to the active top eyes.
            * out["bottom"] : maps to the active bottom eyes.

    Notes
    -----
    1) Burst mode should include all sensors (TOP and BOTTOM) :
        * electrons : [1, 2, 3, 4, 5, 9, 10, 11, 12].
        * ions : [6, 7, 8].

    2) SITL should return (TOP only) :
        * electrons : set_intersection([5, 11, 12], active_eyes).
        * ions : None.

    3) From Drew Turner, 9/7/2017, srvy mode :
        * before 16 August 2017 :
            * electrons : [3, 4, 5, 11, 12].
            * ion : [6, 7, 8].

        * after 16 August 2017 :
            * MMS1 :
                * Top Eyes : 3, 5, 6, 7, 8, 9, 10, 12
                * Bot Eyes : 2, 4, 5, 6, 7, 8, 9, 10
            * MMS2 :
                * Top Eyes : 1, 2, 3, 5, 6, 8, 10, 11
                * Bot Eyes : 1, 4, 5, 6, 7, 8, 9, 11
            * MMS3
                * Top Eyes : 3, 5, 6, 7, 8, 9, 10, 12
                * Bot Eyes : 1, 2, 3, 6, 7, 8, 9, 10
            * MMS4 :
                * Top Eyes : 3, 4, 5, 6, 8, 9, 10, 11
                * Bot Eyes : 3, 5, 6, 7, 8, 9, 10, 12

    """

    if isinstance(mms_id, str):
        mms_id = int(mms_id)

    sensors = {}

    if var["tmmode"].lower() == "brst" and var["dtype"].lower() == "electron": 
        sensors["top"] = [1, 2, 3, 4, 5, 9, 10, 11, 12]
        sensors["bottom"] = [1, 2, 3, 4, 5, 9, 10, 11, 12]

    if var["tmmode"].lower() == "brst" and var["dtype"].lower() == "ion": 
        sensors["top"] = [6, 7, 8]
        sensors["bottom"] = [6, 7, 8]

    # old eyes, srvy mode, prior to 16 August 2017
    if var["dtype"].lower() == "electron":
        sensors["top"] = [3, 4, 5, 11, 12]
        sensors["bottom"] = [3, 4, 5, 11, 12]
    else:
        sensors["top"] = [6, 7, 8]
        sensors["bottom"] = [6, 7, 8]

    if isinstance(tint[0], str): 
        start_time = parser.parse(tint[0])
    else:
        start_time = tint[0]

    # srvy mode, after 16 August 2017
    if start_time >= datetime.datetime(2017, 8, 16) and var["tmmode"].lower() == "srvy":
        active_table = {"1-electron": {}, "1-ion": {}, "2-electron": {}, "2-ion": {},
                        "3-electron": {}, "3-ion": {}, "4-electron": {}, "4-ion": {}}

        active_table["1-electron"]["top"] = [3, 5, 9, 10, 12]
        active_table["1-electron"]["bottom"] = [2, 4, 5, 9, 10]

        active_table["1-ion"]["top"] = [6, 7, 8]
        active_table["1-ion"]["bottom"] = [6, 7, 8]

        active_table["2-electron"]["top"] = [1, 2, 3, 5, 10, 11]
        active_table["2-electron"]["bottom"] = [1, 4, 5, 9, 11]

        active_table["2-ion"]["top"] = [6, 8]
        active_table["2-ion"]["bottom"] = [6, 7, 8]

        active_table["3-electron"]["top"] = [3, 5, 9, 10, 12]
        active_table["3-electron"]["bottom"] = [1, 2, 3, 9, 10]

        active_table["3-ion"]["top"] = [6, 7, 8]
        active_table["3-ion"]["bottom"] = [6, 7, 8]

        active_table["4-electron"]["top"] = [3, 4, 5, 9, 10, 11]
        active_table["4-electron"]["bottom"] = [3, 5, 9, 10, 12]

        active_table["4-ion"]["top"] = [6, 8]
        active_table["4-ion"]["bottom"] = [6, 7, 8]
       
        sensors = active_table["{:d}-{}".format(mms_id, var["dtype"].lower())]
        
        if var["lev"].lower() == "sitl":
            sensors["top"] = list(set(sensors["top"]) & {5, 11, 12})
            sensors["bottom"] = []
            return {"top": list(set(sensors["top"]) & {5, 11, 12}), "bottom": []}

    if var["lev"].lower() == "sitl":
        sensors["top"] = [5, 11, 12]
        sensors["bottom"] = []
        
    return sensors
