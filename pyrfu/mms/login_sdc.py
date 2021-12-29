#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import pickle
import requests
import warnings

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.12"
__status__ = "Prototype"

lasp = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/download/science"


def login_sdc(user: str, password: str):
    r"""Login to LASP colorado.
    
    Parameters
    ----------
    user : str
        Login.
    password : str
        Password.
        
    Returns
    -------
    session : requests.sessions.Session
        Login session.

    user : str
        Login.

    """

    session = requests.Session()
    session.auth = (user, password)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        _ = session.post("https://lasp.colorado.edu", verify=True, timeout=5)
        testget = session.get(lasp, verify=True, timeout=5)
        
    assert testget != "401", "Login failed!!"
    
    if user:
        pkg_path = os.path.dirname(os.path.abspath(__file__))
        auth_file = os.path.join(pkg_path, "mms", "mms_auth_login.pkl") 
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(auth_file, "wb") as f:
            pickle.dump({"user": user, "password": password}, f)

    return session, user
