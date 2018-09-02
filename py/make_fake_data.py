"""
This file is part of the Chemical Tangents project
Copyright 2018 David W. Hogg (NYU)

to-do
-----
- write code
- test code
"""

import numpy as np
from chemical_tangents import *

def make_one_fake_star(pars):
    """
    bugs:
    -----
    - not written
    - not adding noise to anything
    """
    # unpack pars
    foo, a, b, c, sunpars, dynpars = pars
    # choose x,y
    x, y = np.random.uniform(-1000., 1000., size=2) # pc
    # choose Ez and metallicity
    lnEz = foo + np.log(np.random.normal() ** 2)
    abundnace = a + b * lnEz + c * lnEz * lnEz + 0.25 * np.random.normal()
    # choose an angle
    phi_z = 2. * np.pi * np.random.uniform()
    # integrate
    v0 = np.sqrt(2. * np.exp(lnEz))
    z, vz = integrate_to_phi(phi_z, v0)
    return x, y, z, vz

def selection_function(x, y, z):
    """
    evaluate selection function at a point.

    bugs:
    -----
    - need to add sky cuts
    """
    if x * x + y * y + z * z > 1000. * 1000.:
        return 0.
    return 1.

def make_catalog():
    N = 5000
    n = 0
    poss, vzs = np.zeros((N, 3)), np.zeros(N)
    while n < N:
        r = np.random.uniform()
        x, y, z, vz = make_one_fake_star(pars)
        if r < selection_function(x, y, z):
            poss[n, :] = x, y, z
            vzs[n] = vz
            n += 1

def write_catalog():
    whatever
    return None
