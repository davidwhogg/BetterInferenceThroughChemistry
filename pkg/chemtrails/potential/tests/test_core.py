# Standard library
import os

# Third-party
from astropy.constants import G
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest
from pyia import GaiaData
from scipy.misc import derivative

# This package
from ..core import (sech2_density, sech2_gradient, sech2_potential)

def test_sech2():
    Sigma = 65. * u.Msun/u.pc**2
    hz = 250 * u.pc
    _G = G.decompose([u.pc, u.Msun, u.Myr]).value
    rho0 = Sigma / (4*hz)

    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        assert quantity_allclose(sech2_density(0.*u.pc, Sigma, hz),
                                 rho0)

    # check numerical derivatives of potential against functions
    rnd = np.random.RandomState(42)
    for z0 in rnd.uniform(100, 500, size=16): # check a few random places
        num_grad = derivative(sech2_potential, z0, dx=1e-2, n=1,
                              args=(Sigma.value, hz.value, _G))
        grad = sech2_gradient(z0, Sigma.value, hz.value, _G)
        assert np.allclose(grad, num_grad)

        num_dens = derivative(sech2_potential, z0, dx=1e-2, n=2,
                              args=(Sigma.value, hz.value, _G)) / (4*np.pi*_G)
        dens = sech2_density(z0, Sigma.value, hz.value)
        assert np.allclose(dens, num_dens)
