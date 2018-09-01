# Standard library
from collections import OrderedDict
from math import pi

# Third-party
import numpy as np
from gala.potential import PotentialBase

__all__ = ['sech2_potential', 'sech2_gradient', 'sech2_density',
           'Sech2Potential',
           'uniform_potential', 'uniform_gradient', 'uniform_density',
           'UniformPotential']

# ------------------------------------------------------------------------------
# Sech^2 density stuff
def sech2_potential(z, sigma, hz, G):
    return 4*pi*G * sigma * hz * np.log(np.cosh(0.5 * z / hz))

def sech2_gradient(z, sigma, hz, G):
    return 2*pi*G * sigma * np.tanh(0.5 * z / hz)

def sech2_density(z, sigma, hz):
    return sigma / (4 * hz) / np.cosh(0.5 * z / hz)**2

class Sech2Potential(PotentialBase):

    def __init__(self, Sigma, hz, units=None, origin=None):
        params = OrderedDict()
        ptypes = OrderedDict()

        params['Sigma'] = Sigma
        ptypes['Sigma'] = 'surface density'

        params['hz'] = hz
        ptypes['hz'] = 'length'

        super(Sech2Potential, self).__init__(parameters=params,
                                             parameter_physical_types=ptypes,
                                             units=units,
                                             origin=origin,
                                             ndim=1)

    def _energy(self, q, t=0.):
        return sech2_potential(q[:, 0],
                               self.parameters['Sigma'].value,
                               self.parameters['hz'].value,
                               self.G)

    def _gradient(self, q, t=0.):
        return sech2_gradient(q[:, 0],
                              self.parameters['Sigma'].value,
                              self.parameters['hz'].value,
                              self.G)

    def _density(self, q, t=0.):
        return sech2_density(q[:, 0],
                             self.parameters['Sigma'].value,
                             self.parameters['hz'].value)


# ------------------------------------------------------------------------------
# uniform density stuff
def uniform_potential(z, rho0, G):
    return 2*pi*G * rho0 * z**2

def uniform_gradient(z, rho0, G):
    return 4*pi*G * rho0 * z

def uniform_density(z, rho0):
    return rho0

class UniformPotential(PotentialBase):

    def __init__(self, rho0, units=None, origin=None):
        params = OrderedDict()
        ptypes = OrderedDict()

        params['rho0'] = rho0
        ptypes['rho0'] = 'mass density'

        super(UniformPotential, self).__init__(parameters=params,
                                               parameter_physical_types=ptypes,
                                               units=units,
                                               origin=origin,
                                               ndim=1)

    def _energy(self, q, t=0.):
        return uniform_potential(q[:, 0],
                                 self.parameters['rho0'].value,
                                 self.G)

    def _gradient(self, q, t=0.):
        return uniform_gradient(q[:, 0],
                                self.parameters['rho0'].value,
                                self.G)

    def _density(self, q, t=0.):
        return uniform_density(q[:, 0],
                               self.parameters['rho0'].value)
