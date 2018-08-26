# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

# Standard library
from collections import OrderedDict

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from gala.potential.potential.cpotential cimport (CPotentialWrapper,
                                                  energyfunc,
                                                  gradientfunc)
from gala.potential.potential.cpotential import CPotentialBase
from gala.units import galactic

cdef extern from "src/potential.h":
    double sech2_energy(double t, double *pars, double *q, int n_dim) nogil
    void sech2_gradient(double t, double *pars, double *q, int n_dim,
                        double *grad) nogil

__all__ = ['Sech2Potential']

cdef class Sech2Wrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0), n_dim=1)
        self.cpotential.value[0] = <energyfunc>(sech2_energy)
        self.cpotential.gradient[0] = <gradientfunc>(sech2_gradient)

class Sech2Potential(CPotentialBase):
    r"""
    Sech2Potential(rho0, z0, units=None, origin=None)

    Parameters
    ----------
    rho0 : :class:`~astropy.units.Quantity`, numeric [mass density]
        Mass density at midplane.
    z0 : :class:`~astropy.units.Quantity`, numeric [length]
        Scale height.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, rho0, z0, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['rho0'] = rho0
        ptypes['rho0'] = 'mass density'

        parameters['z0'] = z0
        ptypes['z0'] = 'length'

        super(Sech2Potential, self).__init__(parameters=parameters,
                                             parameter_physical_types=ptypes,
                                             units=units,
                                             origin=origin,
                                             Wrapper=Sech2Wrapper,
                                             ndim=1)
