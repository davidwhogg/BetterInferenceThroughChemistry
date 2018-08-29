# Third-party
import astropy.units as u
from astropy.constants import G
import numpy as np
from gala.units import UnitSystem
from scipy.special import logsumexp

# This package
from .potential import sech2_potential
from .data import get_abundance_data

__all__ = ['Model']

class Model:

    def __init__(self, galcen, element_data, abundance_names,
                 metals_deg=3, usys=None):
        """TODO

        Parameters
        ----------
        galcen : `astropy.coordinates.CartesianRepresentation`
        element_data : TODO
        abundance_names : iterable
        metals_deg : int (optional)
            The degree of the polynomial used to represent the abundance
            distribution mean as a function of invariants.
        usys : `gala.units.UnitSystem` (optional)
            The unit system to work in. Defaults to (pc, Myr, Msun, km/s).

        """

        # Make sure the unit system has a default
        if usys is None:
            usys = UnitSystem(u.pc, u.Myr, u.Msun, u.radian, u.km/u.s)
        self.usys = usys

        # kinematic and abundance data
        self.galcen = galcen

        # put requested abundances into a dictionary
        self.element_data = dict()
        for name in abundance_names:
            self.element_data[name] = get_abundance_data(element_data, name)
        self.abundance_names = abundance_names

        # Convert data to units we expect
        self._z = galcen.z.decompose(self.usys).value
        self._vz = galcen.v_z.decompose(self.usys).value

        # Define a dummy potential object
        # HACK: hard-coded for now!
        self._potential = sech2_potential
        self._G = G.decompose(self.usys).value

        # Cache array used below
        self.metals_deg = int(metals_deg)
        self._AT = np.ones((self.metals_deg+1, len(galcen)))

    def unpack_pars(self, p):
        par_dict = dict()
        par_dict['sun_z'] = p[0]
        par_dict['sun_vz'] = p[1]
        par_dict['lnsigma'] = p[2]
        par_dict['lnhz'] = p[3]
        return par_dict

    def ln_prior(self, par_dict):

        if not -64 < par_dict['sun_z'] < 64: # pc
            return -np.inf

        if not -8 < par_dict['sun_vz'] < 8: # pc/Myr
            return -np.inf

        if not np.log(16) < par_dict['lnsigma'] < np.log(256): # ln(Msun/pc^2)
            return -np.inf

        if not np.log(32) < par_dict['lnhz'] < np.log(1024): # ln(pc)
            return -np.inf

        return 0.

    def ln_metal_likelihood(self, qs, invariants):
        # TODO: hard-coded
        priorvars = np.exp(np.arange(np.log(0.02), np.log(0.25), np.log(1.01)))
        lndprior = -np.log(len(priorvars))

        for k in range(1, self.metals_deg+1):
            self._AT[k] = self._AT[k-1] * invariants

        AT = self._AT
        A = AT.T
        ATA = np.dot(AT, A)
        x = np.linalg.solve(ATA, np.dot(AT, qs))

        # TODO: re-enable this
        # if residuals:
        #     return qs - np.dot(A, x)

        _, lnATA = np.linalg.slogdet(ATA)
        resid2sum = np.sum((qs - np.dot(A, x)) ** 2)

        nobj = len(qs)
        summed_likelihood = -0.5 * logsumexp((resid2sum / priorvars + nobj * np.log(priorvars)) +
                                             (lnATA - np.log(priorvars)))
        return lndprior + summed_likelihood

    def get_energy(self, par_dict):
        z = self._z + par_dict['sun_z']
        vz = self._vz + par_dict['sun_vz']

        Sigma = np.exp(par_dict['lnsigma'])
        hz = np.exp(par_dict['lnhz'])
        Es = 0.5*vz**2 + self._potential(z, Sigma, hz, self._G)

        return Es

    def ln_likelihood(self, par_dict):
        Es = self.get_energy(par_dict)
        invariants = Es - np.mean(Es)

        ln_l = 0.
        for abundance_name in self.abundance_names:
            metals = self.element_data[abundance_name]
            ln_l += self.ln_metal_likelihood(metals, invariants)

        return ln_l

    def ln_posterior(self, p):
        par_dict = self.unpack_pars(p)

        lnp = self.ln_prior(par_dict)
        if not np.isfinite(lnp):
            return -np.inf

        lnl = self.ln_likelihood(par_dict)
        if not np.isfinite(lnl):
            return -np.inf

        return lnp + lnl

    def __call__(self, *args, **kwargs):
        return self.ln_posterior(*args, **kwargs)
