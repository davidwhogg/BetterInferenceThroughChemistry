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

def ln_normal(x, mu, var):
    return -0.5 * (np.log(2*np.pi * var) + (x-mu)**2 / var)


class Model:

    def __init__(self, galcen, element_data, abundance_names,
                 frozen_pars=None, usys=None,
                 metals_deg=3,
                 marginalize=None):
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

        self.marginalize_all = False
        self.marginalize_alpha = False
        if marginalize == 'all':
            self.marginalize_all = True
        elif marginalize == 'alpha':
            self.marginalize_alpha = True

        # kinematic and abundance data
        self.galcen = galcen

        # put requested abundances into a dictionary
        self.element_data = dict()
        for name in abundance_names:
            self.element_data[name] = get_abundance_data(element_data, name)
        self.abundance_names = abundance_names

        # if enforce_finite:
        #     elem_mask = np.ones(len(galcen), dtype=bool)
        #     for name in abundance_names:
        #         elem_mask &= np.isfinite(self.element_data[name])
        #
        #     for name in abundance_names:
        #         self.element_data[name] = self.element_data[name][elem_mask]
        #     self.galcen = self.galcen[elem_mask]

        if frozen_pars is None:
            frozen_pars = dict()
        self.frozen_pars = frozen_pars

        # Convert data to units we expect
        self._z = self.galcen.z.decompose(self.usys).value
        self._vz = self.galcen.v_z.decompose(self.usys).value

        # TODO HACK: hard-coded for now!
        self._potential = sech2_potential
        self._G = G.decompose(self.usys).value

        # Cache array used below
        self.metals_deg = int(metals_deg)
        self._AT = np.ones((self.metals_deg+1, len(self.galcen)))

    def unpack_pars(self, p):
        par_dict = dict()

        p_i = 0

        for par in ['sun_z', 'sun_vz', 'lnsigma', 'lnhz']:
            par_dict[par] = self.frozen_pars.get(par, p[p_i])
            if par not in self.frozen_pars:
                p_i += 1

        if self.marginalize_alpha:
            par_dict['lnvar'] = dict()
            for k, name in enumerate(self.abundance_names):
                lnvar = p[p_i]
                par_dict['lnvar'][name] = lnvar
                p_i += 1

        elif not self.marginalize_all:
            par_dict['alpha'] = dict()
            par_dict['lnvar'] = dict()
            for k, name in enumerate(self.abundance_names):
                lnvar = p[p_i]
                p_i += 1

                alpha = p[p_i:p_i+self.metals_deg+1]
                p_i += self.metals_deg+1

                par_dict['lnvar'][name] = lnvar
                par_dict['alpha'][name] = alpha

        return par_dict

    def ln_prior(self, par_dict):

        lp = 0.

        if 'sun_z' not in self.frozen_pars: # pc
            if not -128 < par_dict['sun_z'] < 128:
                return -np.inf

        if 'sun_vz' not in self.frozen_pars: # pc/Myr
            if not -32 < par_dict['sun_vz'] < 32:
                return -np.inf

        if 'lnsigma' not in self.frozen_pars: # ln(Msun/pc^2)
            if not np.log(16) < par_dict['lnsigma'] < np.log(256):
                return -np.inf

        if 'lnhz' not in self.frozen_pars: # ln(pc)
            if not np.log(32) < par_dict['lnhz'] < np.log(512):
                return -np.inf

        if self.marginalize_alpha or not self.marginalize_all:
            for abundance_name in self.abundance_names:
                lnvar = par_dict['lnvar'][abundance_name]
                if not 2*np.log(0.04) < lnvar < 2*np.log(0.5):
                    return -np.inf

        # TODO: prior on alpha??
        # alpha = par_dict['alpha'][abundance_name]

        return lp

    def get_residual(self, X_Ys, invariants):
        for k in range(1, self.metals_deg+1):
            self._AT[k] = self._AT[k-1] * invariants

        AT = self._AT
        A = AT.T
        ATA = np.dot(AT, A)
        x = np.linalg.solve(ATA, np.dot(AT, X_Ys))

        resid = X_Ys - np.dot(A, x)

        return resid, ATA

    def ln_metal_fully_marginal_likelihood(self, X_Ys, invariants):
        """ TODO: """
        # TODO: hard-coded variance values!
        priorvars = np.logspace(np.log10(0.1**2), np.log10(0.5**2), 1024)
        lndprior = -np.log(len(priorvars))

        resid, ATA = self.get_residual(X_Ys, invariants)
        _, lnATA = np.linalg.slogdet(ATA)
        resid2sum = np.sum(resid**2)

        nobj = len(X_Ys)
        summed_likelihood = -0.5 * logsumexp((resid2sum / priorvars + nobj * np.log(priorvars)) +
                                             (lnATA - np.log(priorvars)))
        return lndprior + summed_likelihood

    def ln_metal_marginal_likelihood(self, var, X_Ys, invariants):
        """ TODO: this currently evaulates incorrectly!!

        Parameters
        ----------
        var : numeric
        X_Ys : `numpy.ndarray`
        invariants : `numpy.ndarray`
        """
        resid, ATA = self.get_residual(X_Ys, invariants)
        _, lnATA = np.linalg.slogdet(ATA)
        return -0.5 * (np.sum(resid**2) / var + np.log(2*np.pi * var) + lnATA)

    def ln_metal_likelihood(self, alpha, var, X_Ys, invariants):
        """Compute the un-marginalized likelihood for our model.

        Parameters
        ----------
        alpha : iterable
        var : numeric
        X_Ys : `numpy.ndarray`
        invariants : `numpy.ndarray`
        """
        mu = np.poly1d(alpha)(invariants)
        return ln_normal(X_Ys, mu, var).sum()

    def get_energy(self, par_dict):
        z = self._z + par_dict['sun_z']
        vz = self._vz + par_dict['sun_vz']

        Sigma = np.exp(par_dict['lnsigma'])
        hz = np.exp(par_dict['lnhz'])
        Es = 0.5*vz**2 + self._potential(z, Sigma, hz, self._G)

        return Es

    def ln_likelihood(self, par_dict):
        # Es = self.get_energy(par_dict).value / 1000.
        Es = np.log(self.get_energy(par_dict) / 1000.)
        invariants = Es - np.mean(Es)

        ln_l = 0.
        for abundance_name in self.abundance_names:
            metals = self.element_data[abundance_name]

            if self.marginalize_alpha:
                lnvar = par_dict['lnvar'][abundance_name]
                ln_l += self.ln_metal_marginal_likelihood(np.exp(lnvar),
                                                          metals, invariants)

            elif self.marginalize_all:
                ln_l += self.ln_metal_fully_marginal_likelihood(metals,
                                                                invariants)

            else:
                alpha = par_dict['alpha'][abundance_name]
                lnvar = par_dict['lnvar'][abundance_name]
                ln_l += self.ln_metal_likelihood(alpha, np.exp(lnvar),
                                                 metals, invariants)

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
