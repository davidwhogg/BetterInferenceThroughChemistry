"""
This file is part of the ChemicalTangents project
copyright 2018 David W. Hogg (NYU) (MPIA) (Flatiron)

notes:
------
- Just the likelihood function and some inference and plotting
  routines.

bugs / to-dos:
--------------
- Make it easy / possible to plot residuals away from the best-fit
  (or any chosen) model.
- NEED TO SWITCH to using Astropy units correctly.
"""

import numpy as np
from scipy.misc import logsumexp
import astropy.units as u
import emcee
import corner
from integrate_orbits import *

def ln_like(qs, invariants, order=3, residuals=False):
    """
    inputs:
    - `qs`: a set of scalar abundnaces for a set of stars
    - `invariants`: a set of dynamical invariants (one component, like Jz) for the same stars
    - `order` (default 3): order of the polynomial model for the abundance mean
    - `residuals` (default False): return residuals instead of the likelihood
    
    output:
    - The fully marginalized log likelihood.
    - OR, `if residuals`, the residuals away from the mean prediction (for visualization purposes)

    comments:
    - This function has a `posteriorlnvar` addition that approximates
      the relevant marginalization over the `order+1` linear
      parameters.
    - It is best to call this with something like `invariants -
      mean(invariants)` so the pivot for the fitting is stable.

    bugs:
    - prior var grid hard-coded
    - possible sign issue with posteriorlnvar
    - can't compare values taken at different orders bc priors not
      proper
    - FORMULAE NEED CHECKING: lots of 0.5s and signs and so on
    """
    priorvars = np.exp(np.arange(np.log(0.02), np.log(0.25), np.log(1.01)))
    lndprior = -1. * np.log(len(priorvars))
    AT = np.vstack([invariants ** k for k in range(order+1)])
    A = AT.T
    ATA = np.dot(AT, A)
    x = np.linalg.solve(ATA, np.dot(AT, qs))
    if residuals:
        return qs - np.dot(A, x)
    foo, lnATA = np.linalg.slogdet(ATA)
    resid2sum = np.sum((qs - np.dot(A, x)) ** 2)
    nobj = len(qs)
    summed_likelihood = -0.5 * logsumexp((resid2sum / priorvars + nobj * np.log(priorvars))
                                         + (lnATA - np.log(priorvars)))
    return lndprior + summed_likelihood

def ln_prior(pars):
    """
    such bad code
    """
    if pars[0] < -20.:
        return -np.Inf
    if pars[0] > 20.:
        return -np.Inf
    if pars[1] < -5.:
        return -np.Inf
    if pars[1] > 5.:
        return -np.Inf
    if pars[2] < np.log(32.):
        return -np.Inf
    if pars[2] > np.log(128.):
        return -np.Inf
    if pars[3] < np.log(200.):
        return -np.Inf
    if pars[3] > np.log(600.):
        return -np.Inf
    return 0.

def ln_post(pars, kinematicdata, elementdata, abundances):
    """
    comments:
    - This function unpacks the pars, creates the invariants out of
      the data, extracts the relevant abundances, and computes the
      posterior on everything.
    - Assumes that the `kinematicdata` input has various methods
      defined that return z in pc and vz in km / s
    - Note the `exp()` on the `dynpars`.
    """
    ln_p = ln_prior(pars)
    if not np.isfinite(ln_p):
        return -np.Inf
    sunpars = np.array([pars[0] * pc, pars[1] * km / s]) # units insanity
    dynpars = np.array([np.exp(pars[2]) * sigunits, np.exp(pars[3]) * pc]) # units insanity
    zs = kinematicdata.z * pc
    vs = kinematicdata.vz * km / s
    Es = paint_energies(zs, vs, sunpars, dynpars)
    invariants = Es - np.mean(Es)
    ln_l = 0.
    for abundance in abundances:
        metals = getattr(elementdata, abundance)
        okay = (metals > -2.) & (metals < 2.) # HACKY
        ln_l += ln_like(metals[okay], invariants[okay])
    return ln_p + ln_l

def sample(kinematicdata, elementdata, abundances):
    """
    bugs:
    -----
    - initialization hard-coded
    - all integers hard-coded
    """
    p0 = np.array([0., 0., np.log(65.), np.log(400.), ])
    nsteps = 512
    nwalkers = 64
    ndim = len(p0)
    p0 = p0[None, :] + 0.01 * np.random.normal(size = (nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_post, args=[kinematicdata, elementdata, abundances])
    print("sample(): starting burn-in")
    pos, prob, state = sampler.run_mcmc(p0, nsteps)
    sampler.reset()
    print("sample(): starting proper run")
    sampler.run_mcmc(pos, nsteps)
    print("sample(): done")
    return sampler.flatchain

def sample_and_plot(kinematicdata, elementdata, abundances):
    chain = sample(kinematicdata, elementdata, abundances)
    figure = corner.corner(chain,
                           labels=[r"$z_\mathrm{Sun}$ (pc)",
                                   r"$v_{z\mathrm{Sun}}$ (km/s)",
                                   r"$\ln\Sigma$",
                                   r"$\ln h$", ],
                           range=[[-20., 20.],
                                  [-5., 5.],
                                  [np.log(32.), np.log(128.)],
                                  [np.log(200.), np.log(600.)], ])
    return chain, figure
