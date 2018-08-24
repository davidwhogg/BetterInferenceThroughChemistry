"""
This file is part of the ChemicalTangents project
copyright 2018 David W. Hogg (NYU) (MPIA) (Flatiron)

notes:
------
- Just the likelihood function and some inference and plotting
  routines.

bugs / to-dos:
--------------
- Make it possible to use multiple abundances simultaneously.
- Make it easy / possible to plot residuals away from the best-fit
  (or any chosen) model.
"""

import numpy as np
from scipy.misc import logsumexp
from integrate_orbits import *

def ln_like(qs, invariants, order=3):
    """
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
    - impossibly complex list comprehension
    """
    priorvars = np.exp(np.arange(np.log(0.02), np.log(0.25), np.log(1.01)))
    lndprior = -1. * np.log(len(priorvars))
    AT = np.vstack([invariants ** k for k in range(order+1)])
    A = AT.T
    ATA = np.dot(AT, A)
    x = np.linalg.solve(ATA, np.dot(AT, qs))
    foo, lnATA = np.linalg.slogdet(ATA)
    summed_likelihood = logsumexp([-0.5 * np.sum((qs - np.dot(A, x)) ** 2 / var + np.log(var))
                                    - 0.5 * (lnATA - np.log(var)) for var in priorvars])
    return lndprior + summed_likelihood

def ln_prior(pars):
    """
    such bad code
    """
    if pars[0] < -50.:
        return -np.Inf
    if pars[0] > 50.:
        return -np.Inf
    if pars[1] < -10.:
        return -np.Inf
    if pars[1] > 10.:
        return -np.Inf
    if pars[2] < np.log(20.):
        return -np.Inf
    if pars[2] > np.log(180.):
        return -np.Inf
    if pars[3] < np.log(100.):
        return -np.Inf
    if pars[3] > np.log(1000.):
        return -np.Inf
    return 0.

def ln_post(pars, kinematicdata, elementdata, abundances=["fe_h", "mg_fe", ]):
    """
    comments:
    - This function unpacks the pars, creates the invariants out of
      the data, extracts the relevant abundances, and computes the
      posterior on everything.
    - Assumes that the `kinematicdata` input has various methods
      defined.
    - Note the `exp()` on the `dynpars`.
    """
    ln_p = ln_prior(pars)
    if not np.isfinite(ln_p):
        return -np.Inf
    sunpars = pars[:2]
    dynpars = np.exp(pars[2:])
    zs = kinematicdata.z.to(u.pc).value
    vs = kinematicdata.v_z.to(u.km/u.s).value
    Jzs, phis, blob = paint_actions_angles(zs, vs, sunpars0, dynpars0)
    invariants = Jzs - mean(Jzs)
    ln_l = 0.
    for abundance in abundances:
        metals = getattr(elementdata, abundance)
        okay = (metals > -2.) & (metals < 2.) # HACKY
        ln_l += ln_like(metals[okay], invariants[okay])
    return ln_p + ln_l

def sample(pars0):
    return 0.
