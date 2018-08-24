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
