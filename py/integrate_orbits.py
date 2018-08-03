"""
This file is part of the ChemicalTangents project
copyright 2018 David W. Hogg (NYU) (MPIA) (Flatiron)

Strictly SI units, which is INSANE
"""
import numpy as np
fourpiG = 4. * np.pi * 6.67408e-11 # m m m / kg / s / s
pc = 3.0857e16 # m
M_sun = 1.9891e30 # kg
km = 1000. # m
s = 1. # s
yr = 365.25 * 24. * 3600. * s

def leapfrog_step(z, v, dt, acceleration, pars):
    znew = z + v * dt
    dv = acceleration(znew, pars) * dt
    return znew, v + 0.5 * dv, v + dv

def leapfrog(vmax, dt, acceleration, pars):
    """
    assumes that pars[0] is the midplane location
    """
    midplane = pars[0]
    maxstep = 32768
    zs = np.zeros(maxstep) - np.Inf
    vs = np.zeros(maxstep) - np.Inf
    zs[0] = midplane
    vs[0] = vmax
    v = vs[0]
    for t in range(maxstep-1):
        zs[t + 1], vs[t + 1], v = leapfrog_step(zs[t], v, dt, acceleration, pars)
        if zs[t] < midplane and zs[t+1] >= midplane:
            fraction = (zs[t+1] - midplane) / (zs[t+1] - zs[t])
            print(zs[t], zs[t+1], fraction)
            period = dt * (t + fraction)
            phis = 2 * np.pi * np.arange(t+2) * dt / period
            break
    return zs[:t+2], vs[:t+2], phis

def pure_exponential(z, pars):
    midplane, surfacedensity, scaleheight = pars
    zprime = z - midplane
    return -1. * fourpiG * surfacedensity * (1. - np.exp(-np.abs(zprime) / scaleheight)) * np.sign(zprime)

def dummy(z, pars):
    return -1.5 * np.sign(z)

if __name__ == "__main__":
    import pylab as plt
    plt.rc('text', usetex=True)

    sigunits = 1. * M_sun / (pc ** 2)
    timestep = 1e5 * yr
    plt.clf()
    for pars, plotstring in [
        (np.array([-10. * pc, 50. * sigunits, 100 * pc]), "b-"),
        (np.array([-10. * pc, 50. * sigunits, 200 * pc]), "r-"),
        (np.array([-10. * pc, 70. * sigunits, 140 * pc]), "k-"),
        ]:
        for vmax in np.arange(1., 30., 2.):
            zs, vs, phis = leapfrog(vmax * km / s, timestep, pure_exponential, pars)
            zs = zs / (pc)
            vs = vs / (km / s)
            plt.plot(vs, zs, plotstring, alpha=0.5, zorder=-10.)
    plt.xlabel(r"$v_z$ [km\,s$^{-1}$]")
    plt.ylabel(r"$z$ [pc]")
    plt.xlim(np.min(vs), np.max(vs))
    plt.ylim(np.min(zs), np.max(zs))
    plt.savefig("eatme.png")
