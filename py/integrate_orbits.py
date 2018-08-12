"""
This file is part of the ChemicalTangents project
copyright 2018 David W. Hogg (NYU) (MPIA) (Flatiron)

notes:
------
- Strictly SI units, which is INSANE

bugs / to-dos:
--------------
- Need to re-test now that I moved the midplane parameter.
"""

import numpy as np
from sklearn.neighbors import KDTree

G = 6.67408e-11 # m m m / kg / s / s
twopiG = 2. * np.pi * G
fourG = 4. * G
pc = 3.0857e16 # m
M_sun = 1.9891e30 # kg
km = 1000. # m
s = 1. # s
yr = 365.25 * 24. * 3600. * s
sigunits = 1. * M_sun / (pc ** 2)

def leapfrog_step(z, v, dt, acceleration, pars):
    znew = z + v * dt
    dv = acceleration(znew, pars) * dt
    return znew, v + 0.5 * dv, v + dv

def leapfrog(vmax, dt, acceleration, pars):
    """
    assumes that pars[0:2] are the position and velocity of the barycenter,
    and pars[2:] are the acceleration pars.
    """
    zsun, vzsun = pars[0:2]
    maxstep = 32768
    zs = np.zeros(maxstep) - np.Inf
    vs = np.zeros(maxstep) - np.Inf
    zs[0] = 0.
    vs[0] = vmax
    v = vs[0]
    for t in range(maxstep-1):
        zs[t + 1], vs[t + 1], v = leapfrog_step(zs[t], v, dt, acceleration, pars[2:])
        if zs[t] < 0. and zs[t+1] >= 0.:
            fraction = (0. - zs[t]) / (zs[t+1] - zs[t])
            period = dt * (t + fraction)
            phis = 2 * np.pi * np.arange(t+2) * dt / period
            break
    return zs[:t+2] - zsun, vs[:t+2] - vzsun, phis

def make_actions_angles_one(vmax, pars, timestep = 1e5 * yr):
    Ngrid = 512
    phigrid = np.arange(np.pi / Ngrid, 2. * np.pi, 2. * np.pi / Ngrid)
    zs, vs, phis = leapfrog(vmax * km / s, timestep, pure_sech, pars)
    zs = np.interp(phigrid, phis, zs / (pc))
    vs = np.interp(phigrid, phis, vs / (km / s))
    vmaxs = np.zeros_like(phigrid) + vmax
    return zs, vs, vmaxs, phigrid

def make_actions_angles(pars, vlim=75.):
    vmaxlist = np.arange(2., vlim + 5., 4.) # magic numbers
    zs, vs, phis, vmaxs = [], [], [], []
    for vmax in vmaxlist:
        tzs, tvs, tphis, tvmaxs = make_actions_angles_one(vmax, pars)
        zs = np.append(zs, tzs) # bad!
        vs = np.append(vs, tvs)
        phis = np.append(phis, tphis)
        vmaxs = np.append(vmaxs, tvmaxs)
    return zs, vs, vmaxs, phis

def paint_actions_angles(atzs, atvs, pars):
    print("paint_actions_angles: intgrating orbits")
    zs, vs, phis, vmaxs = make_actions_angles(pars)
    print("paint_actions_angles: making KDTree")
    tree = KDTree(np.vstack([(zs / 1500.).T, (vs / 75.).T]).T)
    print("paint_actions_angles: getting nearest neighbors")
    inds = tree.query(np.vstack([(atzs / 1500.).T, (atvs / 75.).T]).T, return_distance=False)
    print("paint_actions_angles: done")
    return vmaxs[inds].flatten(), phis[inds].flatten()

def pure_sech(z, pars):
    surfacedensity, scaleheight = pars
    return -8. * G * surfacedensity * np.arctan(np.tanh(z / scaleheight))

def pure_exponential(z, pars):
    surfacedensity, scaleheight = pars
    return -twopiG * surfacedensity * (1. - np.exp(-np.abs(z) / scaleheight)) * np.sign(z)

def dummy(z, pars):
    return -1.5 * np.sign(z)

if __name__ == "__main__":
    import pylab as plt
    plt.rc('text', usetex=True)
    pars = np.array([-10. * pc, 0. * km / s, 100. * sigunits, 400 * pc])
    kmpspMyr = 1 * ((km / s) / (1e6 * yr))

    plt.clf()
    dz = 1.
    zs = np.arange(-1500. + 0.5 * dz, 1500., dz) * pc
    plt.plot(zs / pc, pure_exponential(zs, pars[2:]) / kmpspMyr, "k-", alpha=0.75)
    plt.plot(zs / pc, pure_sech(zs, pars[2:]) / kmpspMyr, "b-", alpha=0.75)
    plt.savefig("sech.png")

if False:

    zlim = 1500. # pc
    vlim =   75. # km/s

    Nstars = 1000
    zs = zlim * (np.random.uniform(size=Nstars) * 2. - 1)
    vs = vlim * (np.random.uniform(size=Nstars) * 2. - 1)
    vmaxs, phis = paint_actions_angles(zs, vs, pars)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    plt.scatter(vs, zs, c=phis, s=10.)
    plt.xlim(-vlim, vlim)
    plt.ylim(-zlim, zlim)
    plt.savefig("phis.png")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    plt.scatter(vs, zs, c=vmaxs, s=10.)
    plt.xlim(-vlim, vlim)
    plt.ylim(-zlim, zlim)
    plt.savefig("vmaxs.png")

    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    for pars, plotstring in [
        (np.array([10. * pc, 0.5 * km / s, 50. * sigunits, 100 * pc]), "b"),
        (np.array([10. * pc, 0.5 * km / s, 50. * sigunits, 200 * pc]), "r"),
        (np.array([10. * pc, 0.5 * km / s, 70. * sigunits, 140 * pc]), "k"),
        ]:
        zs, vs, vmaxs, phis = make_actions_angles(pars)
        ax1.plot(vs, zs, plotstring+".", alpha=0.5)
        ax2.plot(vs, zs, plotstring+"-", alpha=0.5)
        ax3.plot(phis, vmaxs, plotstring+".", alpha=0.5)
    ax1.set_xlabel(r"$v_z$ [km\,s$^{-1}$]")
    ax1.set_ylabel(r"$z$ [pc]")
    ax1.set_xlim(35., 35. + 0.25 * vlim)
    ax1.set_ylim(-600., -600. + 0.25 * zlim)
    fig1.savefig("biteme.png")
    ax2.set_xlabel(r"$v_z$ [km\,s$^{-1}$]")
    ax2.set_ylabel(r"$z$ [pc]")
    ax2.set_xlim(0, vlim)
    ax2.set_ylim(0, zlim)
    fig2.savefig("eatme.png")

    # testing
    pars[0] = 7.5 * pc
    pars[1] = 1.5 * (km / 2)
    plt.clf()
    for vmax in np.arange(1., 30., 2.):
        zs, vs, phis = leapfrog(vmax * km / s, 1e5 * yr, pure_exponential, pars)
        zs = zs / (pc)
        vs = vs / (km / s)
        plt.plot(phis, zs, "k-", alpha=0.5)
    plt.xlim(2. * np.pi - 0.1, 2. * np.pi + 0.1)
    plt.axhline(-pars[0] / pc)
    plt.ylim(-pars[0] / pc - 10., -pars[0] / pc + 10.)
    plt.savefig("test_2_pi.png")

    plt.clf()
    for vmax in np.arange(1., 30., 2.):
        zs, vs, phis = leapfrog(vmax * km / s, 1e5 * yr, pure_exponential, pars)
        zs = zs / (pc)
        vs = vs / (km / s)
        plt.plot(phis, vs, "k-", alpha=0.5)
    plt.xlim(1.5 * np.pi - 0.1, 1.5 * np.pi + 0.1)
    plt.axhline(-pars[1] / (km / s))
    plt.ylim(-pars[1] / (km / s) - 1., -pars[1] / (km / s) + 1.)
    plt.savefig("test_1p5_pi.png")
