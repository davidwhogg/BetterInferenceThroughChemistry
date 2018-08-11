"""
This file is part of the ChemicalTangents project
copyright 2018 David W. Hogg (NYU) (MPIA) (Flatiron)

Strictly SI units, which is INSANE
"""
import numpy as np
from sklearn.neighbors import KDTree

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
            fraction = (midplane - zs[t]) / (zs[t+1] - zs[t])
            period = dt * (t + fraction)
            phis = 2 * np.pi * np.arange(t+2) * dt / period
            break
    return zs[:t+2], vs[:t+2], phis

def make_actions_angles_one(vmax, pars):
    Ngrid = 128
    phigrid = np.arange(np.pi / Ngrid, 2. * np.pi, 1. / Ngrid)
    zs, vs, phis = leapfrog(vmax * km / s, timestep, pure_exponential, pars)
    zs = np.interp(phigrid, phis, zs / (pc))
    vs = np.interp(phigrid, phis, vs / (km / s))
    vmaxs = np.zeros_like(phigrid) + vmax
    return zs, vs, vmaxs, phigrid

def make_actions_angles(pars, zlim=1500., vlim=75.):
    vmaxlist = np.arange(0.5, 200., 1.)
    zs, vs, phis, vmaxs = [], [], [], []
    for vmax in vmaxlist:
        tzs, tvs, tphis, tvmaxs = make_actions_angles_one(vmax, pars)
        zs = np.append(zs, tzs) # bad!
        vs = np.append(vs, tvs)
        phis = np.append(phis, tphis)
        vmaxs = np.append(vmaxs, tvmaxs)
        if np.sum((np.abs(zs) < zlim) & (np.abs(vs) < vlim)) == 0:
            break
    return zs, vs, vmaxs, phis

def paint_actions_angles(atzs, atvs, pars):
    print("paint_actions_angles: intgrating orbits")
    zs, vs, phis, vmaxs = make_actions_angles(pars)
    print("paint_actions_angles: making KDTree")
    tree = KDTree(np.vstack([(zs / 1500.).T, (vs / 75.).T]).T)
    print("paint_actions_angles: getting nearest neighbors")
    inds = tree.query(np.vstack([(atzs / 1500.).T, (atvs / 75.).T]).T, return_distance=False)
    print("paint_actions_angles: done")
    return vmaxs[inds], phis[inds]

def pure_exponential(z, pars):
    midplane, surfacedensity, scaleheight = pars
    zprime = z - midplane
    return -1. * fourpiG * surfacedensity * (1. - np.exp(-np.abs(zprime) / scaleheight)) * np.sign(zprime)

def dummy(z, pars):
    return -1.5 * np.sign(z)

if __name__ == "__main__":
    import pylab as plt
    plt.rc('text', usetex=True)

    zlim = 1500. # pc
    vlim =   75. # km/s

    sigunits = 1. * M_sun / (pc ** 2)
    timestep = 1e5 * yr
    pars = np.array([-10. * pc, 50. * sigunits, 100 * pc])

    Nstars = 1000
    zs = zlim * (np.random.uniform(size=Nstars) * 2. - 1)
    vs = vlim * (np.random.uniform(size=Nstars) * 2. - 1)
    vmaxs, phis = paint_actions_angles(zs, vs, pars)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    plt.scatter(vs, zs, c=phis)
    plt.xlim(-vlim, vlim)
    plt.ylim(-zlim, zlim)
    plt.savefig("phis.png")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    plt.scatter(vs, zs, c=vmaxs)
    plt.xlim(-vlim, vlim)
    plt.ylim(-zlim, zlim)
    plt.savefig("vmaxs.png")

if False:
    sigunits = 1. * M_sun / (pc ** 2)
    timestep = 1e5 * yr
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
    for pars, plotstring in [
        (np.array([-10. * pc, 50. * sigunits, 100 * pc]), "b."),
        (np.array([-10. * pc, 50. * sigunits, 200 * pc]), "r."),
        (np.array([-10. * pc, 70. * sigunits, 140 * pc]), "k."),
        ]:
        zs, vs, vmaxs, phis = make_actions_angles(pars)
        plt.plot(vs, zs, plotstring, alpha=0.5)
    plt.xlabel(r"$v_z$ [km\,s$^{-1}$]")
    plt.ylabel(r"$z$ [pc]")
    plt.xlim(-vlim, vlim)
    plt.ylim(-zlim, zlim)
    plt.savefig("eatme.png")
    plt.xlim(35., 40.)
    plt.ylim(-600., -500.)
    plt.savefig("biteme.png")

    # testing
    plt.clf()
    for vmax in np.arange(1., 30., 2.):
        zs, vs, phis = leapfrog(vmax * km / s, timestep, pure_exponential, pars)
        zs = zs / (pc)
        vs = vs / (km / s)
        plt.plot(phis, zs, "k-", alpha=0.5)
    plt.xlim(2. * np.pi - 0.1, 2. * np.pi + 0.1)
    plt.axhline(pars[0] / pc)
    plt.ylim(pars[0] / pc - 1., pars[0] / pc + 1.)
    plt.savefig("test_2_pi.png")
