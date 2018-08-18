"""
This file is part of the ChemicalTangents project
copyright 2018 David W. Hogg (NYU) (MPIA) (Flatiron)

notes:
------
- Strictly SI units, which is INSANE.
- These functions ASSUME that the Sun's position and velocity have
  been accounted for, so that the positions and velocities are
  properly referenced to the Galactic Plane, not the Sun.

bugs / to-dos:
--------------
- Deal with edge stars by just assigning them to the largest action
  ring we have.
- I have to replace the nearest-neighbors with a home-built 2-d
  interpolation. That might require making the grid not in vmax, phi
  but in z, vz instead. That's some work but not a crazy amount. It's
  half-done now. The interpolation should take the look-up table as
  an input, because we don't always need to recompute it.
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
Myr = 1.e6 * yr
sigunits = 1. * M_sun / (pc ** 2)

def ln_like(pars, qs, vmaxs):
    """
    Note the MAGIC offset.
    """
    var = pars
    offset = 30.
    AT = np.vstack([np.ones_like(qs), vmaxs - offset])
    A = AT.T
    x = np.linalg.solve(np.dot(AT, A), np.dot(AT, qs))
    return -0.5 * np.sum((qs - np.dot(A, x)) ** 2 / var + np.log(var))

def leapfrog_step(z, v, dt, acceleration, pars):
    znew = z + v * dt
    dv = acceleration(znew, pars) * dt
    return znew, v + 0.5 * dv, v + dv

def leapfrog_full_circle(vmax, dt, acceleration, pars):
    maxstep = 32768 * 8
    zs = np.zeros(maxstep) - np.Inf
    vs = np.zeros(maxstep) - np.Inf
    zs[0] = 0.
    vs[0] = vmax
    v = vs[0]
    phis = None
    for t in range(maxstep-1):
        zs[t + 1], vs[t + 1], v = leapfrog_step(zs[t], v, dt, acceleration, pars)
        if zs[t] < 0. and zs[t+1] >= 0.:
            fraction = (0. - zs[t]) / (zs[t+1] - zs[t])
            period = dt * (t + fraction)
            phis = 2 * np.pi * np.arange(t+2) * dt / period
            break
    if phis is None:
        print("leapfrog: uh oh:", pars, t, zs[t + 1] / pc, vs[t + 1] / (km / s))
        assert phis is not None
    return zs[:t+2], vs[:t+2], phis

def leapfrog_back_to_midplane(z, v, timestep, acceleration, pars):
    if z == 0:
        return v, 0.
    maxstep = 32768
    zs = np.zeros(maxstep)
    vs = np.zeros(maxstep)
    zs[0] = z
    vs[0] = v
    dt = -np.abs(timestep)
    phis = None
    for t in range(maxstep-1):
        zs[t + 1], vs[t + 1], v = leapfrog_step(zs[t], v, dt, acceleration, pars)
        if zs[t + 1] < 0. and zs[t] >= 0.:
            fraction = (0. - zs[t]) / (zs[t+1] - zs[t])
            time_at_midplane = dt * (t + fraction)
            vmax = vs[t] + acceleration(zs[t], pars) * fraction * dt \
                + 0.5 * (acceleration(zs[t+1], pars) -
                         acceleration(zs[t], pars)) * dt * fraction * fraction
            break
    return vmax, time_at_midplane

def leapfrog_forward_to_zmax(z, v, timestep, acceleration, pars):
    if v == 0:
        return z, 0.
    maxstep = 32768
    zs = np.zeros(maxstep)
    vs = np.zeros(maxstep)
    zs[0], vs[0] = z, v
    dt = np.abs(timestep)
    phis = None
    for t in range(maxstep-1):
        zs[t + 1], vs[t + 1], v = leapfrog_step(zs[t], v, dt, acceleration, pars)
        if vs[t + 1] < 0. and vs[t] >= 0.:
            fraction = (0. - vs[t]) / (vs[t+1] - vs[t])
            time_at_zmax = dt * (t + fraction)
            zmax = zs[t] + vs[t] * fraction * dt + 0.5 * (vs[t+1] - vs[t]) * dt * fraction * fraction
            break
    return zmax, time_at_zmax

def make_actions_angles_one_quadrant(z, v, pars, timestep = 0.1 * Myr):
    assert z >= 0.
    assert v >= 0.
    if z == 0. and v == 0.: # horrible special case
        return 0., 0., 0.25 * np.pi
    vmax, time_at_midplane = leapfrog_back_to_midplane(z, v, timestep, pure_sech, pars)
    zmax, time_at_zmax = leapfrog_forward_to_zmax(z, v, timestep, pure_sech, pars)
    phi = (0. - time_at_midplane) * 0.5 * np.pi / (time_at_zmax - time_at_midplane)
    print(v / (km / s), vmax / (km / s), time_at_midplane / (Myr), z / (pc), zmax / (pc), time_at_zmax / (Myr), phi)
    return zmax, vmax, phi

def make_actions_angles_one(vmax, pars, timestep = 0.1 * Myr):
    Ngrid = 512
    phigrid = np.arange(np.pi / Ngrid, 2. * np.pi, 2. * np.pi / Ngrid)
    zs, vs, phis = leapfrog_full_circle(vmax * km / s, timestep, pure_sech, pars)
    zs = np.interp(phigrid, phis, zs / (pc))
    vs = np.interp(phigrid, phis, vs / (km / s))
    vmaxs = np.zeros_like(phigrid) + vmax
    return zs, vs, vmaxs, phigrid

def make_actions_angles(pars, vlim=75.):
    vmaxlist = np.arange(1., vlim + 0.001, 1.) # magic numbers
    zs, vs, phis, vmaxs = [], [], [], []
    for vmax in vmaxlist:
        tzs, tvs, tphis, tvmaxs = make_actions_angles_one(vmax, pars)
        zs = np.append(zs, tzs) # bad!
        vs = np.append(vs, tvs)
        phis = np.append(phis, tphis)
        vmaxs = np.append(vmaxs, tvmaxs)
    return zs, vs, vmaxs, phis

def paint_actions_angles(atzs, atvs, sunpars, dynpars, blob=None):
    if blob is None:
        print("paint_actions_angles: integrating orbits")
        zs, vs, phis, vmaxs = make_actions_angles(dynpars)
        print("paint_actions_angles: making KDTree")
        tree = KDTree(np.vstack([(zs / 1500.).T, (vs / 75.).T]).T)
        blob = (phis, vmaxs, tree)
    else:
        phis, vmaxs, tree = blob
    print("paint_actions_angles: getting nearest neighbors")
    inzs = atzs + sunpars[0] / pc
    invs = atvs + sunpars[1] / (km / s)
    inds = tree.query(np.vstack([(inzs / 1500.).T, (invs / 75.).T]).T, return_distance=False)
    print("paint_actions_angles: done")
    return vmaxs[inds].flatten(), phis[inds].flatten(), blob

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
    pars = np.array([100. * sigunits, 400 * pc])
    kmpspMyr = 1 * ((km / s) / (Myr))

    zgrid = np.arange(0., 1500.1, 60.) * (pc)
    vgrid = np.arange(0., 75.1, 3.) * (km / s)
    zs = np.outer(zgrid, np.ones_like(vgrid)).flatten()
    vs = np.outer(np.ones_like(zgrid), vgrid).flatten()
    zmaxs = np.zeros_like(zs)
    vmaxs = np.zeros_like(zs)
    phis = np.zeros_like(zs)
    for i, (z, v) in enumerate(zip(zs, vs)):
        zmaxs[i], vmaxs[i], phis[i] = make_actions_angles_one_quadrant(z, v, pars)
    """
    zs = np.append(zs, zs)
    vs = np.append(vs, 0. - vs)
    zmaxs = np.append(zmaxs, zmaxs)
    vmaxs = np.append(vmaxs, vmaxs)
    phis = np.append(phis, np.pi - phis)
    zs = np.append(zs, 0. - zs)
    vs = np.append(vs, vs)
    zmaxs = np.append(zmaxs, zmaxs)
    vmaxs = np.append(vmaxs, vmaxs)
    phis = np.append(phis, 2. * np.pi - phis)
    """
    plt.clf()
    plt.scatter(vs / (km / s), zs / (pc), c=(zmaxs / (pc)), alpha=0.5)
    plt.colorbar()
    plt.savefig("deleteme1.png")
    plt.clf()
    plt.scatter(vs / (km / s), zs / (pc), c=(vmaxs / (km / s)), alpha=0.5)
    plt.colorbar()
    plt.savefig("deleteme2.png")
    plt.clf()
    plt.scatter(vs / (km / s), zs / (pc), c=(phis * 180. / np.pi), alpha=0.5)
    plt.colorbar()
    plt.savefig("deleteme3.png")

    uzs = np.unique(np.sort(zs))
    TINY = 1e-3 * pc
    for uz in uzs:
        if uz < TINY:
            I = zs == uz
            plt.clf()
            plt.plot(vs[I] / (km / s), phis[I], "ro", alpha=0.5)
        if uz > 155.5 * pc:
            I = zs == uz
            plt.plot(vs[I] / (km / s), phis[I], "ko", alpha=0.5)
            plt.savefig("deleteme4.png")
            break

    uvs = np.unique(np.sort(vs))
    TINY = 1e-3 * km / s
    for uv in uvs:
        if uv < TINY:
            I = vs == uv
            plt.clf()
            plt.plot(zs[I] / (pc), phis[I], "ro", alpha=0.5)
        if uv > 20. * km / s:
            I = vs == uv
            plt.plot(zs[I] / (pc), phis[I], "ko", alpha=0.5)
            plt.savefig("deleteme5.png")
            break

if False:

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
    vmaxs, phis, blob = paint_actions_angles(zs, vs, pars)

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
        (np.array([50. * sigunits, 100 * pc]), "b"),
        (np.array([50. * sigunits, 200 * pc]), "r"),
        (np.array([70. * sigunits, 140 * pc]), "k"),
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
    plt.clf()
    for vmax in np.arange(1., 30., 2.):
        zs, vs, phis = leapfrog_full_circle(vmax * km / s, 1e5 * yr, pure_exponential, pars)
        zs = zs / (pc)
        vs = vs / (km / s)
        plt.plot(phis, zs, "k-", alpha=0.5)
    plt.xlim(2. * np.pi - 0.1, 2. * np.pi + 0.1)
    plt.axhline(0. / pc)
    plt.ylim(0. / pc - 10., 0. / pc + 10.)
    plt.savefig("test_2_pi.png")

    plt.clf()
    for vmax in np.arange(1., 30., 2.):
        zs, vs, phis = leapfrog_full_circle(vmax * km / s, 1e5 * yr, pure_exponential, pars)
        zs = zs / (pc)
        vs = vs / (km / s)
        plt.plot(phis, vs, "k-", alpha=0.5)
    plt.xlim(1.5 * np.pi - 0.1, 1.5 * np.pi + 0.1)
    plt.axhline(0. / (km / s))
    plt.ylim(0. / (km / s) - 1., 0. / (km / s) + 1.)
    plt.savefig("test_1p5_pi.png")
