"""
This file is part of the ChemicalTangents project
copyright 2018 David W. Hogg (NYU) (MPIA) (Flatiron)

notes:
------
- Strictly SI units, which is INSANE.
- These functions ASSUME that the Sun's position and velocity have
  been accounted for, so that the positions and velocities are
  properly referenced to the Galactic Plane, not the Sun. That is, any
  Solar position you fit for should be done outside this code.
"""

import numpy as np

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

def ln_like(qs, invariants, order=3):
    """
    comments:
    - This function has a `posteriorlnvar` addition that approximates
      the relevant marginalization over the `order+1` linear
      parameters.
    - It is best to call this with something like
      `invariants - mean(invariants)`.

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

def pure_sech(z, pars):
    surfacedensity, scaleheight = pars
    return -8. * G * surfacedensity * np.arctan(np.tanh(z / scaleheight))

def pure_exponential(z, pars):
    surfacedensity, scaleheight = pars
    return -twopiG * surfacedensity * (1. - np.exp(-np.abs(z) / scaleheight)) * np.sign(z)

def dummy(z, pars):
    return -1.5 * np.sign(z)

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
    """
    bugs:
    - repeated code with `leapfrog_forward_to_zmax()`
    """
    if z == 0:
        return v, 0., 0., 0.
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
    action1 = 0.5 * np.sum((zs[1:t+1] + zs[:t]) * np.abs(vs[1:t+1] - vs[:t]))
    action1 += 0.5 * fraction * (zs[t+1] + zs[t]) * np.abs(vs[t+1] - vs[t])
    action2 = 0.5 * np.sum(np.abs(zs[1:t+1] - zs[:t]) * (vs[1:t+1] + vs[:t]))
    action2 += 0.5 * fraction * np.abs(zs[t+1] - zs[t]) * (vs[t+1] + vs[t])
    return vmax, time_at_midplane, action1, action2

def leapfrog_forward_to_zmax(z, v, timestep, acceleration, pars):
    """
    bugs:
    - repeated code with `leapfrog_back_to_midplane()`
    """
    if v == 0:
        return z, 0., 0., 0.
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
    action1 = 0.5 * np.sum((zs[1:t+1] + zs[:t]) * np.abs(vs[1:t+1] - vs[:t]))
    action1 += 0.5 * fraction * (zs[t+1] + zs[t]) * np.abs(vs[t+1] - vs[t])
    action2 = 0.5 * np.sum(np.abs(zs[1:t+1] - zs[:t]) * (vs[1:t+1] + vs[:t]))
    action2 += 0.5 * fraction * np.abs(zs[t+1] - zs[t]) * (vs[t+1] + vs[t])
    return zmax, time_at_zmax, action1, action2

def make_actions_angles_one_quadrant(z, v, pars, timestep = 0.1 * Myr, acceleration = pure_sech):
    """
    bugs:
    - Note horrible (0., 0.) hack
    """
    assert z >= 0.
    assert v >= 0.
    if z == 0. and v == 0.: # horrible special case
        return 0., 0., 0., 0.25 * np.pi
    vmax, time_at_midplane, jz1, jz2 = leapfrog_back_to_midplane(z, v, timestep, acceleration, pars)
    zmax, time_at_zmax, jz3, jz4 = leapfrog_forward_to_zmax(z, v, timestep, acceleration, pars)
    phi = (0. - time_at_midplane) * 0.5 * np.pi / (time_at_zmax - time_at_midplane)
    return zmax, vmax, 2. * (jz1 + jz2 + jz3 + jz4), phi

def make_action_angle_grid(zgrid, vgrid, pars):
    zs = np.outer(zgrid, np.ones_like(vgrid))
    vs = np.outer(np.ones_like(zgrid), vgrid)
    zmaxs = np.zeros_like(zs)
    vmaxs = np.zeros_like(zs)
    phis =  np.zeros_like(zs)
    Jzs =   np.zeros_like(zs)
    for i in range(len(zgrid)):
        for j in range(len(vgrid)):
            zmaxs[i, j], vmaxs[i, j], Jzs[i, j], phis[i, j] = \
                make_actions_angles_one_quadrant(zs[i, j], vs[i, j], pars)
    return zs, vs, zmaxs, vmaxs, Jzs, phis

def linearly_interpolate(inzs, invs, blob):
    """
    comments:
    - synchronized with `paint_actions_angles()`
    - deals with edge case by snapping to edge

    issues:
    - tons of repeated code
    - note `TINY`
    """
    dz, nz, dv, nv, xs, ys = blob
    TINY = 1.e-9 # dimensionless
    zindrs = np.clip(inzs / dz, TINY, nz - TINY)
    vindrs = np.clip(invs / dv, TINY, nv - TINY)
    zinds = zindrs.astype(int)
    vinds = vindrs.astype(int)
    zrs = zindrs - zinds
    mzrs = 1. - zrs
    vrs = vindrs - vinds
    mvrs = 1. - vrs
    outxs = mzrs * mvrs * xs[zinds    , vinds]     + \
            mzrs * vrs  * xs[zinds    , vinds + 1] + \
            zrs  * mvrs * xs[zinds + 1, vinds]     + \
            zrs  * vrs  * xs[zinds + 1, vinds + 1]
    outys = mzrs * mvrs * ys[zinds    , vinds]     + \
            mzrs * vrs  * ys[zinds    , vinds + 1] + \
            zrs  * mvrs * ys[zinds + 1, vinds]     + \
            zrs  * vrs  * ys[zinds + 1, vinds + 1]
    return outxs, outys

def paint_actions_angles(atzs, atvs, sunpars, dynpars, blob=None):
    """
    - There are dangerous units issues here. Need all inputs in SI
      units.
    - MAGIC numbers in `dz, nz, dv, nv`.
    - Check `np.sign()` insanity!
    - See `linearly_interpolate()` for edge issues.
    """
    if blob is None:
        print("paint_actions_angles: making action-angle grid")
        dz, nz = 100. * (pc), 20
        dv, nv = 5. * (km / s), 15
        zs, vs, zmaxs, vmaxs, Jzs, phis = \
            make_action_angle_grid(np.arange(nz + 1) * dz,
                                   np.arange(nv + 1) * dv, dynpars)
        xs = np.sqrt(Jzs) * np.cos(phis)
        ys = np.sqrt(Jzs) * np.sin(phis)
        blob = (dz, nz, dv, nv, xs, ys)
    else:
        dz, nz, dv, nv, xs, ys = blob
    print("paint_actions_angles: interpolating")
    inzs = atzs + sunpars[0]
    invs = atvs + sunpars[1]
    outxs, outys = linearly_interpolate(inzs * np.sign(inzs), invs * np.sign(invs), blob)
    outphis = np.arctan2(outys * np.sign(inzs), outxs * np.sign(invs))
    outphis[outphis < 0.] += 2. * np.pi
    print("paint_actions_angles: done")
    return outxs ** 2 + outys ** 2, outphis, blob

if __name__ == "__main__":
    import pylab as plt
    plt.rc('text', usetex=True)
    kmpspMyr = 1 * ((km / s) / (Myr))

    pars = np.array([70. * sigunits, 350 * pc])
    zgrid = np.arange(26) * 30. * (pc)
    vgrid = np.arange(26) * 3. * (km / s)
    zs, vs, zmaxs, vmaxs, Jzs, phis = make_action_angle_grid(zgrid, vgrid, pars)

    plt.clf()
    plt.scatter(vs.flatten() / (km / s), zs.flatten() / (pc), c=(Jzs.flatten() / (pc * km / s)), alpha=0.5)
    plt.colorbar()
    plt.savefig("deleteme0.png")
    plt.clf()
    plt.scatter(vs.flatten() / (km / s), zs.flatten() / (pc), c=(zmaxs.flatten() / (pc)), alpha=0.5)
    plt.colorbar()
    plt.savefig("deleteme1.png")
    plt.clf()
    plt.scatter(vs.flatten() / (km / s), zs.flatten() / (pc), c=(vmaxs.flatten() / (km / s)), alpha=0.5)
    plt.colorbar()
    plt.savefig("deleteme2.png")
    plt.clf()
    plt.scatter(vs.flatten() / (km / s), zs.flatten() / (pc), c=(phis.flatten() * 180. / np.pi), alpha=0.5)
    plt.colorbar()
    plt.savefig("deleteme3.png")

    plt.clf()
    for i in range(len(vs)):
        plt.plot(vs[i] / (pc), phis[i], "k-", alpha=0.5)
    plt.savefig("deleteme4.png")

    plt.clf()
    for i in range(len(vs.T)):
        plt.plot(zs[:, i] / (km / s), phis[:, i], "k-", alpha=0.5)
    plt.savefig("deleteme5.png")

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
    Jzs, phis, blob = paint_actions_angles(zs, vs, pars)

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
