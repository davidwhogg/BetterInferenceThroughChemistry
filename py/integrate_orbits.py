import numpy as np

def leapfrog_step(z, v, dt, acceleration, pars):
    znew = z + v * dt
    dv = acceleration(znew, pars) * dt
    return znew, v + 0.5 * dv, v + dv

def leapfrog(vmax, dt, acceleration, pars):
    maxstep = 32768 * 16
    zs = np.zeros(maxstep) - np.Inf
    vs = np.zeros(maxstep) - np.Inf
    zs[0] = 0.
    vs[0] = vmax
    v = vs[0]
    for t in range(maxstep-1):
        zs[t + 1], vs[t + 1], v = leapfrog_step(zs[t], v, dt, acceleration, pars)
        if zs[t] < 0. and zs[t+1] >= 0.:
            break
    return zs[:t+2], vs[:t+2]

def dummy(z, pars):
    return -1.5 * np.sign(z)

if __name__ == "__main__":
    import pylab as plt

    pars = None
    vmax = 25.
    zs, vs = leapfrog(vmax, 0.01, dummy, pars)
    print(len(zs), len(vs), min(zs))
    print(zs[-3:], vs[-3:], min(zs))
    plt.plot(vs, zs, "k-")
    plt.savefig("eatme.png")
    plt.xlim(vmax-1., vmax+1.)
    plt.ylim(-1., 1.)
    plt.savefig("biteme.png")
