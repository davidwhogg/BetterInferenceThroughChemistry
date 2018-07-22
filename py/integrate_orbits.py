import numpy as np

def leapfrog_step(z, v, dt, acceleration, pars):
    znew = z + v * dt
    dv = acceleration(znew, pars) * dt
    return znew, v + 0.5 * dv, v + dv

def leapfrog(vmax, dt, acceleration, pars):
    maxstep = 32768
    zs = np.zeros(maxstep) - np.Inf
    vs = np.zeros(maxstep) - np.Inf
    zs[0] = 0.
    vs[0] = vmax
    v = vs[0]
    for t in range(maxstep-1):
        zs[t + 1], vs[t + 1], v = leapfrog_step(zs[t], v, dt, acceleration, pars)
        if zs[t] < 0. and zs[t+1] >= 0.:
            break
    return zs[:t], vs[:t]

def dummy(z, pars):
    return -1. * np.sign(z)

if __name__ == "__main__":
    import pylab as plt

    pars = None
    zs, vs = leapfrog(100., 0.1, dummy, pars)
    print(len(zs), len(vs), min(zs))
    plt.plot(vs, zs, "k.")
    plt.savefig("eatme.png")
