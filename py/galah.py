"""
This file is part of the ChemicalTangents project.
Copyright 2018 David W. Hogg (MPIA).

bugs:
-----
- I don't know what parameters Pyia is using to go to Galactic
  6-space.
"""

from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyia import GaiaData
from integrate_orbits import *

def hogg_savefig(name):
    print("saving figure {}".format(name))
    return plt.savefig(name)

def plot_some_abundances(galah, galcen):
    nx, ny = 3, 2
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 5, ny * 5), sharex=True, sharey=True)
    ax = ax.flatten()
    for i,aname in enumerate(abundancenames):
        abundance = getattr(galah, aname)
        good = np.abs(abundance) < 5.
        vmin, vmax = np.percentile(abundance[good], [5., 95.])
        foo = ax[i].scatter(galcen[good].v_z.to(u.km/u.s).value, 
                         galcen[good].z.to(u.pc).value,
                         marker=".", s=3000/np.sqrt(np.sum(good)),
                         c=abundance[good], vmin=vmin, vmax=vmax, alpha=0.3,
                         cmap=mpl.cm.plasma, rasterized=True)
        ax[i].text(-vzmax * 0.96, zmax * 0.96, abundancelabels[aname], ha="left", va="top", backgroundcolor="w")
        if i % nx == 0:
            ax[i].set_ylabel("$z$ [pc]")
        if i // nx + 1 == ny:
            ax[i].set_xlabel("$v_z$ [km/s]")
    ax[0].set_xlim(-vzmax, vzmax)
    ax[0].set_ylim(-zmax, zmax)
    fig.tight_layout()
    return fig, ax

def plot_uphis(galah, galcen, parindex, offset):
    # get actions and angles
    sunpars = np.array([0. * pc, 2. * km / s])
    dynpars = np.array([65. * sigunits, 350 * pc])
    if parindex < 2:
        sunpars[parindex] += offset
    else:
        dynpars[parindex - 2] += offset
    zs = galcen.z.to(u.pc).value + sunpars[0] / pc
    vs = galcen.v_z.to(u.km/u.s).value + sunpars[1] / (km / s)
    Jzs, phis, blob = paint_actions_angles(zs, vs, dynpars)
    # exclude inner and outer rings, as it were
    good = (Jzs > (np.min(Jzs) + 0.1)) & (Jzs < (np.max(Jzs) - 0.1))
    Jzs = Jzs[good]
    phis = phis[good]

    # do some cray shit; not quite the right thing to do
    uJzs = np.unique(np.sort(Jzs))
    uphis  = np.unique(np.sort(phis))
    mg_fe_minus_mean = 1. * galah[good].mg_fe
    for vmax in uJzs:
        this = Jzs == vmax
        mg_fe_minus_mean[this] -= np.mean(galah[good][this].mg_fe)
    mg_fe_offsets = np.zeros_like(uphis)
    mg_fe_offsets_err = np.zeros_like(uphis)
    for i, phi in enumerate(uphis):
        this = phis == phi
        mg_fe_offsets[i] = np.mean(mg_fe_minus_mean[this])
        mg_fe_offsets_err[i] = np.sqrt(np.var(mg_fe_minus_mean[this]) / np.sum(this))

    # plot mg_fe_offsets
    nx, ny = 1, 1
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 10, ny * 5), sharex=True, sharey=True)
    foo = ax.errorbar(uphis, mg_fe_offsets, fmt="k.", yerr=mg_fe_offsets_err)
    ax.axhline(0., color="k", alpha=0.5, zorder=-10.)
    ax.set_xlabel(r"$\theta_z$ [rad]")
    ax.set_ylabel("mean [Mg/Fe] offset [dex]")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-0.2, 0.2)
    plt.title(r"$z = {:.1f}$ pc; $v_z = {:.1f}$ km/s; $\Sigma = {:.0f}$ M/pc$^2$; $h = {:.0f}$ pc".format(
            sunpars[0] / (pc), sunpars[1] / (km / s), dynpars[0] / (sigunits), dynpars[1] / (pc)))
    return fig, ax

def plot_uJzs(galah, galcen, parindex, offset):
    # get actions and angles
    sunpars = np.array([0. * pc, 2. * km / s])
    dynpars = np.array([65. * sigunits, 400 * pc])
    if parindex < 2:
        sunpars[parindex] += offset
    else:
        dynpars[parindex - 2] += offset
    zs = galcen.z.to(u.pc).value + sunpars[0] / pc
    vs = galcen.v_z.to(u.km/u.s).value + sunpars[1] / (km / s)
    Jzs, phis, blob = paint_actions_angles(zs, vs, dynpars)
    # exclude inner and outer rings, as it were
    good = (Jzs > (np.min(Jzs) + 0.1)) & (Jzs < (np.max(Jzs) - 0.1))
    Jzs = Jzs[good]
    phis = phis[good]

    # do some cray shit; not quite the right thing to do
    uJzs = np.unique(np.sort(Jzs))
    mg_fe_lnvars = np.zeros_like(uJzs)
    denominators = np.zeros_like(uJzs)
    TINY = -8
    for i,vmax in enumerate(uJzs):
        this = Jzs == vmax
        denominators[i] = np.sum(this) - 1.
        var = np.var(galah[good][this].mg_fe)
        mg_fe_lnvars[i] = np.log(var + np.exp(TINY))

    # plot mg_fe_offsets
    nx, ny = 1, 1
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 10, ny * 5), sharex=True, sharey=True)
    ax.plot(uJzs, mg_fe_lnvars, "k.")
    mean_lnvar = np.sum(mg_fe_lnvars * denominators) / np.sum(denominators)
    print(mean_lnvar)
    ax.axhline(mean_lnvar, color="k", alpha=0.5, zorder=-10)
    ax.set_xlabel(r"$v_{\max}$ [km/s]")
    ax.set_ylabel("log variance of [Mg/Fe] at that orbit [nat]")
    ax.set_xlim(0., 80.) # km/s
    ax.set_ylim(-4., -3.) # nats
    plt.title(r"$z = {:.1f}$ pc; $v_z = {:.1f}$ km/s; $\Sigma = {:.0f}$ M/pc$^2$; $h = {:.0f}$ pc".format(
            sunpars[0] / (pc), sunpars[1] / (km / s), dynpars[0] / (sigunits), dynpars[1] / (pc)))
    return fig, ax

if __name__ == "__main__":

    # read data and cut
    print("__main__: reading and cutting galah data")
    galah = GaiaData('../data/GALAH-GaiaDR2-xmatch.fits.gz')
    galah = galah[np.isfinite(galah.parallax_error)]
    galah = galah[(galah.parallax / galah.parallax_error) > 10.]
    galah = galah[(galah.teff > 4000*u.K) & (galah.teff < 6500*u.K)]
    galah = galah[galah.logg < 3.5]
    galah = galah[np.isfinite(galah.mg_fe)]
    galah = galah[(galah.mg_fe > -2.) * (galah.mg_fe < 2.)]

    # make coordinates
    print("__main__: dealing with coordinates")
    c = galah.get_skycoord(radial_velocity=galah.rv_synt)
    galcen = c.transform_to(coord.Galactocentric(z_sun=0*u.pc))
    zs = galcen.z.to(u.pc).value
    vs = galcen.v_z.to(u.km/u.s).value

    # trim on coordinates
    zlim = 2000. # pc
    vlim =   75. # km / s
    inbox = (zs / zlim) ** 2 + (vs / vlim) ** 2 < 1.
    galah = galah[inbox]
    galcen = galcen[inbox]

    # make phase space
    zs = galcen.z.to(u.pc).value * pc # note UNITS craziness
    vs = galcen.v_z.to(u.km/u.s).value * km / s # note UNITS craziness

    # set fiducial parameters
    sunpars0 = np.array([-15. * pc, 1. * km / s])
    dynpars0 = np.array([65. * sigunits, 390 * pc])
    metalpars0 = np.array([0.0379])
    metal, metalname = galah.mg_fe, "[Mg / Fe]"

    # plot various things for some standard potential
    if True:
        Jzs, phis, blob = paint_actions_angles(zs, vs, sunpars0, dynpars0)
        plt.clf()
        plt.scatter(vs / (km / s), zs / (pc), c=np.log(Jzs / (pc * km / s)), s=2)
        plt.colorbar()
        hogg_savefig("deleteme_galah0.png")
        plt.clf()
        plt.scatter(vs / (km / s), zs / (pc), c=(phis * 180. / np.pi), s=2)
        plt.colorbar()
        hogg_savefig("deleteme_galah3.png")

    if False:
        plt.clf()
        plt.plot(Jzs + 2. * np.random.uniform(-1, 1, size=len(Jzs)), galah.mg_fe, "k.", alpha=0.25)
        plotx = np.array([0., 76.])
        plt.plot(plotx, 0. + 0.004 * plotx, "r-", zorder=10)
        plt.xlabel(r"$v_\mathrm{max}$ (km / s)")
        plt.ylabel("{} (dex)".format(metalname))
        hogg_savefig("slope.png")

if True:

    # plot some likelihood sequences
    for k, units, name, scale in [(0, pc, "zsun", 20.),
                                  (1, km / s, "vsun", 1.5),
                                  (2, sigunits, "sigma", 8.),
                                  (3, pc, "scaleheight", 200.),
                                  (4, 1., "var", 0.001),
                                  ]:
        sunpars = 1. * sunpars0
        dynpars = 1. * dynpars0
        metalpars = 1. * metalpars0
        if k < 2:
            pars = sunpars
            i = k
            recompute = False
        elif k < 4:
            pars = dynpars
            i = k - 2
            recompute = True
        else:
            pars = metalpars
            i = k - 4
            recompute = False
        parsis = pars[i] + np.arange(-1., 1.001, 0.1) * scale * units
        llfs = np.zeros_like(parsis)
        blob = None
        for j, parsi in enumerate(parsis):
            pars[i] = parsi
            if recompute:
                blob = None
            Jzs, phis, blob = paint_actions_angles(zs, vs, sunpars, dynpars, blob=blob)
            invariants = Jzs
            invariants -= np.mean(invariants)
            llfs[j] = ln_like(metalpars, metal, invariants)
        plt.clf()
        plt.plot(parsis / units, llfs, "ko", alpha=0.75)
        plt.plot(parsis / units, llfs, "k-", alpha=0.75)
        plt.ylim(np.max(llfs)-10., np.max(llfs)+1.)
        plt.axhline(np.max(llfs)-2., color="k", alpha=0.25, zorder=-10)
        plt.xlabel(name)
        plt.ylabel("log LF")
        hogg_savefig("lf_{}_test.png".format(name))

if False:

    # decide what and how to plot
    galah.mn_mg = galah.mn_fe - galah.mg_fe
    galah.eu_mg = galah.eu_fe - galah.mg_fe
    abundancenames = ["fe_h", "mg_fe", "o_fe", "al_fe", "mn_mg", "eu_mg"]
    abundancelabels = {}
    abundancelabels["fe_h"] = "[Fe/H]"
    abundancelabels["mg_fe"] = "[Mg/Fe]"
    abundancelabels["o_fe"] = "[O/Fe]"
    abundancelabels["al_fe"] = "[Al/Fe]"
    abundancelabels["mn_fe"] = "[Mn/Fe]"
    abundancelabels["eu_fe"] = "[Eu/Fe]"
    abundancelabels["mn_mg"] = "[Mn/Mg]"
    abundancelabels["eu_mg"] = "[Eu/Mg]"
    zmax = 1500. # pc
    vzmax = 75. # km / s

    # plot
    fig, ax = plot_some_abundances(galah, galcen)
    fig.savefig("galah_full_sample.pdf")
    plt.close(fig)

    # make some plot sequences
    print("__main__: starting plotting cycle")
    # plotname, plotfunc = "offset_uphi", plot_uphis
    plotname, plotfunc = "lnvar_uvmax", plot_uJzs
    i = 0
    for j, off in enumerate(np.arange(-50., 51., 25.)):
        fig, ax = plotfunc(galah, galcen, i, off * pc)
        fig.savefig("{}_{}_{}.pdf".format(plotname, i, j))
        plt.close(fig)

    i = 1
    for j, off in enumerate(np.arange(-4., 4.1, 2.)):
        fig, ax = plotfunc(galah, galcen, i, off * km / s)
        fig.savefig("{}_{}_{}.pdf".format(plotname, i, j))
        plt.close(fig)

    i = 2
    for j, off in enumerate(np.arange(-20., 21., 10.)):
        fig, ax = plotfunc(galah, galcen, i, off * sigunits)
        fig.savefig("{}_{}_{}.pdf".format(plotname, i, j))
        plt.close(fig)

    i = 3
    for j, off in enumerate(np.arange(-200., 201., 100.)):
        fig, ax = plotfunc(galah, galcen, i, off * pc)
        fig.savefig("{}_{}_{}.pdf".format(plotname, i, j))
        plt.close(fig)

    print("__main__: done")

if False:

    # plot
    nx, ny = 2, 1
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 5, ny * 5), sharex=True, sharey=True)
    ax = ax.flatten()
    for i, q in enumerate([Jzs, phis]):
        vmin, vmax = np.percentile(q, [0.1, 99.9])
        foo = ax[i].scatter(vs, zs,
                         marker=".", s=3000/np.sqrt(len(vs)),
                         c=q, vmin=vmin, vmax=vmax, alpha=0.3,
                         cmap=mpl.cm.plasma, rasterized=True)
        if i % nx == 0:
            ax[i].set_ylabel("$z$ [pc]")
        if i // nx + 1 == ny:
            ax[i].set_xlabel("$v_z$ [km/s]")
    ax[0].set_xlim(-vlim, vlim)
    ax[0].set_ylim(-zlim, zlim)
    fig.savefig("galah_action_angle.pdf")
    plt.close(fig)

    # plot actions and angles
    nx, ny = 1, 1
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 5, ny * 5), sharex=True, sharey=True)
    q = mg_fe_minus_mean
    vmin, vmax = -0.25, 0.25
    foo = ax.scatter(phis, Jzs,
                     marker=".", s=3000/np.sqrt(len(vs)),
                     c=q, vmin=vmin, vmax=vmax, alpha=0.3,
                     cmap=mpl.cm.RdBu, rasterized=True)
    ax.set_xlabel(r"conjugate angle $\theta_z$ [rad]")
    ax.set_ylabel(r"$v_\mathrm{max}$ [km/s]")
    ax.set_xlim(0, 2. * np.pi)
    ax.set_ylim(0, np.max(Jzs))
    fig.savefig("galah_foo.pdf")
    plt.close(fig)

    # plot angle plots
    nx, ny = 1, 8
    fig, ax = plt.subplots(ny, nx, figsize=(10, 10), sharex=True, sharey=True)
    q = Jzs
    vmin, vmax = np.percentile(q, [0.5, 99.5])
    for i in range(ny):
        vmaxlims = np.percentile(Jzs, [100 * i / ny, 100 * (i + 1) / ny])
        inside = (Jzs > vmaxlims[0]) * (Jzs < vmaxlims[1])
        foo = ax[i].scatter(phis[inside], mg_fe_minus_mean[inside],
                            marker=".", s=1000/np.sqrt(np.sum(inside)),
                            c=q[inside], vmin=vmin, vmax=vmax, alpha=0.3,
                            cmap=mpl.cm.plasma, rasterized=True)
        if i % nx == 0:
            ax[-1].set_ylabel("[Mg/Fe] offset [dex]")
        if i // nx + 1 == ny:
            ax[-1].set_xlabel(r"conjugate angle $\theta_z$ [rad]")
    ax[-1].set_xlim(0., 2. * np.pi)
    ax[-1].set_ylim(-0.5, 0.5)
    fig.savefig("galah_mg_angle.pdf")
    plt.close(fig)

