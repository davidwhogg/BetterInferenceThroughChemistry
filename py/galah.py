"""
This file is part of the ChemicalTangents project.
Copyright 2018 David W. Hogg (MPIA).

to-do items:
------------
- Try splitting the sample by R_GC and see if there is a density
  gradient?

bugs:
-----
- I don't know what parameters Pyia is using to go to Galactic
  6-space.
- NEED TO SWITCH to using Astropy units correctly.
"""

from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path
from pyia import GaiaData
from integrate_orbits import *
from chemical_tangents import *

def pickle_to_file(thing, name):
    print("pickle_to_file(): writing {}".format(name))
    outfile = open(name, "wb")
    pickle.dump(thing, outfile)
    return outfile.close()

def unpickle_from_file(name):
    print("unpickle_from_file(): reading {}".format(name))
    outfile = open(name, "rb")
    thing = pickle.load(outfile)
    outfile.close()
    return thing

def hogg_savefig(thing, name):
    print("hogg_savefig(): saving figure {}".format(name))
    return thing.savefig(name)

def get_abundancenames():
                 #         8       12       13       14       20       28       39      63      
    abundances = ["fe_h", "o_fe", "mg_fe", "al_fe", "si_fe", "ca_fe", "ni_fe", "y_fe", "eu_fe", ]
    abundancelabels = []
    for abundance in abundances:
        foo = abundance.split("_")
        abundancelabels.append("["+foo[0].capitalize()+"/"+foo[1].capitalize()+"]")
    return np.array(abundances), np.array(abundancelabels)

def setup_abundance_plot_grid():
    nx, ny = 3, 3
    fig, axes = plt.subplots(ny, nx, figsize=(nx * 4, ny * 4), sharex=True, sharey=True)
    axes = axes.flatten()
    foo, alabels = get_abundancenames()
    [ax.text(0.02, 0.98, label, ha="left", va="top", transform=ax.transAxes) for ax, label in zip(axes, alabels)]
    return fig, axes, nx, ny

def plot_samplings():
    fig, ax, nx, ny = setup_abundance_plot_grid()
    abundances, foo = get_abundancenames()

    picklefn = "samples_{}.pkl".format("all")
    samples = unpickle_from_file(picklefn)
    allxs = np.exp(samples[:, 2])
    allys = np.exp(samples[:, 3])

    for i, aname in enumerate(abundances):
        picklefn = "samples_{}.pkl".format(aname)
        samples = unpickle_from_file(picklefn)
        xs = np.exp(samples[:, 2])
        ys = np.exp(samples[:, 3])
        foo = ax[i].scatter(xs, ys,
                            marker=".", s=10,
                            c="0.5", alpha=0.3, rasterized=True)
        foo = ax[i].scatter(allxs, allys,
                            marker=".", s=10,
                            c="k", alpha=0.3, rasterized=True)
        if i % nx == 0:
            ax[i].set_ylabel(r"scaleheight $h$ [pc]")
        if i // nx + 1 == ny:
            ax[i].set_xlabel(r"integrated surface density $\Sigma$ [usual units]")
    ax[0].set_xlim(30., 120.)
    ax[0].set_ylim(200., 600.)
    fig.tight_layout()
    return fig, ax

def plot_abundances(galah, galcen):
    fig, ax, nx, ny = setup_abundance_plot_grid()
    abundances, foo = get_abundancenames()

    for i, aname in enumerate(abundances):
        abundance = getattr(galah, aname)
        good = np.abs(abundance) < 5.
        vmin, vmax = np.percentile(abundance[good], [5., 95.])
        foo = ax[i].scatter(galcen[good].v_z.to(u.km/u.s).value, 
                            galcen[good].z.to(u.pc).value,
                            marker=".", s=3000/np.sqrt(np.sum(good)),
                            c=abundance[good], vmin=vmin, vmax=vmax, alpha=0.3,
                            cmap=mpl.cm.plasma, rasterized=True)
        if i % nx == 0:
            ax[i].set_ylabel("$z$ [pc]")
        if i // nx + 1 == ny:
            ax[i].set_xlabel("$v_z$ [km/s]")
    ax[0].set_xlim(-75., 75.)
    ax[0].set_ylim(-2000., 2000.)
    fig.tight_layout()
    return fig, ax

def plot_lf_slices(sunpars0, dynpars0, metalname, metallabel):

    # plot some likelihood sequences
    for k, units, name, scale in [
#        (0, pc, "zsun", 80.),
#        (1, km / s, "vsun", 4.),
        (2, sigunits, "sigma", 15.),
#        (3, pc, "scaleheight", 300.),
        ]:
        metals = getattr(galah, metalname)
        okay = (metals > -2.) & (metals < 2.) # HACKY
        sunpars = 1. * sunpars0
        dynpars = 1. * dynpars0
        if k < 2:
            pars = sunpars
            i = k
            recompute = False
        elif k < 4:
            pars = dynpars
            i = k - 2
            recompute = True
        parsis = pars[i] + np.arange(-1., 1.001, 0.05) * scale * units
        llfs = np.zeros_like(parsis)
        blob = None
        for j, parsi in enumerate(parsis):
            pars[i] = parsi
            if recompute:
                blob = None
            Jzs, phis, blob = paint_actions_angles(zs, vs, sunpars, dynpars, blob=blob)
            invariants = Jzs
            invariants -= np.mean(invariants)
            llfs[j] = ln_like(metals[okay], invariants[okay])
        plt.clf()
        plt.plot(parsis / units, llfs, "ko", alpha=0.75)
        plt.plot(parsis / units, llfs, "k-", alpha=0.75)
        plt.ylim(np.max(llfs)-10., np.max(llfs)+1.)
        plt.axhline(np.max(llfs)-2., color="k", alpha=0.25, zorder=-10)
        plt.xlabel(name)
        plt.ylabel("log LF")
        plt.title(metallabel)
        hogg_savefig(plt, "lf_{}_{}_test.png".format(name, metalname))

if __name__ == "__main__":
    plt.rc('text', usetex=True)

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

    # make abundance plots
    if False:
        fig, ax = plot_abundances(galah, galcen)
        hogg_savefig(fig, "galah_abundances.png")

    # sample and corner plot all, and then each individually
    abundances, abundancelabels = get_abundancenames()
    picklefn = "samples_{}.pkl".format("all")
    if os.path.isfile(picklefn):
        print("__main__: skipping {}".format("all"))
    else:
        open(picklefn, "wb").close()
        print("__main__: working on {}".format("all"))
        samples, fig = sample_and_plot(galcen, galah, abundances)
        hogg_savefig(fig, "corner_{}.png".format("all"))
        pickle_to_file(samples, picklefn)
    for abundance in abundances:
        picklefn = "samples_{}.pkl".format(abundance)
        if os.path.isfile(picklefn):
            print("__main__: skipping {}".format(abundance))
        else:
            open(picklefn, "wb").close()
            print("__main__: working on {}".format(abundance))
            samples, fig = sample_and_plot(galcen, galah, [abundance, ])
            hogg_savefig(fig, "corner_{}.png".format(abundance))
            pickle_to_file(samples, picklefn)

    if True:
        fig, ax = plot_samplings()
        hogg_savefig(fig, "dynpars_samplings.png")

if False:

    # set fiducial parameters
    sunpars0 = np.array([0. * pc, 0. * km / s])
    dynpars0 = np.array([64. * sigunits, 400. * pc])

    # make all slice plots
    metalnames, metallabels = get_abundancenames()
    for metalname, metallabel in zip(metalnames, metallabels):
        plot_lf_slices(sunpars0, dynpars0, metalname, metallabel)

    # plot various things for some standard potential
    if False:
        Jzs, phis, blob = paint_actions_angles(zs, vs, sunpars0, dynpars0)
        plt.clf()
        plt.scatter(vs / (km / s), zs / (pc), c=np.log(Jzs / (pc * km / s)), s=2)
        plt.colorbar()
        hogg_savefig(plt, "deleteme_galah0.png")
        plt.clf()
        plt.scatter(vs / (km / s), zs / (pc), c=(phis * 180. / np.pi), s=2)
        plt.colorbar()
        hogg_savefig(plt, "deleteme_galah3.png")

    if False:
        plt.clf()
        plt.plot(Jzs + 2. * np.random.uniform(-1, 1, size=len(Jzs)), galah.mg_fe, "k.", alpha=0.25)
        plotx = np.array([0., 76.])
        plt.plot(plotx, 0. + 0.004 * plotx, "r-", zorder=10)
        plt.xlabel(r"$v_\mathrm{max}$ (km / s)")
        plt.ylabel("{} (dex)".format(metallabel))
        hogg_savefig(plt, "slope.png")

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
    hogg_savefig(fig, "galah_full_sample.pdf")
    plt.close(fig)

    # make some plot sequences
    print("__main__: starting plotting cycle")
    # plotname, plotfunc = "offset_uphi", plot_uphis
    plotname, plotfunc = "lnvar_uvmax", plot_uJzs
    i = 0
    for j, off in enumerate(np.arange(-50., 51., 25.)):
        fig, ax = plotfunc(galah, galcen, i, off * pc)
        hogg_savefig(fig, "{}_{}_{}.pdf".format(plotname, i, j))
        plt.close(fig)

    i = 1
    for j, off in enumerate(np.arange(-4., 4.1, 2.)):
        fig, ax = plotfunc(galah, galcen, i, off * km / s)
        hogg_savefig(fig, "{}_{}_{}.pdf".format(plotname, i, j))
        plt.close(fig)

    i = 2
    for j, off in enumerate(np.arange(-20., 21., 10.)):
        fig, ax = plotfunc(galah, galcen, i, off * sigunits)
        hogg_savefig(fig, "{}_{}_{}.pdf".format(plotname, i, j))
        plt.close(fig)

    i = 3
    for j, off in enumerate(np.arange(-200., 201., 100.)):
        fig, ax = plotfunc(galah, galcen, i, off * pc)
        hogg_savefig(fig, "{}_{}_{}.pdf".format(plotname, i, j))
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
    hogg_savefig(fig, "galah_action_angle.pdf")
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
    hogg_savefig(fig, "galah_foo.pdf")
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
    hogg_savefig(fig, "galah_mg_angle.pdf")
    plt.close(fig)

