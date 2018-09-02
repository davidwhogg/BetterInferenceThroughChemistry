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

from collections import namedtuple
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
from pyia import GaiaData
from integrate_orbits import *
from chemical_tangents import *

from os import path
import sys
_pwd = path.split(path.abspath(__file__))[0]
_path = path.abspath(path.join(_pwd, '..', 'pkg'))
if _path not in sys.path:
    sys.path.append(_path)
from chemtrails.data import (get_label_from_abundancename, get_abundance_data,
                             load_nominal_galah)

def hogg_savefig(thing, name):
    print("hogg_savefig(): saving figure {}".format(name))
    return thing.savefig(name)

def get_abundancenames(reference):
    if reference == "fe":
                     #         3        8       12       19      30       39      56       63
        abundances = ["fe_h", "li_fe", "o_fe", "mg_fe", "k_fe", "zn_fe", "y_fe", "ba_fe", "eu_fe", ]
    if reference == "o":
                     #        3       12      19     26      30      39     56      63
        abundances = ["o_h", "li_o", "mg_o", "k_o", "fe_o", "zn_o", "y_o", "ba_o", "eu_o", ]
    if reference == "h":
              #        3       12     19      26     28      30      39     56      63
        abundances = ["li_h", "o_h", "mg_h", "k_h", "fe_h", "zn_h", "y_h", "ba_h", "eu_h", ]
    abundancelabels = [get_label_from_abundancename(name) for name in abundances]
    return np.array(abundances), np.array(abundancelabels)

def setup_abundance_plot_grid(reference):
    nx, ny = 3, 3
    fig, axes = plt.subplots(ny, nx, figsize=(nx * 4, ny * 4), sharex=True, sharey=True)
    axes = axes.flatten()
    foo, alabels = get_abundancenames(reference)
    [ax.text(0.02, 0.98, label, ha="left", va="top", transform=ax.transAxes) for ax, label in zip(axes, alabels)]
    fig.tight_layout()
    return fig, axes, nx, ny

def plot_samplings(reference):
    fig, ax, nx, ny = setup_abundance_plot_grid(reference)
    fig2, ax2, foo, bar = setup_abundance_plot_grid(reference)
    abundances, foo = get_abundancenames(reference)

    color, alpha = "0.5", 0.3
    allcolor, allalpha = "k", 0.01

    picklefn = "samples_all_{}.pkl".format(reference)
    allsamples = unpickle_from_file(picklefn)
    allxs = np.exp(allsamples[:, 2])
    allys = np.exp(allsamples[:, 3])

    for i, aname in enumerate(abundances):
        picklefn = "samples_{}.pkl".format(aname)
        samples = unpickle_from_file(picklefn)
        xs = np.exp(samples[:, 2])
        ys = np.exp(samples[:, 3])
        foo = ax[i].scatter(xs, ys,
                            marker=".", s=10,
                            c=color, alpha=alpha, rasterized=True)
        foo = ax[i].scatter(allxs, allys,
                            marker=".", s=10,
                            c=allcolor, alpha=allalpha, rasterized=True)
        foo = ax2[i].scatter(samples[:, 1], samples[:, 0],
                             marker=".", s=10,
                             c=color, alpha=alpha, rasterized=True)
        foo = ax2[i].scatter(allsamples[:, 1], allsamples[:, 0],
                             marker=".", s=10,
                             c=allcolor, alpha=allalpha, rasterized=True)
        if i // nx + 1 == ny:
            ax[i].set_xlabel(r"integrated surface density $\Sigma$ [usual units]")
            ax2[i].set_xlabel(r"Sun's $v_z$ velocity [km / s]")
        if i % nx == 0:
            ax[i].set_ylabel(r"scaleheight $h$ [pc]")
            ax2[i].set_ylabel(r"sun's $z$ position [pc]")
    ax[0].set_xlim(30., 150.)
    ax[0].set_ylim(100., 800.)
    ax2[0].set_xlim(-6., 6.)
    ax2[0].set_ylim(-100., 100.)
    return fig, ax, fig2, ax2

def plot_abundances(galah, kinematicdata, reference):
    fig, ax, nx, ny = setup_abundance_plot_grid(reference)
    abundances, foo = get_abundancenames(reference)

    for i, aname in enumerate(abundances):
        abundance = get_abundance_data(galah, aname)
        good = np.abs(abundance) < 2. # hack
        vmin, vmax = np.percentile(abundance[good], [5., 95.])
        zs, vs = kinematicdata.z, kinematicdata.vz
        foo = ax[i].scatter(vs[good], zs[good],
                            marker=".", s=3000/np.sqrt(np.sum(good)),
                            c=abundance[good], vmin=vmin, vmax=vmax, alpha=0.3,
                            cmap=mpl.cm.plasma, rasterized=True)
        if i // nx + 1 == ny:
            ax[i].set_xlabel("$v_z$ [km/s]")
        if i % nx == 0:
            ax[i].set_ylabel("$z$ [pc]")
    ax[0].set_xlim(-40., 40.)
    ax[0].set_ylim(-1000., 1000.)
    return fig, ax

def plot_lf_slices(galah, kinematicdata, sunpars0, dynpars0, metalname, metallabel):

    # plot some likelihood sequences
    for k, units, name, scale in [
#        (0, pc, "zsun", 80.),
#        (1, km / s, "vsun", 4.),
        (2, sigunits, "sigma", 35.),
#        (3, pc, "scaleheight", 300.),
        ]:
        metals = get_abundance_data(galah, metalname)
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
        parsis = pars[i] + np.arange(-1., 1.001, 0.025) * scale * units
        llfs = np.zeros_like(parsis)
        blob = None
        for j, parsi in enumerate(parsis):
            pars[i] = parsi
            zs, vzs = kinematicdata.z * pc, kinematicdata.vz * km / s
            Es = paint_energies(zs, vzs, sunpars, dynpars)
            invariants = Es
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
    print("reading and cutting galah data")
    galah, galcen = load_nominal_galah('../data/GALAH-GaiaDR2-xmatch.fits.gz',
                                       zlim=1*u.kpc,
                                       vlim=40*u.km/u.s)
    reference = "h"

    # make kinematic-data object for `chemical_tangents.py`
    KinematicData = namedtuple("KinematicData", ["z", "vz"])
    kinematicdata = KinematicData(galcen.z.to(u.pc).value, galcen.v_z.to(u.km/u.s).value)

    # make all slice plots
    if False:
        sunpars0 = np.array([0. * pc, 0. * km / s])
        dynpars0 = np.array([90. * sigunits, 150. * pc])
        metalnames = [l for l in galah.__dir__() if ("_fe" in l and l[:2] != "e_")]
        metalnames += [l.split("_")[0]+"_h" for l in metalnames]
        metalnames = ["fe_h", ] + metalnames
        for metalname in metalnames:
            metallabel = get_label_from_abundancename(metalname)
            plot_lf_slices(galah, kinematicdata, sunpars0, dynpars0, metalname, metallabel)

    # make abundance plots
    if True:
        fig, ax = plot_abundances(galah, kinematicdata, reference)
        hogg_savefig(fig, "galah_abundances_{}.png".format(reference))

    # sample and corner plot all, and then each individually
    abundances, abundancelabels = get_abundancenames(reference)
    for abundance in abundances:
        if abundance not in galah.__dir__():
            galah.data[abundance] = get_abundance_data(galah, abundance) # STUPID HACK to make getattr() work
    picklefn = "samples_all_{}.pkl".format(reference)
    if os.path.isfile(picklefn):
        print("__main__: skipping all {}".format(reference))
    else:
        open(picklefn, "wb").close()
        print("__main__: working on all {}".format(reference))
        samples, fig = sample_and_plot(kinematicdata, galah, abundances)
        hogg_savefig(fig, "corner_all_{}.png".format(reference))
        pickle_to_file(samples, picklefn)
    for abundance in abundances:
        picklefn = "samples_{}.pkl".format(abundance)
        if os.path.isfile(picklefn):
            print("__main__: skipping {}".format(abundance))
        else:
            open(picklefn, "wb").close() # touch to lock
            print("__main__: working on {}".format(abundance))
            samples, fig = sample_and_plot(kinematicdata, galah, [abundance, ])
            hogg_savefig(fig, "corner_{}.png".format(abundance))
            pickle_to_file(samples, picklefn)

    if True:
        fig, ax, fig2, ax2 = plot_samplings(reference)
        hogg_savefig(fig, "dynpars_samplings_{}.png".format(reference))
        hogg_savefig(fig2, "sunpars_samplings_{}.png".format(reference))
        plt.close(fig)
        plt.close(fig2)
