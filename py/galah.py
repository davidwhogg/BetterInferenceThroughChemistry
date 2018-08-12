"""
This file is part of the ChemicalTangents project.
Copyright 2018 David W. Hogg (MPIA).

bugs:
-----
- I don't know what parameters Pyia is using to go to Galactic 6-space.
- stop cutting in z and vz and cut instead on vmax!
"""

from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyia import GaiaData
from integrate_orbits import *

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
    pars = np.array([0. * pc, 2. * km / s, 40. * sigunits, 350 * pc])
    pars[parindex] += offset
    zs = galcen.z.to(u.pc).value
    vs = galcen.v_z.to(u.km/u.s).value
    vmaxs, phis = paint_actions_angles(zs, vs, pars)

    # do some cray shit; not quite the right thing to do
    uvmaxs = np.unique(np.sort(vmaxs))
    uphis  = np.unique(np.sort(phis))
    mg_fe_minus_mean = 1. * galah.mg_fe
    for vmax in uvmaxs:
        this = vmaxs == vmax
        mg_fe_minus_mean[this] -= np.mean(galah[this].mg_fe)
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
            pars[0] / (pc), pars[1] / (km / s), pars[2] / (sigunits), pars[3] / (pc)))
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
    zlim = 1500. # pc
    vlim =   75. # km / s
    inbox = (np.abs(zs / zlim) < 1.) & (np.abs(vs / vlim) < 1)
    galah = galah[inbox]
    galcen = galcen[inbox]

    # makd some plots
    print("__main__: starting plotting cycle")
    fig, ax = plot_uphis(galah, galcen, 0, 0.)
    fig.savefig("offset_uphi.pdf")
    plt.close(fig)

    i = 0
    for j, off in enumerate(np.arange(-50., 51., 25.)):
        fig, ax = plot_uphis(galah, galcen, i, off * pc)
        fig.savefig("offset_uphi_{}_{}.pdf".format(i, j))
        plt.close(fig)

    i = 1
    for j, off in enumerate(np.arange(-4., 4.1, 2.)):
        fig, ax = plot_uphis(galah, galcen, i, off * km / s)
        fig.savefig("offset_uphi_{}_{}.pdf".format(i, j))
        plt.close(fig)

    i = 2
    for j, off in enumerate(np.arange(-20., 21., 10.)):
        fig, ax = plot_uphis(galah, galcen, i, off * sigunits)
        fig.savefig("offset_uphi_{}_{}.pdf".format(i, j))
        plt.close(fig)

    i = 3
    for j, off in enumerate(np.arange(-200., 201., 100.)):
        fig, ax = plot_uphis(galah, galcen, i, off * pc)
        fig.savefig("offset_uphi_{}_{}.pdf".format(i, j))
        plt.close(fig)

    print("__main__: done")

if False:

    # plot
    nx, ny = 2, 1
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 5, ny * 5), sharex=True, sharey=True)
    ax = ax.flatten()
    for i, q in enumerate([vmaxs, phis]):
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
    foo = ax.scatter(phis, vmaxs,
                     marker=".", s=3000/np.sqrt(len(vs)),
                     c=q, vmin=vmin, vmax=vmax, alpha=0.3,
                     cmap=mpl.cm.RdBu, rasterized=True)
    ax.set_xlabel(r"conjugate angle $\theta_z$ [rad]")
    ax.set_ylabel(r"$v_\mathrm{max}$ [km/s]")
    ax.set_xlim(0, 2. * np.pi)
    ax.set_ylim(0, np.max(vmaxs))
    fig.savefig("galah_foo.pdf")
    plt.close(fig)

    # plot angle plots
    nx, ny = 1, 8
    fig, ax = plt.subplots(ny, nx, figsize=(10, 10), sharex=True, sharey=True)
    q = vmaxs
    vmin, vmax = np.percentile(q, [0.5, 99.5])
    for i in range(ny):
        vmaxlims = np.percentile(vmaxs, [100 * i / ny, 100 * (i + 1) / ny])
        inside = (vmaxs > vmaxlims[0]) * (vmaxs < vmaxlims[1])
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

if False:

    # decide what and how to plot
    abundancenames = ["fe_h", "mg_fe", "o_fe", "al_fe", "mn_fe", "eu_fe"]
    abundancelabels = {}
    abundancelabels["fe_h"] = "[Fe/H]"
    abundancelabels["mg_fe"] = "[Mg/Fe]"
    abundancelabels["o_fe"] = "[O/Fe]"
    abundancelabels["al_fe"] = "[Al/Fe]"
    abundancelabels["mn_fe"] = "[Mn/Fe]"
    abundancelabels["eu_fe"] = "[Eu/Fe]"
    zmax = 1500. # pc
    vzmax = 75. # km / s

    # plot
    fig, ax = plot_some_abundances(galah, galcen)
    fig.savefig("galah_full_sample.pdf")
    plt.close(fig)
