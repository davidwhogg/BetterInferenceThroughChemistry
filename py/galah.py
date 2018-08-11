"""
This file is part of the ChemicalTangents project.
Copyright 2018 David W. Hogg (MPIA).
"""

from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyia import GaiaData

def plot_some_abundances(galah, galcen):
    nx, ny = 3, 2
    fig, ax = plt.subplots(ny, nx, figsize=(15, 10), sharex=True, sharey=True)
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
    return fig.tight_layout()

if __name__ == "__main__":

    # read data and cut
    galah = GaiaData('../data/GALAH-GaiaDR2-xmatch.fits.gz')
    galah = galah[np.isfinite(galah.parallax_error)]
    galah = galah[(galah.parallax / galah.parallax_error) > 10.]
    galah = galah[np.isfinite(galah.teff) & (galah.teff > 4000*u.K) & (galah.teff < 6500*u.K)]
    galah = galah[np.isfinite(galah.logg) & (galah.logg < 3.5)]

    # make coordinates
    c = galah.get_skycoord(radial_velocity=galah.rv_synt)
    galcen = c.transform_to(coord.Galactocentric(z_sun=0*u.pc))

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
    foo = plot_some_abundances(galah, galcen)
    plt.savefig("galah_full_sample.pdf")
