# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from pyia import GaiaData

__all__ = ['load_nominal_galah', 'get_label_from_abundancename',
           'get_catalog_name', 'get_abundance_data']

def load_nominal_galah(filename,
                       zlim=2*u.kpc, vlim=75*u.km/u.s,
                       teff_lim=[4000., 6500]*u.K,
                       logg_lim=[-0.5, 3.5],
                       parallax_snr_lim=10):
    """TODO

    Parameters
    ----------
    filename : str
    zlim : `astropy.units.Quantity`
    vlim : `astropy.units.Quantity`

    Returns
    -------
    galah : `pyia.GaiaData`
    galcen : `astropy.coordinates.Galactocentric`

    """

    # read data and cut
    galah = GaiaData(filename)
    galah = galah[np.isfinite(galah.parallax_error)]

    if parallax_snr_lim is not None:
        plx_snr = galah.parallax / galah.parallax_error
        galah = galah[plx_snr > parallax_snr_lim]

    galah = galah[(galah.teff > teff_lim[0]) & (galah.teff < teff_lim[1])]
    galah = galah[(galah.logg > logg_lim[0]) & (galah.logg < logg_lim[1])]

    # make coordinates
    c = galah.get_skycoord(radial_velocity=galah.rv_synt)
    galcen = c.transform_to(coord.Galactocentric(z_sun=0*u.pc))

    zs = galcen.z.to(u.pc)
    vs = galcen.v_z.to(u.km/u.s)

    # trim on coordinates
    inbox = (zs / zlim) ** 2 + (vs / vlim) ** 2 < 1.
    galah = galah[inbox]
    galcen = galcen[inbox]

    return galah, galcen


def get_label_from_abundancename(name):
    """Given a name from a spectroscopic catalog (e.g., fe_h), return a label
    like [Fe/H].

    Parameters
    ----------
    name : str
        Abundance name from a catalog, like fe_h.
    """
    num, den = name.split("_")
    return "[{}/{}]".format(num.title(), den.title())


def get_catalog_name(data, name):
    """Given a name like fe_h or FE_H, get the catalog version of the name.

    Parameters
    ----------
    data : `pyia.GaiaData`
        The source data catalog.
    name : str
        A name like 'fe_h' or 'FE_H' to be compared against the names in the
        catalog.

    Returns
    -------
    catalog_name : str
    """

    name = name.lower()
    lower_names = np.array([x.lower() for x in data.__dir__()])
    try:
        return lower_names[lower_names == name][0]
    except IndexError:
        return None


def get_abundance_data(data, name):
    """Retrieve abundance data from the source catalog. This supports
    computing element ratios that aren't explicitly in the catalog, e.g.,
    [O/Al], as long as [O/Fe] and [Al/Fe] both exist.

    Parameters
    ----------
    data : `pyia.GaiaData`
        The source data catalog.
    name : str
        A name like 'fe_h' or 'FE_H' to be compared against the names in the
        catalog.

    Returns
    -------
    elems : `numpy.ndarray`
    """

    low_name = name.lower()
    X, Y = low_name.split("_")

    # short-circuit if we just want [Fe/H], or an element [X/Fe]
    if low_name == 'fe_h' or Y == 'fe':
        cat_name = get_catalog_name(data, low_name)
        return np.array(getattr(data, cat_name))

    # other element combos:
    X_name = get_catalog_name(data, X + '_fe')
    if Y == 'h':
        Y_name = get_catalog_name(data, 'fe_h')
        sign = 1.

    else:
        Y_name = get_catalog_name(data, Y + '_fe')
        sign = -1.

    if X_name is None or Y_name is None:
        raise AttributeError('Cannot construct abundance "{0}" from source '
                             ' data'.format(name))

    return np.array(getattr(data, X_name) + sign*getattr(data, Y_name))
