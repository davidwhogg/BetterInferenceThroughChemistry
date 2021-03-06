# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from pyia import GaiaData

__all__ = ['load_nominal_galah', 'load_nominal_apogee',
           'get_label_from_abundancename',
           'get_catalog_name', 'get_abundance_data']

def load_nominal_galah(filename,
                       zlim=1*u.kpc, vzlim=100*u.km/u.s, dlim=1.5*u.kpc,
                       teff_lim=[3500., 6000]*u.K,
                       logg_lim=[-0.5, 3.5],
                       spec_snr_lim=16, parallax_snr_lim=8):
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
    g = GaiaData(filename)
    g = g[np.isfinite(g.parallax_error)]

    if parallax_snr_lim is not None:
        plx_snr = g.parallax / g.parallax_error
        g = g[plx_snr > parallax_snr_lim]

    if spec_snr_lim is not None:
        g = g[(g.snr_c1 > spec_snr_lim) &
              (g.snr_c2 > spec_snr_lim) &
              (g.snr_c3 > spec_snr_lim)]

    g = g[(g.teff > teff_lim[0]) & (g.teff < teff_lim[1])]
    g = g[(g.logg > logg_lim[0]) & (g.logg < logg_lim[1])]

    # make coordinates
    c = g.get_skycoord(radial_velocity=g.rv_synt)
    galcen = c.transform_to(coord.Galactocentric(z_sun=0*u.pc))

    zs = galcen.z.to(u.pc)
    vs = galcen.v_z.to(u.km/u.s)

    # trim on coordinates
    mask = np.isfinite(g.parallax) & (g.distance < dlim)
    mask &= (zs / zlim) ** 2 + (vs / vzlim) ** 2 < 1.
    g = g[mask]
    galcen = galcen[mask]

    return g, galcen


def load_nominal_apogee(filename,
                        zlim=2*u.kpc, vlim=75*u.km/u.s,
                        teff_lim=[4000., 6500]*u.K,
                        logg_lim=[-0.5, 3.5],
                        parallax_snr_lim=10):
    """TODO: duplicated code!"""

    # read data and cut
    g = GaiaData(filename)
    g = g[np.isfinite(g.parallax_error) & (g.parallax > 0.*u.mas)]

    if parallax_snr_lim is not None:
        plx_snr = g.parallax / g.parallax_error
        g = g[plx_snr > parallax_snr_lim]

    g = g[(g.TEFF > teff_lim[0].value) & (g.TEFF < teff_lim[1].value)]
    g = g[(g.LOGG > logg_lim[0]) & (g.LOGG < logg_lim[1])]

    # make coordinates
    c = g.get_skycoord(radial_velocity=g.VHELIO_AVG * u.km/u.s)
    galcen = c.transform_to(coord.Galactocentric(z_sun=0*u.pc))

    zs = galcen.z.to(u.pc)
    vs = galcen.v_z.to(u.km/u.s)

    # trim on coordinates
    inbox = (zs / zlim) ** 2 + (vs / vlim) ** 2 < 1.
    g = g[inbox]
    galcen = galcen[inbox]

    return g, galcen


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
    cat_names = np.array(data.__dir__())
    lower_names = np.array([x.lower() for x in cat_names])
    try:
        return cat_names[lower_names == name][0]
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
    if low_name == 'fe_h' or Y == 'fe' or Y == 'm':
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
