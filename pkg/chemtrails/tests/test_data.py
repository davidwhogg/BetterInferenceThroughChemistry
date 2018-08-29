# Standard library
import os

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from pyia import GaiaData

# This package
from ..data import (load_nominal_galah, get_label_from_abundancename,
                    get_catalog_name, get_abundance_data)

galah_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                          'galah-small.fits')
def test_load_galah():
    data, galcen = load_nominal_galah(galah_path, parallax_snr_lim=None)

    assert len(data) == 16
    assert isinstance(data, GaiaData)

    # test filters
    data, galcen = load_nominal_galah(galah_path, logg_lim=[2, 4])
    assert data.logg.min() > 2.
    assert data.logg.max() < 4.

    data, galcen = load_nominal_galah(galah_path, teff_lim=[4000, 5000]*u.K)
    assert data.teff.min() > 4000*u.K
    assert data.teff.max() < 5000*u.K

    data, galcen = load_nominal_galah(galah_path, parallax_snr_lim=20)
    assert np.all((data.parallax / data.parallax_error) > 20.)


def test_get_label():
    label = get_label_from_abundancename('fe_h')
    assert label == '[Fe/H]'

    label = get_label_from_abundancename('FE_H')
    assert label == '[Fe/H]'

    label = get_label_from_abundancename('O_FE')
    assert label == '[O/Fe]'


def test_get_catalog_name():
    # GALAH
    data, galcen = load_nominal_galah(galah_path)

    name = get_catalog_name(data, 'fe_h')
    assert name == 'fe_h'

    name = get_catalog_name(data, 'FE_H')
    assert name == 'fe_h'

    name = get_catalog_name(data, 'o_fe')
    assert name == 'o_fe'

    name = get_catalog_name(data, 'ba_ba')
    assert name is None

    # TODO: test APOGEE


def test_get_abundance_data():
    # GALAH
    data, galcen = load_nominal_galah(galah_path)

    abun = get_abundance_data(data, 'fe_h')
    assert np.all(abun == data.fe_h)

    abun = get_abundance_data(data, 'o_fe')
    assert np.all(abun == data.o_fe)

    abun = get_abundance_data(data, 'o_al')
    assert np.allclose(abun, (data.o_fe - data.al_fe))

    abun = get_abundance_data(data, 'o_h')
    assert np.allclose(abun, (data.o_fe + data.fe_h))

    with pytest.raises(AttributeError):
        abun = get_abundance_data(data, 'was_sup')

    # TODO: test APOGEE
