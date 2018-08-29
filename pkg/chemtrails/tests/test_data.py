# Standard library
import os

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from pyia import GaiaData

# This package
from ..data import load_nominal_galah

def test_load_galah():
    path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                        'galah-small.fits')
    data, galcen = load_nominal_galah(path, parallax_snr_lim=None)

    assert len(data) == 16
    assert isinstance(data, GaiaData)

    # test filters
    data, galcen = load_nominal_galah(path, logg_lim=[2, 4])
    assert data.logg.min() > 2.
    assert data.logg.max() < 4.

    data, galcen = load_nominal_galah(path, teff_lim=[4000, 5000]*u.K)
    assert data.teff.min() > 4000*u.K
    assert data.teff.max() < 5000*u.K

    data, galcen = load_nominal_galah(path, parallax_snr_lim=20)
    assert np.all((data.parallax / data.parallax_error) > 20.)
