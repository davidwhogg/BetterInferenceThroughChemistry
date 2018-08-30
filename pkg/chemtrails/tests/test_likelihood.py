# Standard library
import os

# Third-party
from astropy.constants import G
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np

# This package
from ..data import load_nominal_galah
from ..likelihood import Model

def test_compare_hogg():
    # Set up paths to import Hogg's code:
    from os import path
    import sys
    _pwd = path.split(path.abspath(__file__))[0]
    _path = path.abspath(path.join(_pwd, '../../..', 'py'))
    if _path not in sys.path:
        sys.path.append(_path)
    from integrate_orbits import paint_energies
    from chemical_tangents import ln_like

    # Load a subset of the data
    galah_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                              'galah-small.fits')
    data, galcen = load_nominal_galah(galah_path, parallax_snr_lim=8.)

    Sigma = 65 * u.Msun/u.pc**2
    hz = 250 * u.pc
    sunz = 8.2 * u.pc
    sunvz = -1.4 * u.pc/u.Myr

    par_dict = {
        'sun_z': sunz.to(u.pc).value,
        'sun_vz': sunvz.to(u.pc/u.Myr).value,
        'lnsigma': np.log(Sigma.to(u.Msun/u.pc**2).value),
        'lnhz': np.log(hz.to(u.pc).value)
    }

    model = Model(galcen, data, ['fe_h'])
    E_apw = model.get_energy(par_dict) * u.pc**2/u.Myr**2

    atzs = galcen.z.to(u.m).value
    atvs = galcen.v_z.to(u.m/u.s).value
    sunpars = [sunz.si.value, sunvz.si.value]
    dynpars = [Sigma.si.value, hz.si.value]
    E_dwh = paint_energies(atzs, atvs, sunpars, dynpars) * u.m**2/u.s**2

    assert quantity_allclose(E_apw, E_dwh)

    # now check likelihood evaluation
    qs = model.element_data['fe_h']
    ll_apw = model.ln_metal_likelihood(qs, E_apw.value)
    ll_dwh = ln_like(qs, E_dwh.to(E_apw.unit).value,
                     order=model.metals_deg)
    assert quantity_allclose(ll_apw, ll_dwh)
