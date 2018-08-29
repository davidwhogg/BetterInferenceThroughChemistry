# Standard library
import os

# Third-party
import astropy.units as u
import numpy as np
import pytest
from pyia import GaiaData

# This package
from ..potential import *

def test_sech2():
