"""
- Collection of constants used in alf
- It's the same as alf in fortran
"""
__all__ = ['ALF_HOME', 'mypi', 'clight', 'msun', 'lsun', 'pc2cm',
           'huge_number', 'tiny_number']
import os, numpy as np
import scipy.constants
ALF_HOME = os.environ['ALF_HOME']

# ---- from alf_var.py ---- #
# ---- Physical Constants ---- !
# ---- in cgs units where applicable ---- !
mypi   = scipy.constants.pi  # pi
clight = scipy.constants.speed_of_light*1e2  # speed of light (cm/s)
msun   = 1.989e33  # Solar mass in grams
lsun   = 3.839e33  # Solar luminosity in erg/s
pc2cm  = 3.08568e18  # cm in a pc
huge_number = 1e33  # define small and large numbers
tiny_number = 1e-33

