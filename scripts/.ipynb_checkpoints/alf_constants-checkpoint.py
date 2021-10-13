"""
- Collection of constants used in alf
- It's the same as alf in fortran
"""
__all__ = ['ALF_HOME', 'mypi', 'clight', 'msun', 'lsun', 'pc2cm',
           'huge_number', 'tiny_number', 'key_list', 'default_arr', 'key_arr']
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

key_list = ['velz', 'sigma', 'logage', 'zh', 'feh',
            'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
            'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
            'teff','imf1','imf2','logfy','sigma2','velz2',
            'logm7g','hotteff','loghot','fy_logage',
            'logemline_h','logemline_oii','logemline_oiii',
            'logemline_sii','logemline_ni','logemline_nii',
            'logtrans','jitter','logsky', 'imf3','imf4','h3','h4']

key_arr = np.array(key_list)
default_arr = np.array([0.0, 11.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 1.3, 2.3, -4.0, 10.1, 0.0,
                        -5.5, 20.0, -4., 0.3,
                        -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
                        -5.5, 1.0, -8.5, 0.08, 0.0, 0.0, 0.0])
