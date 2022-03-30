# ==================================== #
# - edit this file to specify parameters 
#   included in the fitting
# ==================================== #
# ======== parameters to fit ========= #
# - (included or not, default value) - #
tofit_params = {
    # -------------------------------- #
    'velz':  (True, 0.0), #0
    'sigma': (True, 11.0), 
    'logage': (True, 1.0), 
    'zh': (True, 0.0), 
    # -------------------------------- #
    'feh': (True, 0.0), #4
    'ah': (True, 0.0), 
    'ch': (True, 0.0), 
    'nh': (True, 0.0), 
    'nah': (True, 0.0), 
    'mgh': (True, 0.0), 
    'sih': (True, 0.0),
    'kh': (True, 0.0),
    'cah': (True, 0.0),
    'tih': (True, 0.0), 
    # -------------------------------- #
    'vh': (True, 0.0), #14
    'crh': (True, 0.0),
    'mnh': (True, 0.0),
    'coh': (True, 0.0),
    'nih': (True, 0.0),
    'cuh': (True, 0.0),
    'srh': (True, 0.0),
    'bah': (True, 0.0),
    'euh': (True, 0.0),
    # -------------------------------- #
    'teff': (False, 0.0), #23, # killed off in alf.f90
    'imf1': (True, 1.3), 
    'imf2': (True, 2.3), 
    'logfy': (True, -5.9), 
    'sigma2': (True, 10.1),
    'velz2': (True, 0.0),
    'logm7g': (False, -6.+1e-5), # killed off in alf.f90
    'hotteff': (True, 8.0+1e-5),
    'loghot': (True, -6.0+1e-5),
    'fy_logage': (True, 0.3),
    # -------------------------------- #
    'logemline_h': (True, -6.0+1e-5), #33
    'logemline_oii': (True, -6.0+1e-5),
    'logemline_oiii': (True, -6.0+1e-5),
    'logemline_sii': (True, -6.0+1e-5),
    'logemline_ni': (True, -6.0+1e-5),
    'logemline_nii': (True, -6.0+1e-5), 
    # -------------------------------- #
    'logtrans': (False, -5.9), #39 
    'jitter': (True, 1.0), 
    'logsky': (False, -5.9), 
    'imf3': (True, 0.08+1e-5), 
    'imf4': (False, 0.0), 
    'h3': (False, 0.0),
    'h4': (False, 0.0),
}
