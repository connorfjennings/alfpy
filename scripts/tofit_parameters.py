# ==================================== #
# edit this file to specify parameters 
# included in the fitting
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
    'nah': (True, 0.5), 
    'mgh': (True, 0.0), 
    'sih': (True, 0.0),
    'kh': (True, 0.0),
    'cah': (True, 0.0),
    'tih': (True, 0.0), 
    # -------------------------------- #
    'vh': (False, 0.0), #14
    'crh': (False, 0.0),
    'mnh': (False, 0.0),
    'coh': (False, 0.0),
    'nih': (False, 0.0),
    'cuh': (False, 0.0),
    'srh': (False, 0.0),
    'bah': (False, 0.0),
    'euh': (False, 0.0),
    # -------------------------------- #
    'teff': (False, 0.0), #23
    'imf1': (True, 1.3), 
    'imf2': (True, 2.3), 
    'logfy': (True, -5.9), 
    'sigma2': (True, 10.1),
    'velz2': (True, 0.0),
    'logm7g': (True, -5.5), 
    'hotteff': (True, 20.0),
    'loghot': (True, -4.0),
    'fy_logage': (True, 0.3),
    # -------------------------------- #
    'logemline_h': (False, -6.0+1e-5), #33
    'logemline_oii': (False, -6.0+1e-5),
    'logemline_oiii': (False, -6.0+1e-5),
    'logemline_sii': (False, -6.0+1e-5),
    'logemline_ni': (False, -6.0+1e-5),
    'logemline_nii': (False, -6.0+1e-5), 
    # -------------------------------- #
    'logtrans':  (False, -5.9), #39 
    'jitter': (True, 1.0), 
    'logsky': (False, -8.9), 
    'imf3': (False, 0.08+1e-5), 
    'imf4': (False, 0.0), 
    'h3': (False, 0.0),
    'h4': (False, 0.0),
}
