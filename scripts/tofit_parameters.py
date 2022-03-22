# ============================ #
# edit this file to specify 
# parameters included in the 
# fitting
# ============================ #


# ==== parameters to fit ==== #
class tofit_param(object):
    def __init__(self, fit, defval):
        self.fit = fit
        self.default_val = defval
    #def add_prior(self, prlo, prhi):
    #    self.prlo = prlo
    #    self.prhi = prhi

        
# ==== parameters to fit ==== #
tofit_params = {
    # -------------------------------- #
    'velz':  tofit_param(True, 0.0), #0
    'sigma': tofit_param(True, 11.0), #1    
    'logage': tofit_param(True, 1.0), #2 
    'zh': tofit_param(True, 0.0), #3
    # -------------------------------- #
    'feh': tofit_param(True, 0.0), #4    
    'ah': tofit_param(True, 0.0), #5
    'ch': tofit_param(True, 0.0), #6
    'nh': tofit_param(True, 0.0), #7
    'nah': tofit_param(True, 0.0), #8
    'mgh': tofit_param(True, 0.0), #9
    'sih': tofit_param(True, 0.0), 
    'kh': tofit_param(True, 0.0), 
    'cah': tofit_param(True, 0.0), 
    'tih': tofit_param(True, 0.0), #13
    # -------------------------------- #
    'vh': tofit_param(True, 0.0), 
    'crh': tofit_param(True, 0.0), 
    'mnh': tofit_param(True, 0.0), 
    'coh': tofit_param(True, 0.0), 
    'nih': tofit_param(True, 0.0), 
    'cuh': tofit_param(True, 0.0), 
    'srh': tofit_param(True, 0.0), 
    'bah': tofit_param(True, 0.0), 
    'euh': tofit_param(True, 0.0), 
    # -------------------------------- #
    'teff': tofit_param(False, 0.0), 
    'imf1': tofit_param(True, 1.3), #24 
    'imf2': tofit_param(True, 2.3), #25
    'logfy': tofit_param(True, -3.9), #26
    'sigma2': tofit_param(True, 10.1), 
    'velz2': tofit_param(True, 0.0), 
    'logm7g': tofit_param(False, -5.5), #29
    'hotteff': tofit_param(True, 8.0), 
    'loghot': tofit_param(True, -3.0), 
    'fy_logage': tofit_param(True, -0.30), 
    # -------------------------------- #
    'logemline_h': tofit_param(True, -4.0), 
    'logemline_oii': tofit_param(True, -4.0), 
    'logemline_oiii': tofit_param(True, -4.0), 
    'logemline_sii': tofit_param(True, -4.0), 
    'logemline_ni': tofit_param(True, -4.0), 
    'logemline_nii': tofit_param(True, -4.0), 
    # -------------------------------- #
    'logtrans':  tofit_param(False, -5.9), #39
    'jitter': tofit_param(True, 1.0), 
    'logsky': tofit_param(False, -8.9), #41
    'imf3': tofit_param(False, 0.08+1e-5),   
    'imf4': tofit_param(False, 0.0), #43
    'h3': tofit_param(False, 0.0),   
    'h4': tofit_param(False, 0.0),
}