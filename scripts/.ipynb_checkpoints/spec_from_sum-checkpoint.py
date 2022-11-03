import numpy as np
from alf_vars import *
from linterp import *
from str2arr import *
from getmodel import *
from getm2l import *
from alf_constants import *
from setup import *

__all__ = ['spec_from_sum']


def spec_from_sum(filename, 
                  getsum = 'minchi2', returnspec = False, 
                  resdir = "{0}results".format(ALF_HOME), 
                  alfvar=None,):
    """
    !takes a *sum file as input and returns the corresponding 
    !model spectrum associated with min(chi^2)
    USE alf_vars; USE alf_utils
    USE nr, ONLY : gasdev,locate,powell,ran1
    USE ran_state, ONLY : ran_seed,ran_init
    """
    
    #!-----------------------------------------------------------!
    #!-----------------------------------------------------------!    
    try:
        f11 = np.loadtxt("{0}/{1}.sum".format(resdir, filename))
    except:
        print('ERROR, file not found: ', filename)
        
        
    #!-----------------------------------------------------------!
    #if in_alfvar is None:
    #    alfvar = pickle.load(open('alfvar_sspgrid_irldss3_imftype1.p', "rb" ))
    #else:
    #    alfvar = copy.deepcopy(in_alfvar)
    if alfvar is None:
        alfvar = ALFVAR()
        alfvar = setup(alfvar, onlybasic = False)    
        
    #read in the header to set the relevant parameters
    char='#'
    with open("{0}/{1}.sum".format(resdir, filename), "r") as myfile:
        temdata = myfile.readlines()
    for iline in temdata:
        if iline.split(' ')[0] == char:
            temline = np.array(iline.split()[1:])
            #print(temline)
            if 'mwimf' in temline:
                mwimf = int(temline[-1].split('\n')[0])
            if 'imf_type' in temline:
                imf_type = int(temline[-1].split('\n')[0])
            if 'fit_type' in temline:
                fit_type = int(temline[-1].split('\n')[0])
            if 'fit_two_ages' in temline:
                fit_two_ages = int(temline[-1].split('\n')[0]) 
            if 'fit_hermite' in temline:
                fit_hermite = int(temline[-1].split('\n')[0])
            if 'nonpimf_alpha' in temline:
                nonpimf_alpha = int(temline[-1].split('\n')[0])
            if 'Nwalkers' in temline:
                nwalkers = int(temline[-1].split('\n')[0])
            if 'Nchain' in temline:
                nchain = int(temline[-1].split('\n')[0])

    alfvar.mwimf = mwimf
    alfvar.imf_type = imf_type
    alfvar.fit_type = fit_type
    alfvar.fit_two_ages = fit_two_ages
    alfvar.fit_hermite = fit_hermite
    #alfvar.nonpimf_alpha = nonpimf_alpha
 
    #!-----------------------------------------------------------!
    alfvar.l1[0] = 0.0
    alfvar.l2[alfvar.nlint-1] = 1e5
    
    # mean burned in f90 code
    if getsum == 'mean':
        mean_ = f11[0]
    elif getsum == 'minchi2':
        mean_ = f11[1]
    elif getsum == 'cl50':
        mean_ = f11[5]
    elif getsum == 'cl16':
        mean_ = f11[4]
    elif getsum == 'cl84':
        mean_ = f11[6]
    
    d1 = np.copy(mean_[0])
    posarr = np.copy(mean_[1:47])
    mlx2 = np.copy(mean_[-6:])

    # ---- copy the input parameter array into the structure
    pos = str2arr(switch = 2, inarr = posarr) #arr->str

    # ---- !setup the models
    lam = alfvar.sspgrid.lam

    # ---- !we turn off the emission lines, since they are often highly
    # ---- !unconstrained if they are not included in the wavelength range
    pos.logemline_h    = -6.0
    pos.logemline_oii  = -6.0
    pos.logemline_oiii = -6.0
    pos.logemline_nii  = -6.0
    pos.logemline_sii  = -6.0
    pos.logemline_ni   = -6.0

    # ------------------------------------------------------------!
    # ---- !here is the place to make changes to the best-fit model,
    # ---- !if so desired
    # pos.loghot = -8.0 # it is commented out in f90 version
    # ------------------------------------------------------------!
    # get the model spectrum
    
    mspec = getmodel(pos, alfvar=alfvar)
    #m2l = getm2l(lam, mspec, pos, alfvar=alfvar) 

    # -- redshift the spectrum to observed frame
    oneplusz = (1+pos.velz/clight*1e5)
    zmspec   = linterp(lam*oneplusz, mspec, lam)
           
    if returnspec == True:
        return pos, alfvar, [lam, zmspec]
    
    else:
        alfname = '{0}models/{1}.bestspec2'.format(ALF_HOME, filename)
        np.savetxt(alfname, np.transpose([lam, zmspec]), 
                   delimiter="     ", 
                   fmt='   %12.4f   %12.4E')
        return pos, alfvar
    
    return pos, alfvar, mspec

