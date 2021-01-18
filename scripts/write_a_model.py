from alf_vars import *
from linterp import locate
from getmodel import *
from getm2l import *
import pickle, numpy as np
from alf_constants import *

__all__ = ['write_a_model']

def write_a_model(alfvar = None):
    """
    !write a model to file
    """
    if alfvar is None:
        alfvar = pickle.load(open('../../alfvar_sspgrid_irldss3_imftype3_full.p', "rb" ))  
    
    nl = alfvar.nl
    pos = ALFPARAM()
        
    #instrumental resolution (<10 -> no broadening)
    ires     = 1.0 #!100.
    alfvar.imf_type = 1
    alfvar.fit_type = 0
    str_ = np.array(['00','01','02','03','04','05','06','07','08','09'])
    
    #!initialize the random number generator
    #np.random.seed()

    lmin = 3700.0
    lmax = 11000.0
    datmax = int(lmax-lmin)
    data = ALFTDATA(ndat = datmax)
    # ---- force a constant instrumental resolution
    # ---- needs to be done this way for setup.f90 to work
    data.lam = np.arange(datmax) + lmin
    data.ires[:] = ires
        
    # ---- read in the SSPs and bandpass filters
    lam = alfvar.sspgrid.lam

    # ---- !define the log wavelength grid used in velbroad.f90
    alfvar.nl_fit = min(max(locate(lam,lmax+500.0),0),alfvar.nl-1)
    alfvar.dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-np.log(alfvar.sspgrid.lam[0]))/alfvar.nl_fit
    alfvar.lnlam = np.arange(alfvar.nl_fit)*alfvar.dlstep + np.log(alfvar.sspgrid.lam[0])
    alfvar.l1[0] = lmin
    alfvar.l2[1] = lmax

    # ---- !loop to generate multiple mock datasets
    for j in range(1):
        # ---- !string for indexing of filenames
        if j <= 10:
            is_ = str_[j+1]
        else:
            print(is_, j)

        file = 'alfpy_test1.dat'
        pos.sigma  = 200.
        s2n  = 500.
        pos.logage = np.log10(10.)
        pos.zh     = -0.3
        pos.feh     = 0.1
        pos.ah     = 0.3
        pos.mgh     = 0.4
        emnorm     = -5.0

        pos.imf1   = 2.501
        pos.imf2   = 2.501
        pos.imf3   = 0.0801

        pos.logemline_h    = emnorm
        pos.logemline_oii  = emnorm
        pos.logemline_oiii = emnorm
        pos.logemline_sii  = emnorm
        pos.logemline_ni   = emnorm
        pos.logemline_nii  = emnorm

        # ---- !get a model spectrum
        mspec = getmodel(pos, alfvar = alfvar)

        # ---- !compute M/L
        m2l = getm2l(lam,mspec,pos, mw=0, alfvar = alfvar)
        # ---- !print to screen
        print('m2l=', m2l)
        
        s2np_arr = np.ones(lam.shape)*s2n*np.sqrt(0.9)
        s2np_arr[lam>=7500] = s2n*np.sqrt(2.5)
        err = mspec/s2np_arr
        gspec = np.random.normal(mspec, err, nl)
 
        #!write model spectrum to file
        header = '0.400 0.470\n0.470 0.560'
        idx = np.logical_and(lam >= lmin, lam<= lmax)
        np.savetxt("{0}models/{1}".format(ALF_HOME, file),
                   np.transpose([lam[idx], gspec[idx], err[idx], 
                                 np.ones(lam.shape)[idx], np.ones(idx.sum())*ires]), 
                   delimiter="     ", fmt='   %12.4f   %12.4E   %12.4E   %12.4f   %12.4f' ,
                   header = header)
        
    return pos
