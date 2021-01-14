from alf_vars import *
from linterp import locate
from getmodel import *
from getm2l import *
import pickle, numpy as np

__all__ = ['write_a_model']

def write_a_model(alfvar = None):
    """
    !write a model to file
    """
    if alfvar is None:
        alfvar = pickle.load(open('alfvar_sspgrid_May26.p', "rb" ))    
    
    nl = alfvar.nl
    pos = ALFPARAM()
        
    #instrumental resolution (<10 -> no broadening)
    ires     = 1. #!100.
    alfvar.imf_type = 1
    alfvar.fit_type = 0
    str_ = np.array(['00','01','02','03','04','05','06','07','08','09'])
    
    #!initialize the random number generator
    np.random.seed()

    lmin = 3700.
    lmax = 11000.
    datmax = int(lmax-lmin)
    data = ALFTDATA(ndat = datmax)
    # force a constant instrumental resolution
    # needs to be done this way for setup.f90 to work
    for i in range(datmax):
        data.lam[i] = lmin + i
        data.ires[i] = ires
        
    # read in the SSPs and bandpass filters
    lam = alfvar.sspgrid.lam

    #!define the log wavelength grid used in velbroad.f90
    nl_fit = min(max(locate(lam,lmax+500.0),0),nl-1)
    dlstep = (np.log(alfvar.sspgrid.lam[nl_fit])-np.log(alfvar.sspgrid.lam[0]))/nl_fit
    
    lnlam = np.empty(nl_fit)
    for i in range(nl_fit):
        lnlam[i] = i*dlstep+np.log(alfvar.sspgrid.lam[0])
 
    alfvar.l1[0] = lmin
    alfvar.l2[1] = lmax

    #!loop to generate multiple mock datasets
    for j in range(1):
        #!string for indexing of filenames
        if j <= 10:
            is_ = str_[j+1]
        else:
            print(is_, j)

        file = 'test2.dat'
        pos.sigma  = 180.
        s2n  = 500.
        pos.logage = np.log10(10.)
        pos.zh     = -0.5
        pos.feh     = 0.0
        pos.ah     = 0.0
        pos.mgh     = 0.0
        emnorm     = -5.0

        pos.imf1   = 2.301
        pos.imf2   = 2.301
        pos.imf3   = 0.0801

        pos.logemline_h    = emnorm
        pos.logemline_oii  = emnorm
        pos.logemline_oiii = emnorm
        pos.logemline_sii  = emnorm
        pos.logemline_ni   = emnorm
        pos.logemline_nii  = emnorm

        #!get a model spectrum
        print('imf_type', alfvar.imf_type)
        mspec = getmodel(pos, alfvar = alfvar)
        print(mspec[:20])

        #!compute M/L
        print(lam.shape, mspec.shape)
        m2l = getm2l(lam,mspec,pos)
        #!print to screen
        print('m2l=', m2l)
        
        s2np_arr = np.ones(lam.shape)*s2n*np.sqrt(0.9)
        s2np_arr[lam>=7500] = s2n*np.sqrt(2.5)
        err = mspec/s2np_arr
        gspec = np.random.normal(mspec, err, nl)
 
        #!write model spectrum to file
        header = '0.400 0.470\n0.470 0.560\n0.570 0.640'
        idx = np.logical_and(lam >= lmin, lam<= lmax)
        np.savetxt("{0}models/{1}".format(ALF_HOME, file),
                   np.transpose([lam[idx], gspec[idx], err[idx], 
                                 np.ones(lam.shape)[idx], np.ones(idx.sum())*ires]), 
                   delimiter="     ", fmt='   %12.4f   %12.4E   %12.4E   %12.4f   %12.4f' ,
                   header = header)
        
    return pos
