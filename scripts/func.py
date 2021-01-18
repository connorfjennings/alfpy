import numpy as np
from linterp import *
#from alf_vars import *
from str2arr import *
from getm2l import *
from getmass import getmass
from getmodel import getmodel
from set_pinit_priors import *
from contnormspec import *
from alf_constants import *

__all__ = ['func',]


key_list = ['velz', 'sigma', 'logage', 'zh', 'feh', 
                'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
                'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
                'teff','imf1','imf2','logfy','sigma2','velz2',
                'logm7g','hotteff','loghot','fy_logage',
                'logemline_h','logemline_oii','logemline_oiii',
                'logemline_sii','logemline_ni','logemline_nii',
                'logtrans','jitter','logsky', 'imf3','imf4','h3','h4']

# ---------------------------------------------------------------- #
def func(alfvar, in_posarr, prhiarr = None, prloarr=None, 
         spec=False, funit=None, usekeys = key_list):
    """  
    !routine to get a new model and compute chi^2.  Optionally,
    !the model spectrum is returned (spec).  The model priors
    !are computed in this routine.
    USE alf_vars; USE nr, ONLY : locate
    USE alf_utils, ONLY : linterp3,contnormspec,getmass,&
         str2arr,getmodel,linterp,getindx,getm2l
    - FUNCTION FUNC(nposarr,spec,funit)
    - USE alf_vars
    - (spec)
    - INPUTS: nposarr, (funit)
    - OUTPUTS: nposarr, func_val (func in f90)
    """

    data = alfvar.data
    l1, l2 = alfvar.l1, alfvar.l2    
    
    # ---------------------------------------------------------------- #    
    if prhiarr is None or prloarr is None:
        _, prlo, prhi = set_pinit_priors(alfvar)
        prloarr = str2arr(switch=1, instr = prlo)
        prhiarr = str2arr(switch=1, instr = prhi)
    
    # ---------------------------------------------------------------- #
    func_val = 0.0

    # --- !this is for Powell minimization
    #if nposarr.size < alfvar.npar:
    #    # ---- !copy over the default parameters first
    #    tposarr = str2arr(switch=1, instr = npos, usekeys=usekeys) #!str->arr
    #    # ---- !only copy the first four params (velz,sigma,age,[Z/H])
    #    tposarr[:alfvar.npowell] = nposarr[:alfvar.npowell]
    #else:
    #    tposarr = np.copy(nposarr)
    
    #npos = str2arr(2, inarr = in_posarr, usekeys=usekeys) #arr->str
    #nposarr = str2arr(1, instr = npos, usekeys=usekeys)
    nposarr = fill_param(in_posarr, usekeys = usekeys)
    npos = str2arr(2, inarr = nposarr)
    # ---------------------------------------------------------------- #
    # ---- !compute priors (don't count all the priors if fitting
    # ---- !in (super) simple mode or in powell fitting mode)
    pr = 1.0
    for i in range(alfvar.npar):
        if np.logical_and(i > alfvar.npowell, 
                          np.logical_or(alfvar.powell_fitting == 1,
                                        alfvar.fit_type == 2)):
            continue
        if (alfvar.fit_type == 1 and i > alfvar.nparsimp):
            continue
        if (alfvar.fit_indices == 1 and i <= 2):
            continue
        if (nposarr[i] > prhiarr[i]) or (nposarr[i] < prloarr[i]):
            pr=0.0
    

    # ---------------------------------------------------------------- #
    # ---- !regularize the non-parametric IMF
    # ---- !the IMF cannot be convex (U shaped)
    if (alfvar.imf_type == 4 and alfvar.nonpimf_regularize == 1):
        if np.logical_and(npos.imf2 - npos.imf1+ alfvar.corr_bin_weight[2]-alfvar.corr_bin_weight[0] < 0.0,
                          npos.imf3 - pos.imf2 + alfvar.corr_bin_weight[4]-alfvar.corr_bin_weight[2] > 0.0) :
             pr=0.0
            
        if np.logical_and(npos.imf3-npos.imf2+alfvar.corr_bin_weight[4]-alfvar.corr_bin_weight[2] < 0.0,
                          npos.imf4-npos.imf3+alfvar.corr_bin_weight[6]-alfvar.corr_bin_weight[4] > 0.0
                         ):
             pr = 0.0
            
        if np.logical_and(npos.imf4-npos.imf3+alfvar.corr_bin_weight[6]-alfvar.corr_bin_weight[4] < 0.0,
                          0.0-npos.imf4+alfvar.corr_bin_weight[8]-alfvar.corr_bin_weight[6] > 0.0): 
             pr = 0.0


    # ---- !only compute the model and chi2 if the priors are >0.
    if (pr > tiny_number):
        # ---- !get a new model spectrum
        mspec = getmodel(npos, alfvar=alfvar)
    else:
        return np.inf
     
    if spec == True: 
        spec_ = np.copy(mspec)

    # ---------------------------------------------------------------- #
    # ---- include external M/L prior (assuming I-band)
    if (alfvar.extmlpr == 1): # ??? check
        mlalf = getm2l(alfvar.sspgrid.lam, mspec, npos)
        klo  = max(min(locate(alfvar.mlprtab[0:alfvar.nmlprtabmax+1,0],mlalf[1]),alfvar.nmlprtabmax-1),0)
        dk   = (mlalf[1]-alfvar.mlprtab[klo,0])/(alfvar.mlprtab[klo+1,0]-alfvar.mlprtab[klo,0])
        dk   = max(min(dk,1.0),0.0) # ---- !no extrapolation
        mlpr = dk*alfvar.mlprtab[klo+1,1] + (1-dk)*alfvar.mlprtab[klo,1]
        pr   = pr*mlpr


    if (alfvar.fit_indices == 0) :
        # ---- !redshift the model and interpolate to data wavelength array
        oneplusz = (1+npos.velz/clight*1e5)
        zmspec = linterp(alfvar.sspgrid.lam[:alfvar.nl_fit]*oneplusz,
                         mspec[:alfvar.nl_fit],
                         alfvar.data.lam[:alfvar.datmax])


        # ---- !compute chi2, looping over wavelength intervals
        for i in range(alfvar.nlint):
            tl1 = max(l1[i]*oneplusz, data.lam[0])
            tl2 = min(l2[i]*oneplusz, data.lam[-1])
            ml  = (tl1+tl2)/2.0
            # ---- !if wavelength interval falls completely outside 
            # ---- !of the range of the data, then skip
            if tl1 >= tl2:
                continue

            i1 = min(max(locate(data.lam, tl1),0),len(data.lam)-2)
            i2 = min(max(locate(data.lam, tl2),1),len(data.lam)-1)

            #print('tl1, tl2, i1, i2 =', tl1,',', tl2,',', i1,',', i2, )
            # ---- !fit a polynomial to the ratio of model and data
            npow, tcoeff, poly = contnormspec(data.lam[:alfvar.datmax], 
                                    data.flx[:alfvar.datmax]/zmspec, 
                                    data.err[:alfvar.datmax]/zmspec, 
                                    tl1, tl2, 
                                    coeff = True, 
                                    return_poly = True)
            mflx  = zmspec * poly
            
            # ---- !compute chi^2
            flx = np.copy(data.flx[i1:i2])
            err = np.copy(data.err[i1:i2])
            sky = np.copy(data.sky[i1:i2])
            model = np.copy(mflx[i1:i2])
                
            if (alfvar.fit_type == 0):
                # ---- !include jitter term
                sky_term = np.square(np.power(10,npos.logsky)*sky)
                err_term = np.square(err)*npos.jitter**2
                tocal_1 = np.square(flx-model)/(err_term + sky_term)
                tocal_2 = np.log(2*mypi*(err_term + sky_term)) 
                tchi2 = np.nansum(tocal_1 + tocal_2)
            else:
                # ---- !no jitter in simple mode
                tchi2 = np.nansum( np.square(flx-model)/np.square(err) )
            
           
            # ---- !error checking
            if (~np.isfinite(tchi2)):
                print(" FUNC ERROR: chi2 returned a NaN") 
                print(" error occured at wavelength interval: ",i)
                print( 'lam  data   err   model   poly')
                for j in range(i1,i2):
                    print("    {0}    {1}    {2}    {3}    {4}".format(data.lam[j],data.flx[j],
                          np.sqrt(data.err[j]**2*npos.jitter**2+(10**npos.logsky*data.sky[j])**2),
                          mflx[j],poly[j])
                         )
                print("\nparams:", tposarr)
              
            else:
                func_val  += tchi2
        
            if funit is not None:
                # ---- write final results to screen and file
                print("%.2f um - %.2f um:  rms: %.2f percetn,Chi2/dof: %.2f" 
                      %(tl1/1e4/oneplusz,tl2/1e4/oneplusz,
                        np.sqrt(np.nansum( (flx/model-1)**2 )/(i2-i1+1) )*100,
                        tchi2/(i2-i1)))
                
                if (alfvar.fit_type == 0):
                    terr = np.sqrt(np.square(data.err)*npos.jitter**2+np.square(np.power(10,npos.logsky)*data.sky))
                else:
                    terr = data.err

                for j in range(i1,i2):
                    print(funit, data.lam[j],mflx[j],
                          data.flx[j],data.flx[j]/terr[j],poly[j],data.err[j])

    # #!include priors (func_val is chi^2)
    # ---- use priors.py instead
    # if (pr <= tiny_number):
    #     func_val = huge_number 
    # else: 
    #     func_val = func_val - 2*np.log(pr)

    if spec == False:
        return func_val
    elif spec == True:
        return func_val, spec_
