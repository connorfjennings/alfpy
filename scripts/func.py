import numpy as np
from linterp import *
from str2arr import *
from getmodel import getmodel, fast_np_power
from set_pinit_priors import *
from contnormspec import *
from alf_constants import *

__all__ = ['func']

# ---------------------------------------------------------------- #
def func(alfvar, in_posarr, usekeys, prhiarr = None, prloarr=None, 
         funit=False):
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
    - updates:
        - remove datmax (not very useful here, but it's kept in alfvar as 
          we read the data, unless we need to cut the data in alfvar (unlikely))
    """

    data = alfvar.data
    l1, l2 = alfvar.l1, alfvar.l2    
    
    # ---------------------------------------------------------------- #    
    if prhiarr is None or prloarr is None:
        _, prlo, prhi = set_pinit_priors(alfvar.imf_type)
        prloarr = str2arr(switch=1, instr = prlo)
        prhiarr = str2arr(switch=1, instr = prhi)
    
    # ---------------------------------------------------------------- #
    func_val = 0.0
    nposarr = fill_param(in_posarr, usekeys = usekeys)
    npos = str2arr(2, inarr = nposarr)
    # ---------------------------------------------------------------- #
    # ---- !compute priors (don't count all the priors if fitting
    # ---- !in (super) simple mode or in powell fitting mode)
    pr = 1.0
    
    if (nposarr>prhiarr).sum() + (nposarr<prloarr).sum() >0:
        pr = 0.0

    # ---------------------------------------------------------------- #
    # ---- !regularize the non-parametric IMF
    # ---- !the IMF cannot be convex (U shaped)
    if (alfvar.imf_type == 4 and alfvar.nonpimf_regularize == 1):
        if (npos.imf2 - npos.imf1+ alfvar.corr_bin_weight[2]-alfvar.corr_bin_weight[0] < 0.0) & \
        (npos.imf3 - npos.imf2 + alfvar.corr_bin_weight[4]-alfvar.corr_bin_weight[2] > 0.0) :
            
            pr=0.0
            
        if (npos.imf3-npos.imf2+alfvar.corr_bin_weight[4]-alfvar.corr_bin_weight[2] < 0.0) & \
        (npos.imf4-npos.imf3+alfvar.corr_bin_weight[6]-alfvar.corr_bin_weight[4] > 0.0):
             pr = 0.0
            
        if (npos.imf4-npos.imf3+alfvar.corr_bin_weight[6]-alfvar.corr_bin_weight[4] < 0.0) & \
        (0.0-npos.imf4+alfvar.corr_bin_weight[8]-alfvar.corr_bin_weight[6] > 0.0): 
             pr = 0.0


    # ---- !only compute the model and chi2 if the priors are >0.
    if (pr > tiny_number):
        # ---- !get a new model spectrum
        #mspec = getmodel_grid(npos, alfvar=alfvar)
        mspec = getmodel(npos, alfvar=alfvar)
        if np.isnan(mspec).any():
            return np.inf
    else:
        return np.inf

    # ---------------------------------------------------------------- #

    if (alfvar.fit_indices == 0) :
        # ---- !redshift the model and interpolate to data wavelength array
        oneplusz = (1+npos.velz/clight*1e5)
        zmspec = linterp(alfvar.sspgrid.lam[:alfvar.nl_fit]*oneplusz,
                         mspec[:alfvar.nl_fit],
                         alfvar.data.lam)

        
        # ---- !compute chi2, looping over wavelength intervals
        datasize = len(data.lam)
        
        tchi2_list = np.zeros((alfvar.nlint))
        for i in range(alfvar.nlint):
            tl1 = max(l1[i]*oneplusz, data.lam[0])
            tl2 = min(l2[i]*oneplusz, data.lam[-1])

            # ---- !if wavelength interval falls completely outside 
            # ---- !of the range of the data, then skip
            if tl1 >= tl2:
                continue

            i1 = min(max(locate(data.lam, tl1),0), datasize-2)
            i2 = min(max(locate(data.lam, tl2),1), datasize-1)+1

            # ---- !fit a polynomial to the ratio of model and data
            npow, tcoeff, poly = contnormspec(data.lam, data.flx/zmspec, 
                                 data.err/zmspec, tl1, tl2, 
                                 coeff = True, return_poly = True)
            mflx  = zmspec * poly
            # ---- !compute chi^2 ---- #
            ind = np.isfinite(data.flx[i1:i2])
            flx_i12 = data.flx[i1:i2][ind].copy()
            err_i12 = data.err[i1:i2][ind].copy()
            sky_i12 = data.sky[i1:i2][ind].copy()
            lam_i12 = data.lam[i1:i2][ind].copy()
            mflx_i12 = mflx[i1:i2][ind].copy()
            poly_i12 = poly[i1:i2][ind].copy()
                
            if alfvar.fit_type == 0:
                # ---- !include jitter term
                sky_term = np.square(fast_np_power(10,npos.logsky)*sky_i12)
                err_term = np.square(err_i12*npos.jitter)
                tocal_1 = np.square(flx_i12-mflx_i12)/(err_term + sky_term)
                tocal_2 = np.log(2*mypi*(err_term + sky_term)) 
                tchi2 = np.nansum(tocal_1 + tocal_2)
            else:
                # ---- !no jitter in simple mode
                tchi2 = np.nansum( np.square(flx_i12-mflx_i12)/np.square(err_i12) )
            
           
            # ---- !error checking
            if (~np.isfinite(tchi2)):
                return np.inf

            func_val  += tchi2
        
        
            if funit is True:
                if (alfvar.fit_type == 0):
                    terr = np.sqrt(sky_term + err_term)
                else:
                    terr = data.err
                    
                len_i12 = len(lam_i12)
                combine_i12 = np.concatenate((lam_i12.reshape(1, len_i12), mflx_i12.reshape(1, len_i12),
                                              flx_i12.reshape(1, len_i12), (flx_i12/terr).reshape(1, len_i12), 
                                              poly_i12.reshape(1, len_i12), err_i12.reshape(1, len_i12),), axis=0)
                
                if i==0:
                    outspec_arr = np.copy(combine_i12)
                else:
                    outspec_arr = np.hstack((outspec_arr, combine_i12))
                    
                # ---- write final results to screen and file
                print("%.2f um - %.2f um:  rms: %.2f percetn,Chi2/dof: %.2f" 
                      %(tl1/1e4/oneplusz, tl2/1e4/oneplusz,np.sqrt(np.nansum( (flx_i12/mflx_i12-1)**2 )/(i2-i1+1) )*100, tchi2/(i2-i1)))


    if funit == False:
        return func_val
    elif funit == True:
        return func_val, outspec_arr
