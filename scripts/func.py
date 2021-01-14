import numpy as np
from linterp import *
from alf_vars import *
from str2arr import *
from getm2l import *
from getmass import getmass
from getmodel import getmodel
from set_pinit_priors import *
from contnormspec import *
contnormspec
__all__ = ['func',]

def func(alfvar, nposarr, spec=None, funit=None):
    """
    !routine to get a new model and compute chi^2.  Optionally,
    !the model spectrum is returned (spec).  The model priors
    !are computed in this routine.
    USE alf_vars; USE nr, ONLY : locate
    USE alf_utils, ONLY : linterp3,contnormspec,getmass,&
         str2arr,getmodel,linterp,getindx,getm2l
    - INPUTS: nposarr, (funit)
    - OUTPUTS: nposarr, func
    """
    tiny_number = 1e-33
    huge_number = 1e33
    clight = 2.9979e10
    mypi = alfvar.mypi

    nfil = alfvar.nfil
    nl = alfvar.nl
    ndat = alfvar.ndat
    npar = alfvar.npar
    npolymax = alfvar.npolymax
    nindx = alfvar.nindx
    nl_fit = alfvar.nl_fit
    npowell = alfvar.npowell
    fit_type = alfvar.fit_type
    fit_indices = alfvar.fit_indices
    imf_type = alfvar.imf_type
    powell_fitting = alfvar.powell_fitting
    extmlpr = alfvar.extmlpr
    sspgrid = alfvar.sspgrid
    data = alfvar.data
    datmax = alfvar.datmax
    nlint = alfvar.nlint
    l1, l2 = alfvar.l1, alfvar.l2
    poly_dlam,npolymax = alfvar.poly_dlam, alfvar.npolymax
    
    # ---------------------------------------------------------------- #
    #mlalf = np.empty(nfil)
    #mspec = np.empty(nl)
    poly,terr = np.empty((2, ndat))
    mindx = np.empty(nindx)
    #npos = ALFPARAM()
    
    
    # ---------------------------------------------------------------- #    
    _, prlo, prhi = set_pinit_priors(alfvar)
    prhi.logm7g = -5.0
    prhi.teff   =  2.0
    prlo.teff   = -2.0
    prloarr = str2arr(switch=1, instr = prlo)
    prhiarr = str2arr(switch=1, instr = prhi)
    
    # ---------------------------------------------------------------- #
    func = 0.0
    tpow = 0

    #!this is for Powell minimization
    if nposarr.size < npar:
        #!copy over the default parameters first
        tposarr = str2arr(switch=1, instr = npos) #!str->arr
        #!only copy the first four params (velz,sigma,age,[Z/H])
        tposarr[:npowell] = nposarr[:npowell]
    else:
        tposarr = np.copy(nposarr)

    npos = str2arr(2, inarr = tposarr) #arr->str
    print(npos.__dict__)

    # ---------------------------------------------------------------- #
    #!compute priors (don't count all the priors if fitting
    #!in (super) simple mode or in powell fitting mode)
    pr = 1.0
    for i in range(npar):
        if np.logical_and(i > npowell, powell_fitting == 1 or fit_type == 2):
            continue
        if (fit_type == 1 and i > alfvar.nparsimp):
            continue
        if (fit_indices == 1 and i <= 2):
            continue
        if (nposarr[i] > prhiarr[i]) or (nposarr[i] < prloarr[i]):
            print('bad pos at ', i, nposarr[i])
            pr=0.0
    

    # ---------------------------------------------------------------- #
    #!regularize the non-parametric IMF
    #!the IMF cannot be convex (U shaped)
    if (imf_type == 4 and alfvar.nonpimf_regularize == 1):
        if no.logical_and(npos.imf2 - npos.imf1+ corr_bin_weight[2]-corr_bin_weight[0] < 0.0,
                          npos.imf3 - pos.imf2 + corr_bin_weight[4]-corr_bin_weight[2] > 0.0) :
             pr=0.0
            
        if np.logical_and(npos.imf3-npos.imf2+corr_bin_weight[4]-corr_bin_weight[2] < 0.0,
                          npos.imf4-npos.imf3+corr_bin_weight[6]-corr_bin_weight[4] > 0.0
                         ):
             pr = 0.0
            
        if np.logical_and(npos.imf4-npos.imf3+corr_bin_weight[6]-corr_bin_weight[4] < 0.0,
                          0.0-npos.imf4+corr_bin_weight[8]-corr_bin_weight[6] > 0.0): 
             pr = 0.0


    #!only compute the model and chi2 if the priors are >0.0
    print('pr=', pr)
    if (pr > tiny_number):
        #!get a new model spectrum
        mspec = getmodel(npos)
     
    if spec is not None: 
        spec = np.copy(mspec)


    # ---------------------------------------------------------------- #
    #include external M/L prior (assuming I-band)
    if (extmlpr == 1):
        mlalf = getm2l(sspgrid.lam, mspec, npos)
        klo  = max(min(locate(mlprtab[0:nmlprtabmax,1],mlalf[1]),nmlprtabmax-2),0)
        dk   = (mlalf[1]-mlprtab[klo,0])/(mlprtab[klo+1,0]-mlprtab[klo,0])
        dk   = max(min(dk,1.0),0.0) #!no extrapolation
        mlpr = dk*mlprtab[klo+1,1] + (1-dk)*mlprtab[klo,1]
        pr   = pr*mlpr


    if (fit_indices == 0) :
        #!redshift the model and interpolate to data wavelength array
        oneplusz = (1+npos.velz/clight*1e5)
        zmspec = linterp(sspgrid.lam[:nl_fit]*oneplusz,
                         mspec[:nl_fit],
                         data.lam)

        #!compute chi2, looping over wavelength intervals
        for i in range(nlint):
            tl1 = max(l1[i]*oneplusz,data.lam[0])
            tl2 = min(l2[i]*oneplusz,data.lam[-1])
            ml  = (tl1+tl2)/2.0
            #!if wavelength interval falls completely outside 
            #!of the range of the data, then skip
            if tl1 >= tl2:
                continue

            i1 = min(max(locate(data.lam,tl1),1),len(data.lam)-2)
            i2 = min(max(locate(data.lam,tl2),2),len(data.lam)-1)

            print('tl1, tl2, i1, i2 =', tl1,',', tl2,',', i1,',', i2, )
            #!fit a polynomial to the ratio of model and data
            poly, tcoeff, poly = contnormspec(data.lam, 
                                              data.flx/zmspec, 
                                              data.err/zmspec, 
                                              tl1, tl2, 
                                              coeff = True, 
                                              return_poly = True)
            

            mflx  = zmspec * poly
            #return mflx
            #!compute chi^2
            flx = np.copy(data.flx[i1:i2])
            err = np.copy(data.err[i1:i2])
            sky = np.copy(data.sky[i1:i2])
            model = np.copy(mflx[i1:i2])
            err[~np.isfinite(flx)] = huge_number
            err[~np.isfinite(err)] = huge_number
            flx[~np.isfinite(flx)] = model[~np.isfinite(flx)]
            flx[~np.isfinite(err)] = model[~np.isfinite(err)]
                
            if (fit_type == 0):
                #!include jitter term
                tchi2 = np.nansum( (flx-model)**2 / (err**2*npos.jitter**2+(10**npos.logsky*sky)**2) +
                                  np.log(2*mypi*err**2*npos.jitter**2 + (10**npos.logsky*sky**2)) 
                                 )
            else:
                #!no jitter in simple mode
                tchi2 = np.nansum( (flx-mflx[i1:i2])**2/err**2 )
                
            print('tchi2=', tchi2)

           
            #!error checking
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

              
            if np.isfinite(tchi2):
                func  += tchi2
        
            if funit is not None:
                #write final results to screen and file
                print("{0}um -{1}um:  rms:{2},Chi2/dof:{3}".format(tl1/1e4/oneplusz,
                                                                    tl2/1e4/oneplusz,
                                                                    np.sqrt(np.nansum( (data.flx[i1:i2]/mflx[i1:i2]-1)**2 )/(i2-i1+1) )*100,tchi2/(i2-i1)
                                                                   ))
                if (fit_type == 0):
                    terr = np.sqrt(data.err**2*npos.jitter**2+(10**npos.logsky*data.sky)**2)
                else:
                    terr = data.err

                for j in range(i1,i2):
                    print(funit, data.lam[j],mflx[j],
                          data.flx[j],data.flx[j]/terr[j],poly[j],data.err[j])


    #else:
        #!compute indices
        #CALL GETINDX(sspgrid%lam,mspec,mindx)
        #!compute chi^2 for the indices
        #func = np.nansum( (data_indx%indx-mindx)**2/data_indx%err**2 )

        #if funit is not None:
        #    for j in range(nindx):
        #        print(funit,'(F8.2,3F9.4)') (indxdef(1,j)+indxdef(2,j))/2.,&
        #           mindx(j),data_indx(j)%indx,data_indx(j)%err


    #!include priors (func is chi^2)
    if (pr <= tiny_number):
        func = huge_number 
    else: 
        func = func - 2*np.log(pr)

    return func
