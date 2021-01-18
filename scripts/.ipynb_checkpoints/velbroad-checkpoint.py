from linterp import * 
import math, scipy, numpy as np
#copy, 
from alf_vars import *
from alf_constants import *

__all__ = ['velbroad']


# ------------------------------------------------------------------------- 
def velbroad_notfast(wave, spec, sigma, minl=None, maxl=None):
    """
    https://github.com/bd-j/prospector/blob/master/prospect/utils/smoothing.py
    """
    ckms = 2.99792458e5
    sigma_to_fwhm = 2.35482
    nsigma=10
    
    lnwave = np.log(wave)
    # sigma_eff is in units of sigma_lambda / lambda
    sigma_eff = sigma / ckms
    
    flux = np.zeros(wave.shape)
    for i, w in enumerate(wave):
        x = (np.log(w) - lnwave) / sigma_eff
        if nsigma > 0:
            good = np.abs(x) < nsigma
            x = x[good]
            _spec = spec[good]
        else:
            _spec = spec
        f = np.exp(-0.5 * x**2)
        flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)  
    if minl is not None:
        flux[wave<minl] = np.copy(spec[wave<minl])
    if maxl is not None:
        flux[wave>maxl] = np.copy(spec[wave>maxl])        
    return flux





#-------------------------------------------------------------------------!
def velbroad(lam, spec, sigma, minl=None, maxl=None, 
             ires=None, velbroad_simple = 1, alfvar=None):
    """
    !routine to compute velocity broadening of an input spectrum
    !the PSF kernel has a width of m*sigma, where m=4
    
    - If optional input ires is present, then the spectrum will be 
      smoothed by a wavelength dependent velocity dispersion in the
      'vebroad_simple=1' mode.
      
    - Note that the h3 and h4 coefficients are passed through the ires
      array as well, and in this mode the broadening is also in the "simple" mode
    - INPUTS:
        lambda
        spec
        sigma
        minl
        maxl
        (ires)
    - OUTPUTS:
        spec
    """
    lam = np.copy(lam)
    spec = np.copy(spec)
    
    if minl ==None:
        minl = np.nanmin(lam)
    if maxl ==None:
        maxl = np.nanmax(lam)    
    nn = lam.size

    m = 6
    h3 = 0.0
    h4 = 0.0

    #no broadening for small sigma
    if sigma <= 10.:
        return spec_
    elif sigma >= 1e4:
        print("VELBROAD ERROR: sigma>1E4 km/s - you've "\
              "probably done something wrong...")
        return spec_

    # ---- compute smoothing the slightly less accurate way
    # ---- but the **only way** in the case of wave-dep smoothing
    if (velbroad_simple==1) or (ires is not None):
        tspec = np.copy(spec)
        sigmal_arr_exist = False
        spec[np.logical_and(lam>=minl, lam<=maxl)] = 0.0 #??? check
        if (ires is not None):
            if ires.size > 2:
                sigmal_arr = ires
                h3 = 0.
                h4 = 0.
                sigmal_arr_exist = True

            elif ires.size == 2:
                sigmal = sigma
                h3 = ires[0]
                h4 = ires[1]
            else:
                sigmal = sigma
        else:
            sigmal = sigma
            
        if sigmal_arr_exist:
            xmax = lam * (m*sigmal_arr/clight*1e5+1)
        else:
            xmax = lam * (m*sigmal/clight*1e5+1)
            
        ih_arr = np.array([np.argmin(abs(lam - i)) for i in xmax])
        ih_arr[ih_arr>nn] = nn
        #il_arr = np.copy( 2*(np.arange(nn)+1) - (ih_arr+1))-1
        il_arr = 2*np.arange(nn) - ih_arr
        il_arr[il_arr<0] = 0
        useindex = np.arange(nn)[(lam>=minl)&(lam<=maxl)&(ih_arr != il_arr)]
  
        for i in useindex:
            if sigmal_arr_exist == True:
                sigmal = sigmal_arr[i]
            
            ih, il = ih_arr[i], il_arr[i]
            vel = (lam[i]/lam[il:ih]-1)*clight/1e5
            temr = vel/sigmal
            if h3==0 and h4==0:
                func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2)                
            else:
                func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2) * \
                    (1 + h3*(2*np.power(temr,3)-3*temr)/math.sqrt(3.) + 
                     h4*(4*np.power(temr,4)-12*np.square(temr)+3)/math.sqrt(24.) )
                
            func /= tsum(vel, func)
            spec[i] = tsum(vel, func*tspec[il:ih]) 
    
    
    # ---- !compute smoothing the correct way (convolution in dloglambda)
    else:
        # ---- !fancy footwork to allow for input spectra of either length
        if nn==alfvar.nl:
            n2 = alfvar.nl_fit
        else:
            n2 = alfvar.nl
        
        if alfvar.dlstep ==0:
            alfvar.dlstep = (math.log(alfvar.sspgrid.lam[-1])-math.log(alfvar.sspgrid.lam[0]))/alfvar.sspgrid.lam.size
            alfvar.lnlam = np.arange(alfvar.nl_fit)*alfvar.dlstep+math.log(alfvar.sspgrid.lam[0])
        fwhm = sigma*2.35482/clight*1e5/alfvar.dlstep
        psig = fwhm/2./math.sqrt(-2.0*math.log(0.5)) #! equivalent sigma for kernel

        grange = math.floor(m*psig) #! range for kernel (-range:range)
        if grange >1:
            tspec = linterp(np.log(lam[0:n2]), spec[0:n2], alfvar.lnlam[0:n2])
            nspec = np.zeros_like(tspec)
            psf = np.zeros(2*grange+1)
            for i in range(0, 2*grange+1):
                psf[i] = 1.0/math.sqrt(2*mypi)/psig*math.exp(-((i-grange)/psig)**2/2.0)
                
            psf= psf/np.nansum(psf)
            for i in range(grange, n2-grange):
                nspec[i] = np.nansum(psf*tspec[i-grange:i+grange+1])
                
            if len(spec)>n2:
                nspec[n2-grange:n2] = spec[n2-grange:n2]
            # ---- !interpolate back to the main array
            spec[0:n2] = linterp(np.exp(alfvar.lnlam[0:n2]),nspec[0:n2],lam[0:n2])    
    
    return spec
        
            


  



