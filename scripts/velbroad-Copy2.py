#from linterp import * 
#from alf_vars import *
import math, numpy as np
from numba import jit, njit
#from alf_constants import *



__all__ = ['velbroad']


# ------------------------------------------------------------------------- 
@njit(fastmath=True)
def find_nearest(array,value):
    """
    - much faster than argmin
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
    
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
        x = (math.log(w) - lnwave) / sigma_eff
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
@njit(fastmath=True)
def fast_smooth_part(lam, inspec, minl, maxl, h3, h4, sigmal, sigmal_arr=None):
    
    mypi = 3.14159265
    clight = 29979245800.0
    m = 6.0
    nn = len(lam)
            
    if sigmal_arr is not None:
        xmax = lam * (m*sigmal_arr/clight*1e5+1.0)
    else:
        xmax = lam * (m*sigmal/clight*1e5+1.0)
    #ih_arr = np.clip(np.array([np.argmin(np.abs(lam - i)) for i in xmax]), None, nn)
    ih_arr = np.clip(np.array([find_nearest(lam,i) for i in xmax]), None, nn)
    il_arr = np.clip(2*np.arange(nn) - ih_arr, 0, None)
    useindex = np.arange(nn)[(lam>=minl)&(lam<=maxl)&(ih_arr != il_arr)]
    outspec = np.copy(inspec)
    
    for i in useindex:
        ih, il = ih_arr[i], il_arr[i]
        if sigmal_arr is not None:
            sigmal = sigmal_arr[i]
        vel = (lam[i]/lam[il:ih]-1)*clight/1e5
        temr = vel/sigmal
        if (h3 == 0) and (h4 == 0):
            func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2) 
        else:
            func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2) * \
                    (1 + h3*(2*np.power(temr,3)-3*temr)/math.sqrt(3.) + \
                    h4*(4*np.power(temr,4)-12*np.square(temr)+3)/math.sqrt(24.) )
                
        func /= np.trapz(y=func, x=vel) #tsum(vel, func)
        outspec[i] = np.trapz(y=func*inspec[il:ih], x=vel)   
    return outspec
    
    
#-------------------------------------------------------------------------!
@njit(fastmath=True)  
def fast_smooth2_part(lam, inspec, ind1, ind2, sigma):
    m = 6.0
    mypi = 3.14159265
    clight = 29979245800.0
    n2 = ind2-ind1
    dlstep = (math.log(lam[ind2])-math.log(lam[ind1]))/n2
    lnlam = np.arange(n2)*dlstep+math.log(lam[ind1])
    outspec = np.copy(inspec)
    
    psig = sigma*2.35482/clight*1e5/dlstep/2./(-2.0*math.log(0.5))**0.5 #! equivalent sigma for kernel
    grange = math.floor(m*psig) #! range for kernel (-range:range)
    if grange >1:
        tspec = np.interp(x=lnlam, xp=np.log(lam[ind1:ind2]), fp= outspec[ind1:ind2])
        nspec = np.copy(tspec)
            
        psf = 1.0/math.sqrt(2*mypi)/psig*np.array([math.exp(-((i-grange)/psig)**2/2.0) for i in range(2*grange+1)]) 
        psf= psf/np.sum(psf)
            
        nspec[grange: n2-grange] = np.array([(psf*tspec[i-grange:i+grange+1]).sum() for i in range(grange, n2-grange)])
        outspec[ind1:ind2] = np.interp(x=lam[ind1:ind2], xp=np.exp(lnlam), fp=nspec)
            
    return outspec


#-------------------------------------------------------------------------!
@jit(nopython=True)
def velbroad(lam, spec, sigma, minl= None, maxl= None, 
             ires=None, velbroad_simple = 1):
    """
    !routine to compute velocity broadening of an input spectrum
    !the PSF kernel has a width of m*sigma, where m=4
    
    - If optional input ires is present, then the spectrum will be 
      smoothed by a wavelength dependent velocity dispersion in the
      'vebroad_simple=1' mode.
      
    - Note that the h3 and h4 coefficients are passed through the ires
      array as well, and in this mode the broadening is also in the "simple" mode
    - INPUTS:
        lambda, spec, sigma, minl, maxl, (ires)
    - only smooth between minl and maxl
    - OUTPUTS:
        spec
    """
    if minl is None :
        minl = np.nanmin(lam)
    if maxl is None:
        maxl = np.nanmax(lam)    
    m = 6.0
    nn = len(lam)
    h3, h4 = 0., 0.
    mypi = 3.14159265
    clight = 29979245800.0
    
    # ---- no broadening for small sigma
    if sigma <= 10.:
        return spec
    elif sigma >= 1e4:
        print("VELBROAD ERROR: sigma>1E4 km/s - you've "\
              "probably done something wrong...")
        return spec

    # ---- !compute smoothing the slightly less accurate way
    # ---- !but the **only way** in the case of wave-dep smoothing
    if (velbroad_simple==1) or (ires is not None):
        
        tspec = np.copy(spec)
        sigmal = sigma
        #sigmal_arr_exist = False
        if (ires is not None) and len(ires) == 2:
            h3 = ires[0]
            h4 = ires[1]
                
        if (ires is not None) and len(ires) == nn:
            sigmal_arr = np.copy(ires)
            
            """
            xmax = lam * (float(m)*sigmal_arr/clight*1e5+1)
            ih_arr = np.clip(np.array([find_nearest(lam,i) for i in xmax]), None, nn)
            il_arr = np.clip(2*np.arange(nn) - ih_arr, 0, None)
            useindex = np.arange(nn)[(lam>=minl)&(lam<=maxl)&(ih_arr != il_arr)]

            
            for i in useindex:
                ih, il, sigmal = ih_arr[i], il_arr[i], sigmal_arr[i]
                vel = (lam[i]/lam[il:ih]-1)*clight/1e5
                temr = vel/sigmal
                if h3 == h4 == 0:
                    func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2)                
                else:
                    func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2) * \
                           (1 + h3*(2*np.power(temr,3)-3*temr)/math.sqrt(3.) + \
                           h4*(4*np.power(temr,4)-12*np.square(temr)+3)/math.sqrt(24.) )
                
                func /= np.trapz(y=func, x=vel) #tsum(vel, func)
                spec[i] = np.trapz(y=func*tspec[il:ih], x=vel) #tsum(vel, func*tspec[il:ih]) 
            """
            
            #spec = fast_smooth_part(spec, lam, ih_arr, il_arr, sigmal, useindex, h3, h4, sigmal_arr)
            spec = fast_smooth_part(lam, spec, minl, maxl, h3, h4, sigmal, sigmal_arr=sigmal_arr)
               
        else:
            """
            xmax = lam * (m*sigmal/clight*1e5+1)
            #ih_arr = np.clip(np.array([np.argmin(np.abs(lam - i)) for i in xmax]), None, nn)
            ih_arr = np.clip(np.array([find_nearest(lam,i) for i in xmax]), None, nn)
            il_arr = np.clip(2*np.arange(nn) - ih_arr, 0, None)
            useindex = np.arange(nn)[(lam>=minl)&(lam<=maxl)&(ih_arr != il_arr)]
            
            
            for i in useindex:
                ih, il = ih_arr[i], il_arr[i]
                vel = (lam[i]/lam[il:ih]-1)*clight/1e5
                temr = vel/sigmal
                if h3 == h4 == 0:
                    func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2) 
                else:
                    func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-np.square(temr)/2) * \
                           (1 + h3*(2*np.power(temr,3)-3*temr)/math.sqrt(3.) + \
                           h4*(4*np.power(temr,4)-12*np.square(temr)+3)/math.sqrt(24.) )
                
                func /= np.trapz(y=func, x=vel) #tsum(vel, func)
                spec[i] = np.trapz(y=func*tspec[il:ih], x=vel) #tsum(vel, func*tspec[il:ih])  
            """       
            
            #spec = fast_smooth_part(spec, lam, ih_arr, il_arr, sigmal, useindex, h3, h4)
            spec = fast_smooth_part(lam, spec, minl, maxl, h3, h4, sigmal)

        
    # ---- !compute smoothing the correct way (convolution in dloglambda)
    else:
        # ---- !fancy footwork to allow for input spectra of either length       
        #ind1, ind2 = np.argmin(np.abs(lam-minl)), np.argmin(np.abs(lam-maxl))
        ind1 = find_nearest(lam,minl)
        ind2 = find_nearest(lam,maxl)
        spec = fast_smooth2_part(lam, spec, ind1, ind2, sigma)
        """
        n2 = ind2-ind1
        dlstep = (math.log(lam[ind2])-math.log(lam[ind1]))/n2
        lnlam = np.arange(n2)*dlstep+math.log(lam[ind1])
            
        #fwhm = sigma*2.35482/clight*1e5/dlstep
        psig = sigma*2.35482/clight*1e5/dlstep/2./(-2.0*math.log(0.5))**0.5 #! equivalent sigma for kernel
        grange = math.floor(m*psig) #! range for kernel (-range:range)
        if grange >1:
            tspec = np.interp(x=lnlam, xp=np.log(lam[ind1:ind2]), fp= spec[ind1:ind2])
            nspec = np.copy(tspec)
            
            psf = 1.0/math.sqrt(2*mypi)/psig*np.array([math.exp(-((i-grange)/psig)**2/2.0) for i in range(2*grange+1)]) 
            psf= psf/np.sum(psf)
            
            nspec[grange: n2-grange] = np.array([(psf*tspec[i-grange:i+grange+1]).sum() for i in range(grange, n2-grange)])
            #for i in range(grange, n2-grange):
            #    nspec[i] = (psf*tspec[i-grange:i+grange+1]).sum() # much faster than np.sum

            # ---- !interpolate back to the main array
            spec[ind1:ind2] = np.interp(x=lam[ind1:ind2], xp=np.exp(lnlam), fp=nspec)
            #linterp(np.exp(lnlam),nspec,lam[ind1:ind2]) 
        """     
    return spec
        
    
