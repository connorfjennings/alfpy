import math, numpy as np
from numba import jit
import scipy.constants
from linterp import locate

__all__ = ['velbroad']

mypi   = 3.14159265 #scipy.constants.pi  # pi
clight = 2.9979E10#scipy.constants.speed_of_light*100  # speed of light (cm/s)


# -------------------------------------------------------------------------
@jit(nopython=True, fastmath=True)
def prospect_smooth_vel(wave, spec, outwave, sigma, nsigma=10, inres=0):
    """
    https://github.com/bd-j/prospector/blob/master/prospect/utils/smoothing.py
    """
    m = 6.0
    nn = len(lam)           
    if sigmal_arr is not None:
        xmax = lam * (m*sigmal_arr/clight*1e5+1.0)
    else:
        xmax = lam * (m*sigmal/clight*1e5+1.0)

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
        sq_temr = np.square(temr)
        if (h3 == 0) and (h4 == 0):
            func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-sq_temr/2) 
        else:
            func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-sq_temr/2) * \
                    (1 + h3*(2*fast_np_power(temr,3)-3*temr)/math.sqrt(3.) + \
                    h4*(4*fast_np_power(temr,4)-12*sq_temr+3)/math.sqrt(24.) )
                
        func /= np.trapz(y=func, x=vel) #tsum(vel, func)
        outspec[i] = np.trapz(y=func*inspec[il:ih], x=vel) 


# -------------------------------------------------------------------------
@jit(nopython=True, fastmath=True)
def fast_np_power(x1, x2):
    # ---- only slightly faster ---- #
    return x1**x2

# ------------------------------------------------------------------------- 
@jit(nopython=True,fastmath=True)
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
    

#-------------------------------------------------------------------------!
@jit(nopython=True,fastmath=True)
def fast_smooth1_part(lam, inspec, minl, maxl, h3, h4, sigmal, sigmal_arr=None):
    
    m = 6.0
    nn = len(lam)           
    if sigmal_arr is not None:
        xmax = lam * (m*sigmal_arr/clight*1e5+1.0)
    else:
        xmax = lam * (m*sigmal/clight*1e5+1.0)

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
        sq_temr = np.square(temr)
        if (h3 == 0) and (h4 == 0):
            func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-sq_temr/2) 
        else:
            func = 1./math.sqrt(2.*mypi)/sigmal * np.exp(-sq_temr/2) * \
                    (1 + h3*(2*fast_np_power(temr,3)-3*temr)/math.sqrt(3.) + \
                    h4*(4*fast_np_power(temr,4)-12*sq_temr+3)/math.sqrt(24.) )
                
        func /= np.trapz(y=func, x=vel) #tsum(vel, func)
        outspec[i] = np.trapz(y=func*inspec[il:ih], x=vel)   
    return outspec
    
    
#-------------------------------------------------------------------------!
@jit(nopython=True,fastmath=True)  
def fast_smooth2_part(lam, inspec, ind1, ind2, sigma, nl_fit):
    m = 6.0
    n2 = ind2-ind1
    
    loglam_min = math.log(lam[0])
    dlstep = (math.log(lam[nl_fit])-loglam_min)/(nl_fit+1)
    lnlam = np.arange(1, nl_fit+2)*dlstep + loglam_min
    outspec = np.copy(inspec)
    
    fwhm = sigma*2.35482/clight*1e5/dlstep
    psig = fwhm/2./(-2.0*math.log(0.5))**0.5 #! equivalent sigma for kernel
    grange = math.floor(m*psig) #! range for kernel (-range:range)
    
    
    if grange >1:
        tspec = np.interp(lnlam, np.log(lam), outspec)
        nspec = np.copy(tspec)
            
        psf = 1.0/math.sqrt(2*mypi)/psig*np.array([math.exp(-((i-grange)/psig)**2/2.0) for i in range(2*grange+1)]) 
        psf= psf/psf.sum()

        nspec[grange: n2-grange] = np.array([(psf*tspec[i-grange:i+grange+1]).sum() for i in range(grange, n2-grange)])
        outspec[ind1:ind2] = np.interp(x=lam[ind1:ind2], xp=np.exp(lnlam[ind1:ind2]), fp=nspec[ind1:ind2])
            
    return outspec



#-------------------------------------------------------------------------!
@jit(nopython=True,fastmath=True) 
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
    - velbroad_simple = 1: very consistent with prospect - smoothing.smoothspec, fftsmooth=False
    """
    if minl is None :
        minl = lam.min()
    if maxl is None:
        maxl = lam.max()   
    m = 6.0
    nn = len(lam)
    h3, h4 = 0., 0.
    
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
        
        #tspec = np.copy(spec)
        sigmal = sigma
        if (ires is not None) and len(ires) == 2:
            h3 = ires[0]
            h4 = ires[1]
                
        if (ires is not None) and len(ires) == nn:
            sigmal_arr = np.copy(ires)
            spec = fast_smooth1_part(lam, spec, minl, maxl, h3, h4, 
                                     sigmal, sigmal_arr=sigmal_arr)
               
        else:
            spec = fast_smooth1_part(lam, spec, minl, maxl, h3, h4, sigmal)

        
    # ---- !compute smoothing the correct way (convolution in dloglambda)
    else:
        # ---- !fancy footwork to allow for input spectra of either length       
        ind1 = max(locate(lam, minl-500), 0)
        ind2 = min(locate(lam, maxl+500), len(lam))
        nl_fit = min(max(locate(lam, maxl+500), 0), len(lam)-1)
        spec = fast_smooth2_part(lam, spec, ind1, ind2, sigma, nl_fit)
   
    return spec
        
    
