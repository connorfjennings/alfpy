from linterp import * 
import copy, scipy, numpy as np
from alf_vars import *
clight = 29979245800.0
mypi = 3.1415926535898

__all__ = ['velbroad']


# ------------------------------------------------------------------------- 
def velbroad_notfast(wave, spec, sigma, minl=None, maxl=None, inres=None):
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
def velbroad_old(lam, spec, sigma, minl=None, maxl=None, 
             ires=None, velbroad_simple = 1):
    """
    !routine to compute velocity broadening of an input spectrum
    !the PSF kernel has a width of m*sigma, where m=4
    !If optional input ires is present, then the spectrum will be
    !smoothed by a wavelength dependent velocity dispersion in the
    !'vebroad_simple=1' mode.
    !Note that the h3 and h4 coefficients are passed through the ires
    !array as well, and in this mode the broadening is also in the "simple" mode
    """
    lam = np.copy(lam)
    spec = np.copy(spec)
    
    
    tiny_number = 1e-33
    if minl ==None:
        minl = np.nanmin(lam)
    if maxl ==None:
        maxl = np.nanmax(lam)    
    nn = lam.size
    
    vel,func = np.empty((2, nn))
    m = 6
    h3 = 0.0
    h4 = 0.0

    #no broadening for small sigma
    if sigma <= 10.:
        return
    elif sigma >= 1e4:
        print("VELBROAD ERROR: sigma>1E4 km/s - you've "\
              "probably done something wrong...")

    #compute smoothing the slightly less accurate way
    #but the only way in the case of wave-dep smoothing
    if (velbroad_simple==1) or (ires is not None):
        tspec = np.copy(spec)
        sigmal_arr_exist = False
        
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
            
            
        #ih   = MIN(locate(lam(1:nn),xmax),nn)
        #il   = MAX(2*i-ih,1)
        ih_arr = np.array([np.argmin(abs(lam - i)) for i in xmax])
        ih_arr[ih_arr>nn] = nn
        il_arr = np.copy( 2*(np.arange(nn)+1) - (ih_arr+1))-1
        il_arr[il_arr<0] = 0
    
        useindex = np.arange(nn)[(lam>=minl)&(lam<=maxl)&(ih_arr != il_arr)]
        for i in useindex:
            if sigmal_arr_exist:
                sigmal = sigmal_arr[i]
            
            ih, il = ih_arr[i]+1, il_arr[i]
            vel[il:ih] = (lam[i]/lam[il:ih]-1)*clight/1e5

            #Gauss-Hermite expansion:  based on Cappellari (2017) Eqn 13, 15
            func[il: ih] = 1./np.sqrt(2.*mypi)/sigmal * np.exp(-vel[il:ih]**2./2/sigmal**2) * \
            (1 + h3*(2*(vel[il:ih]/sigmal)**3-3*(vel[il:ih]/sigmal))/np.sqrt(3.) + 
             h4*(4*(vel[il:ih]/sigmal)**4-12*(vel[il:ih]/sigmal)**2+3)/np.sqrt(24.) )

            #normalize the weights to integrate to unity
            func[il:ih] = func[il:ih] / tsum(vel[il:ih],func[il:ih])
            spec[i]     = tsum(vel[il:ih],func[il:ih]*tspec[il:ih])

      
    # ---------------------------------------------------------------- #
    #compute smoothing the correct way (convolution in dloglambda)
    else:
        nspec,tspec, psf = np.empty((3, nn))
        alfvar = ALFVAR()
        #fancy footwork to allow for input spectra of either length
        if nn == alfvar.nl:
            n2 = alfvar.nl_fit
        else:
            n2 = alfvar.nl

        fwhm   = sigma*2.35482/clight*1e5/alfvar.dlstep
        psig   = fwhm/2./np.sqrt(-2.*np.log(0.5)) # equivalent sigma for kernel
        grange = int(m*psig) #range for kernel (-range:range)

        if grange >1:
            tspec[:] = 0.0
            nspec[:] = 0.0
            tspec[0:n2] = linterp(np.log(lam[:n2]),spec[:n2],alfvar.lnlam[:n2])

        for i in range(2*grange+1):
            psf[i] = 1./np.sqrt(2.*mypi)/psig*np.exp(-((i-grange-1)/psig)**2/2.)
        psf[:2*grange+1] /= np.nansum(psf[:2*grange+1])

        for i in range(grange,n2-grange): # CHECK
            nspec[i] = np.nansum( psf[:2*grange+1]*tspec[i-grange:i+grange] )
        nspec[n2-grange:n2] = spec[n2-grange:n2]

        #interpolate back to the main array
        spec[:n2] = linterp(np.exp(alfvar.lnlam[:n2]),nspec[:n2],lam[:n2])

    return spec



#-------------------------------------------------------------------------!
def velbroad(lam, spec, sigma, minl=None, maxl=None, 
             ires=None, velbroad_simple = 1):
    """
    !routine to compute velocity broadening of an input spectrum
    !the PSF kernel has a width of m*sigma, where m=4
    !If optional input ires is present, then the spectrum will be
    !smoothed by a wavelength dependent velocity dispersion in the
    !'vebroad_simple=1' mode.
    !Note that the h3 and h4 coefficients are passed through the ires
    !array as well, and in this mode the broadening is also in the "simple" mode
    """
    lam = np.copy(lam)
    spec = np.copy(spec)
    
    
    tiny_number = 1e-33
    if minl ==None:
        minl = np.nanmin(lam)
    if maxl ==None:
        maxl = np.nanmax(lam)    
    nn = lam.size
    
    vel,func = np.empty((2, nn))
    m = 6
    h3 = 0.0
    h4 = 0.0

    #no broadening for small sigma
    if sigma <= 10.:
        return
    elif sigma >= 1e4:
        print("VELBROAD ERROR: sigma>1E4 km/s - you've "\
              "probably done something wrong...")

    #compute smoothing the slightly less accurate way
    #but the only way in the case of wave-dep smoothing
    if (velbroad_simple==1) or (ires is not None):
        tspec = np.copy(spec)
        sigmal_arr_exist = False
        
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
        il_arr = np.copy( 2*(np.arange(nn)+1) - (ih_arr+1))-1
        il_arr[il_arr<0] = 0
                                    
        useindex = np.arange(nn)[(lam>=minl)&(lam<=maxl)&(ih_arr != il_arr)]
        if sigmal_arr_exist == False:
            def temfunc(i, ih_arr=ih_arr, il_arr=il_arr, tspe = tspec):
                ih, il = ih_arr[i]+1, il_arr[i]
                vel = (lam[i]/lam[il:ih]-1)*clight/1e5
            
                func = 1./np.sqrt(2.*mypi)/sigmal * np.exp(-vel**2./2/sigmal**2) * \
                (1 + h3*(2*(vel/sigmal)**3-3*(vel/sigmal))/np.sqrt(3.) + 
                 h4*(4*(vel/sigmal)**4-12*(vel/sigmal)**2+3)/np.sqrt(24.) )
            
                func /= tsum(vel, func)
                return tsum(vel, func*tspec[il:ih])
        
            spec[useindex] = np.array(list(map(temfunc, useindex)))
            
        else:
            def temfunc2(i, ih_arr=ih_arr, il_arr=il_arr, tspe = tspec, sigma_arr = sigmal_arr):
                sigmal = sigmal_arr[i]
                ih, il = ih_arr[i]+1, il_arr[i]
                vel = (lam[i]/lam[il:ih]-1)*clight/1e5
            
                func = 1./np.sqrt(2.*mypi)/sigmal * np.exp(-vel**2./2/sigmal**2) * \
                (1 + h3*(2*(vel/sigmal)**3-3*(vel/sigmal))/np.sqrt(3.) + 
                 h4*(4*(vel/sigmal)**4-12*(vel/sigmal)**2+3)/np.sqrt(24.) )
            
                func /= tsum(vel, func)
                return tsum(vel, func*tspec[il:ih])
        
            spec[useindex] = np.array(list(map(temfunc2, useindex)))      
    

    return spec
        
            


  



