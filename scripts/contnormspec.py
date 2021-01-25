import numpy as np
from linterp import locate
import math

__all__ = ['contnormspec']
# ------------------------------------------------------------------------- 
def npoly(x,arr):
    for i in range(arr.size):
        arr[i] = x**i
    return arr


# ------------------------------------------------------------------------- 
def contnormspec(lam, flx, err, il1, il2, coeff=False, return_poly=False):
    """
    !routine to continuum normalize a spectrum by a high-order
    !polynomial.  The order of the polynomial is determined by
    !n=(lam_max-lam_min)/100.  Only normalized over the input
    !min/max wavelength range
    #, lam,flx,err,il1,il2,flxout,coeff=None
    return: normed spectra
    """
    
    npolymax = 10
    poly_dlam = 100.
    
    buff = 0.0
    #mask = np.ones(npolymax+1)
    #covar = np.empty((npolymax+1, npolymax+1))
    
    n1 = lam.size
    flxout = np.copy(flx)

    # ---- !divide by a power-law of degree npow. one degree per poly_dlam.
    # ---- !don't let things get out of hand (force Npow<=npolymax)

    npow = min(int((il2-il1)/poly_dlam), npolymax)
    i1 = min(max(locate(lam, il1-buff),0), n1-2)
    i2 = min(max(locate(lam, il2+buff),1), n1-1)+1
    ml = (il1+il2)/2.0
    
    #!simple linear least squares polynomial fit
    res = np.polyfit(x = lam[i1:i2]-ml, 
                     y = flx[i1:i2], 
                     deg = npow, 
                     full = True, 
                     w = 1./np.square(err[i1:i2]), 
                     cov = True)
    
    covar = res[2]
    chi2sqr = res[1]
    tcoeff = res[0]
    
    p = np.poly1d(tcoeff)
    poly = p(lam-ml)
    
    if coeff == False and return_poly==False:
        return npow
    
    elif coeff == True and return_poly==False:
        return npow, tcoeff    
    
    elif coeff == True and return_poly==True:
        return npow, tcoeff, poly

