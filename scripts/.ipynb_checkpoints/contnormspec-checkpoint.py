import numpy as np
from linterp import locate
import math
from numba import jit
#from velbroad import find_nearest

__all__ = ['contnormspec']
# ------------------------------------------------------------------------- 
#@jit(nopython=True)
#def npoly(x,arr):
#    for i in range(arr.size):
#        arr[i] = x**i
#    return arr

# ------------------------------------------------------------------------- 
@jit(nopython=True, fastmath=True)
def tmp_cal(lam, il1, il2, npow=None, npolymax = 10):
    # ---- !divide by a power-law of degree npow. one degree per poly_dlam.
    # ---- !don't let things get out of hand (force Npow<=npolymax)
    poly_dlam = 100.
    buff = 0.0
    n1 = lam.size
    if npow is None:
        # ---- use -1 to be consistent with alf ---- # 
        # ---- -> returned coeff should have the same length ---- #
        npow = max(1, min((il2-il1)//poly_dlam, npolymax) - 1)  
    i1 = min(max(locate(lam, il1-buff),0), n1-2)
    i2 = min(max(locate(lam, il2+buff),1), n1-1)   
    ml = (il1+il2)/2.0

    return i1, i2, ml, npow


# ------------------------------------------------------------------------- 
def contnormspec(lam, flx, err, il1, il2, coeff=False, return_poly=False, 
                 npolymax = 10, npow = None):
    """
    !routine to continuum normalize a spectrum by a high-order
    !polynomial.  The order of the polynomial is determined by
    !n=(lam_max-lam_min)/100.  Only normalized over the input
    !min/max wavelength range
    #, lam,flx,err,il1,il2,flxout,coeff=None
    return: normed spectra
    """
    i1, i2, ml, npow = tmp_cal(lam, il1, il2, npow, npolymax)
    
    #!simple linear least squares polynomial fit
    ind = np.isfinite(flx[i1:i2])
    res = np.polyfit(x = lam[i1:i2][ind]-ml, 
                     y = flx[i1:i2][ind], 
                     deg = npow, full = True, 
                     w = 1./(err[i1:i2][ind]**2), 
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

