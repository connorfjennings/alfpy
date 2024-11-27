from numba import jit
__all__ = ['airtovac', 'vactoair']

#---------------------------------------------------------------!
@jit(nopython=True, fastmath=True)
def airtovac(lam):
    """
    !convert wavelengths from air to vac
    !see Morton (1991 Ap.J. Suppl. 77, 119)
    !this code was adapted from the IDL routine airtovac.pro
    """
    
    #nn = lam.size
    
    #Convert to wavenumber squared
    sigma2 = (1e4/lam)**2.   

    #!Compute conversion factor
    fact = 1. + 6.4328e-5 + 2.94981e-2/(146. - sigma2) + 2.5540e-4/(41. - sigma2)
    #fact = 1 + (5.792105e-2/(238.0185-sigma2)) + (1.67918e-3/(57.362-sigma2))
  
    #!no conversion for wavelengths <2000A
    #fact[np.where(lam<2e3)] = 1.0
    for i in range(len(lam)):
        if lam[i] < 2000.0:
            fact[i] = 1.0
  
    #!Convert wavelength    
    return lam*fact


#---------------------------------------------------------------!
@jit(nopython=True, fastmath=True)
def vactoair(lam):
    """
    !convert wavelengths from air to vac
    !see Morton (1991 Ap.J. Suppl. 77, 119)
    this code was adapted from the IDL routine vactoair.pro
    """

    #nn = lam.size
    fact= 1. + 2.735182e-4 + 131.4182/lam**2 + 2.76249e8/lam**4
    #fact = 1 + (5.792105e-2/(238.0185-sigma2)) + (1.67918e-3/(57.362-sigma2))

    #!no conversion for wavelengths <2000A
    #fact[np.where(lam<2e3)] = 1.0
    for i in range(len(lam)):
        if lam[i] < 2000.0:
            fact[i] = 1.0
    #!Convert wavelengths
    
    return lam/fact
