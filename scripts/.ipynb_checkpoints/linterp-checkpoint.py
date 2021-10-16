import numpy as np
from numba import jit
__all__ = ['locate', 'linterp', 'tsum']


# ---------------------------------------------------------------- #
@jit(nopython=True)
def locate(xx, x):
    """
    - should be the same as locate.f90
    - note it's different from np.argmin(np.abs(xx-x))
        - for i in [5.3, 5.7, 4.9, 6.1]:
              locate(a, i)):  5,5,4,6
              np.argmin(np.abs(a-i)): 5,6,5,6
    """
    n = len(xx)
    if xx[-1] >= xx[0]:
        ascnd = True
    else:
        ascnd = False
    jl = -1
    ju = n
        
    if ascnd:
        while ju-jl>1:
            jm = int((ju+jl)/2)
            if x>= xx[jm]:
                jl=jm
            else:
                ju=jm
    else:
        while ju-jl>1:
            jm = int((ju+jl)/2)
            if x< xx[jm]:
                jl=jm
            else:
                ju=jm
                 
    if x==xx[0]:
        return 0
    elif x==xx[-1]:
        return n-2
    else:
        return jl
        

# ---------------------------------------------------------------- #
@jit(nopython=True)
def linterp(xin, yin, xout):
    return np.interp(xout, xin, yin)


# ---------------------------------------------------------------- #
@jit(nopython=True,fastmath=True)
def tsum(xin, yin):
    """
    !simple trapezoidal integration of tabulated function (xin,yin)
    """
    return np.nansum( (xin[1:]-xin[:-1]) * (yin[1:]+yin[:-1])/2.  )