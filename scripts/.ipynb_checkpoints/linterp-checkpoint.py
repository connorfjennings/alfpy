import numpy as np

__all__ = ['locate', 'linterp', 'tsum']


# ---------------------------------------------------------------- #
def locate_old(inarra, ina):
    """
    this is different from locate in alf !!!
    """
    return np.argmin(abs(inarra-ina))


# ---------------------------------------------------------------- #
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
def linterp_old(xin,yin,xout):
    """
    # routine to linearly interpolate a function yin(xin) at xout
    """
    xin, yin = np.asarray(xin), np.asarray(yin)
    n = xin.size
    n2 = xout.size
    yout = np.empty(n2) 
    
    for i in range(n2):
        klo = max(min(locate(xin,xout[i]), n-2), 0)
        yout[i] = yin[klo] + (yin[klo+1] - yin[klo])*(xout[i]-xin[klo])/(xin[klo+1]-xin[klo])
    return yout


# ---------------------------------------------------------------- #
def linterp_notuseful(xin,yin,xout):
    """
    # routine to linearly interpolate a function yin(xin) at xout
    """
    xin, yin = np.asarray(xin), np.asarray(yin)    
    def quickcal(ixout, xin=xin, yin=yin):
        k = max(min(np.argmin(abs(xin-ixout)), xin.size-2), 0)
        return yin[k] + (yin[k+1]-yin[k])*(ixout - xin[k])/(xin[k+1]-xin[k])
    
    return np.array(list(map(quickcal, xout)))


# ---------------------------------------------------------------- #
def linterp(xin, yin, xout):
    return np.interp(xout, xin, yin)


# ---------------------------------------------------------------- #
def tsum(xin, yin):
    """
    !simple trapezoidal integration of tabulated function (xin,yin)
    """
    return np.nansum( (xin[1:]-xin[:-1]) * (yin[1:]+yin[:-1])/2.  )