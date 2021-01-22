from alf_vars import *

def update_runtot(runtot, inarr, m2l, m2lmw):
    """
    !routine to update the array that holds the running totals
    !of the parameters, parameters^2, etc. in order to compute
    !averages and errors.
    """

    alfvar = ALFVAR()
    npar = alfvar.npar
    nfil = alfvar.nfil
    #runtot = np.empty((3, alfvar.npar +2*alfvar.nfil))

    runtot += 1
    runtot[1, :npar] += inarr
    runtot[2, :npar] += inarr**2

    runtot[2-1,npar:npar+nfil] += m2l
    runtot[3-1,npar:npar+nfil] += m2l**2

    runtot[1,npar+nfil:npar+2*nfil] += m2lmw
    runtot[2,npar+nfil:npar+2*nfil] += +m2lmw**2
    
    return runtot