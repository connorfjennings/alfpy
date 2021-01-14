import numpy as np
from alf_vars import *
from linterp import *
from getmass import getmass

__all__ = ['getm2l']

def getm2l(lam, spec, pos, mw = 0, alfvar=None):
    """
    !compute mass-to-light ratio in several filters (AB mags)
    -- INPUTS: lam, spec, pos
    -- OUTPUTS: m2l
    """
    if alfvar == None:
        alfvar = ALFVAR()
    nfil = alfvar.nfil
    nl = alfvar.nl
    
    lsun = alfvar.lsun
    clight = alfvar.clight
    mypi = alfvar.mypi
    pc2cm = alfvar.pc2cm
    nstart = alfvar.nstart
    msto_t0 = alfvar.msto_t0; msto_t1 = alfvar.msto_t1
    msto_z0 = alfvar.msto_z0; msto_z1 = alfvar.msto_z1; msto_z2=alfvar.msto_z2
    krpa_imf1, krpa_imf2, krpa_imf3 = alfvar.krpa_imf1, alfvar.krpa_imf2, alfvar.krpa_imf3
    imflo, imfhi = alfvar.imflo, alfvar.imfhi
    
    #f15 = np.loadtxt('{0}infiles/filters.dat'.format(ALF_HOME))
    #alfvar.filters[:,0] = np.copy(f15[alfvar.nstart-1:alfvar.nend, 1])
    #alfvar.filters[:,1] = np.copy(f15[alfvar.nstart-1:alfvar.nend, 2])
    #alfvar.filters[:,2] = np.copy(f15[alfvar.nstart-1:alfvar.nend, 3])
        
    m2l = np.zeros(nfil)
    mag = np.zeros(nfil)
    #aspec = np.empty(nl)

    #!---------------------------------------------------------------!
    #!---------------------------------------------------------------!
    #!convert to the proper units
    aspec = spec*lsun/1e6*lam**2/clight/1e8/4/mypi/pc2cm**2
    msto = 10**(msto_t0 + msto_t1*pos.logage)*(msto_z0 + msto_z1*pos.zh + msto_z2*pos.zh**2 )
    
    if mw == 1:
        mass, imfnorm = getmass(imflo, msto, 
                                krpa_imf1, krpa_imf2, krpa_imf3)
    else:
        if alfvar.imf_type ==0:
            #!single power-law IMF with a fixed lower-mass cutoff
            mass, imfnorm = getmass(imflo, msto, 
                                    pos.imf1, pos.imf1, krpa_imf3)
            
        elif alfvar.imf_type ==1: 
            #!double power-law IMF with a fixed lower-mass cutoff
            mass, imfnorm = getmass(imflo, msto, 
                                    pos.imf1, pos.imf2, krpa_imf3)
            
        elif alfvar.imf_type ==2:
            #!single powerlaw index with variable low-mass cutoff
            mass, imfnorm = getmass(pos.imf3, msto, 
                                    pos.imf1, pos.imf1, krpa_imf3)
            
        elif alfvar.imf_type == 3:
            #!double powerlaw index with variable low-mass cutoff
            mass, imfnorm = getmass(pos.imf3, msto, 
                                    pos.imf1, pos.imf2, krpa_imf3)
            
            
        elif alfvar.imf_type == 4:
            #non-parametric IMF for 0.08-1.0 Msun; Salpeter slope at >1 Msun
            mass, imfnorm = getmass(imflo, msto, 
                                    pos.imf1, pos.imf2, krpa_imf3, 
                                    pos.imf3, pos.imf4)

            
    #loop over the filters
    filters = np.copy(alfvar.filters)
    for j in range(nfil):
        jmag = tsum(lam, aspec * filters[:,j]/lam)
        if jmag <= 0:
            m2l[j] = 0.
        else:
            jmag = -2.5*np.log10(jmag)-48.60
            m2l[j] = mass/10**(2./5*(alfvar.magsun[j]-jmag))
            if m2l[j] > 100:
                m2l[j] = 0.
        mag[j] = jmag

    return m2l
