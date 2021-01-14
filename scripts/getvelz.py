from alf_vars import *
from linterp import *
from contnormspec import *
from velbroad import *
import copy, numpy as np


def getvelz(alfvar):
    """
    ! Function to estimate the recession velocity.
    ! Uses the first two wavelength segments, unless only one exists
    ! This routine is a bit clunky, and often fails at low S/N
    """
    
    max_dvelz   = 5e3
    delchi2_tol = 0.5   #0.5, 0.2
    max_zred    = 0.18  #0.18, 0.03
    nv = 5000
    data = alfvar.data
    sspgrid = alfvar.sspgrid
    nl = alfvar.nl

    
    #!------------------------------------------------------!
    mflx, dflx = np.zeros((2, nl))
    tvz, tchi2, tvza = np.zeros((3, nv))
    iidata = ALFTDATA(ndat = nl) 
    chi2  = alfvar.huge_number
    tchi2[:] = alfvar.huge_number
    getvelz = 0.0

    #!use ni=2 wavelength segments unless only 1 segment exists
    if alfvar.nlint >=3:
        ni = 2
    else:
        ni = 1
        
    for i in range(nv):
        tvz[i] = (i+1.0)/nv*(max_zred*3e5 + 1e3)-1e3
        #!de-redshift the data and interpolate to model wave array
        #!NB: this is the old way of doing things, compare with func.f90
        
        data.lam0 = data.lam/(1.+tvz[i]/alfvar.clight*1e5)
        iidata.flx = linterp(data.lam0, data.flx, sspgrid.lam)
        iidata.err = linterp(data.lam0, data.err, sspgrid.lam)

        #!only use the first ni wavelength segments
        for ij in range(1):
            j=0
            lo = max(alfvar.l1[j], data.lam0[0]) + 50
            hi = min(alfvar.l2[j], data.lam0[-1]) - 50
            #!dont use the near-IR in redshift fitting
            if (lo > 9000.):
                continue
            if lo >= hi:
                if j==0:
                    print('GETVELZ ERROR:, lo>hi & j=1')
                    print(l1[j], l2[j])
                    print(lo, hi)
                    print(data.lam0[0], data.lam0[-1])
                tchi2[i] = alfvar.huge_number
                continue

            #!NB: this is the old way of doing things, compare with func.f90
            dflx = contnormspec(sspgrid.lam, iidata.flx, iidata.err, lo, hi)
            
            #!use a 5 Gyr Zsol SSP
            mflx = contnormspec(sspgrid.lam, 10**sspgrid.logssp[:, 3, 8, 2, 3],
                                np.sqrt(10**sspgrid.logssp[:, 3, 8, 2, 3]), 
                                lo,hi)

            i1 = min(max(locate(sspgrid.lam, lo),0),nl-2) 
            i2 = min(max(locate(sspgrid.lam, hi),1),nl-1) 

            #!only count pixels with non-zero weights
            #ng = 0
            #for k in range(i1, i2):
            #    if iidata.err[k] < alfvar.huge_number/2.:
            #        ng += 1
            ng = np.sum(iidata.err[i1:i2] < alfvar.huge_number/2)
                    
            tchi2[i] = np.nansum(iidata.flx[i1:i2]**2/iidata.err[i1:i2]**2*(dflx[i1:i2]-mflx[i1:i2])**2)/ng

            if tchi2[i] < chi2:
                chi2 = tchi2[i]
                getvelz = tvz[i]


    # test to see if the solution is good
    # we take all points with delta(chi2/dof)<delchi2_tol and
    # ask how large is the range in velocities.
    # If the range in velz is >max_dvelz then we've failed
    print('initial solution: ', getvelz)
    print('at minchi2:', tvz[np.where(tchi2 == np.nanmin(tchi2))])
    tchi2[:] = tchi2[:] - np.nanmin(tchi2)
    tvza[:]  = getvelz
    tvza[np.where(tchi2 < delchi2_tol)] = tvz[np.where(tchi2 < delchi2_tol)]
            
    if np.nanmax(tvza) - np.nanmin(tvza) > max_dvelz:
        print("   Failed to find a redshift solution, setting velz=0.0")
        print("delta(velz|chi2<0.5) = %.2f" %(np.nanmax(tvza) - np.nanmin(tvza)))
        getvelz = 0.0

    return getvelz