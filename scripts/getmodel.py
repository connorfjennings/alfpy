import math, numpy as np
from velbroad import *
from linterp import locate, linterp
from add_response import add_response, add_na_03
from getmass import getmass
from alf_constants import *
#from str2arr import *
from numba import jit
__all__ = ['getmodel']
        

# ---------------------------------------------------------------- #    
@jit(nopython=True, fastmath=True)
def fast_np_power(x1, x2):
    # ---- only slightly faster ---- #
    return x1**x2

# ---------------------------------------------------------------- #    
@jit(nopython=True, fastmath=True)
def get_dv(ingrid, inval, maxv, minv, maxind, minind):
    inind = max(min(locate(ingrid, inval),maxind),minind)  
    return inind, max(min((inval - ingrid[inind])/(ingrid[inind+1]-ingrid[inind]), maxv), minv)


# ---------------------------------------------------------------- #    
@jit(nopython=True, fastmath=True)
def cal_logsspm(inarr, dx1, dx2, dx3, dt, dm3, vv1, vv2, vv3, vm3, vt):
        tmp1 = ((1-dx1)*(1-dx2)*(1-dx3)*inarr[:,vv1,vv2,vt+1,vv3,vm3+1] + 
                   dx1*(1-dx2)*(1-dx3)*inarr[:,vv1+1,vv2,vt+1,vv3,vm3+1] + 
                   (1-dx1)*dx2*(1-dx3)*inarr[:,vv1,vv2+1,vt+1,vv3,vm3+1] + 
                   (1-dx1)*(1-dx2)*dx3*inarr[:,vv1,vv2,vt+1,vv3+1,vm3+1] + 
                   dx1*(1-dx2)*(dx3)*inarr[:,vv1+1,vv2,vt+1,vv3+1,vm3+1] + 
                   (1-dx1)*dx2*dx3*inarr[:,vv1,vv2+1,vt+1,vv3+1,vm3+1] + 
                   (dx1)*(dx2)*(1-dx3)*inarr[:,vv1+1,vv2+1,vt+1,vv3,vm3+1] + 
                   dx1*dx2*dx3*inarr[:,vv1+1,vv2+1,vt+1,vv3+1,vm3+1])

        tmp2 = ((1-dx1)*(1-dx2)*(1-dx3)*inarr[:,vv1,vv2,vt,vv3,vm3+1] + 
                   dx1*(1-dx2)*(1-dx3)*inarr[:,vv1+1,vv2,vt,vv3,vm3+1] + 
                   (1-dx1)*dx2*(1-dx3)*inarr[:,vv1,vv2+1,vt,vv3,vm3+1] + 
                   (1-dx1)*(1-dx2)*dx3*inarr[:,vv1,vv2,vt,vv3+1,vm3+1] + 
                   dx1*(1-dx2)*dx3*inarr[:,vv1+1,vv2,vt,vv3+1,vm3+1] + 
                   (1-dx1)*(dx2)*dx3*inarr[:,vv1,vv2+1,vt,vv3+1,vm3+1] + 
                   dx1*dx2*(1-dx3)*inarr[:,vv1+1,vv2+1,vt,vv3,vm3+1] + 
                   dx1*dx2*dx3*inarr[:,vv1+1,vv2+1,vt,vv3+1,vm3+1])

        tmp3 = ((1-dx1)*(1-dx2)*(1-dx3)*inarr[:,vv1,vv2,vt+1,vv3,vm3] + 
                   dx1*(1-dx2)*(1-dx3)*inarr[:,vv1+1,vv2,vt+1,vv3,vm3] + 
                   (1-dx1)*dx2*(1-dx3)*inarr[:,vv1,vv2+1,vt+1,vv3,vm3] + 
                   (1-dx1)*(1-dx2)*dx3*inarr[:,vv1,vv2,vt+1,vv3+1,vm3] + 
                   dx1*(1-dx2)*dx3*inarr[:,vv1+1,vv2,vt+1,vv3+1,vm3] + 
                   (1-dx1)*dx2*dx3*inarr[:,vv1,vv2+1,vt+1,vv3+1,vm3] + 
                   dx1*dx2*(1-dx3)*inarr[:,vv1+1,vv2+1,vt+1,vv3,vm3] + 
                   dx1*dx2*dx3*inarr[:,vv1+1,vv2+1,vt+1,vv3+1,vm3])

                
        tmp4 = ((1-dx1)*(1-dx2)*(1-dx3)*inarr[:,vv1,vv2,vt,vv3,vm3] + 
                   dx1*(1-dx2)*(1-dx3)*inarr[:,vv1+1,vv2,vt,vv3,vm3] + 
                   (1-dx1)*dx2*(1-dx3)*inarr[:,vv1,vv2+1,vt,vv3,vm3] + 
                   (1-dx1)*(1-dx2)*dx3*inarr[:,vv1,vv2,vt,vv3+1,vm3] + 
                   dx1*(1-dx2)*dx3*inarr[:,vv1+1,vv2,vt,vv3+1,vm3] + 
                   (1-dx1)*dx2*dx3*inarr[:,vv1,vv2+1,vt,vv3+1,vm3] + 
                   dx1*dx2*(1-dx3)*inarr[:,vv1+1,vv2+1,vt,vv3,vm3] + 
                   dx1*dx2*dx3*inarr[:,vv1+1,vv2+1,vt,vv3+1,vm3])
        
        return fast_np_power(10, dt*dm3*tmp1 +(1.-dt)*dm3*tmp2 + dt*(1.-dm3)*tmp3 +(1.-dt)*(1.-dm3)*tmp4 )

    
# ---------------------------------------------------------------- #  
@jit(nopython=True, fastmath=True)
def cal_logssp(inarr, dx1, dx2, dt, dm, vv1, vv2, vm, vt):    
        tmp1 = ((1.0-dx1)*(1.0-dx2)*inarr[:, vv1,vv2,vt+1,vm+1] + 
                   dx1*(1.0-dx2)*inarr[:, vv1+1,vv2,vt+1,vm+1] + 
                   (1.0-dx1)*dx2*inarr[:, vv1,vv2+1,vt+1,vm+1] + 
                   dx1*dx2*inarr[:, vv1+1,vv2+1,vt+1,vm+1])

        tmp2 = ((1.0-dx1)*(1.0-dx2)*inarr[:,vv1,vv2,vt,vm+1] + 
                   dx1*(1.0-dx2)*inarr[:,vv1+1,vv2,vt,vm+1] +
                   (1.0-dx1)*dx2*inarr[:,vv1,vv2+1,vt,vm+1] + 
                   dx1*dx2*inarr[:,vv1+1,vv2+1,vt,vm+1])

        tmp3 = ((1-dx1)*(1-dx2)*inarr[:,vv1,vv2,vt+1,vm] +  
                   dx1*(1-dx2)*inarr[:,vv1+1,vv2,vt+1,vm] + 
                   (1-dx1)*dx2*inarr[:,vv1,vv2+1,vt+1,vm] + 
                   dx1*dx2*inarr[:,vv1+1,vv2+1,vt+1,vm])

        tmp4 = ((1-dx1)*(1-dx2)*inarr[:,vv1,vv2,vt,vm] + 
                   dx1*(1-dx2)*inarr[:,vv1+1,vv2,vt,vm] + 
                   (1-dx1)*dx2*inarr[:,vv1,vv2+1,vt,vm] + 
                   dx1*dx2*inarr[:,vv1+1,vv2+1,vt,vm])
        
        return fast_np_power(10,dt*dm*tmp1 + (1.-dt)*dm*tmp2 + dt*(1.-dm)*tmp3 +(1.-dt)*(1.-dm)*tmp4 )
    

    
# ---------------------------------------------------------------- #
@jit(nopython=True, fastmath=True)
def tmp_add_em(i_norm, i_em, p_velz2, p_sigma2, i_lam):
    ve   = i_em/(1+p_velz2/clight*1e5)
    lsig = max(ve*p_sigma2/clight*1e5, 1.0)  #min dlam=1.0A
    return i_norm * np.exp(-(i_lam-ve)**2/lsig**2/2.0)    


# ---------------------------------------------------------------- #
@jit(nopython=True, fastmath=True)
def tmp_fy(dh, dm, vh, vm, ingrid, inspec, inarr_m7g, in_loghot, in_logm7g):
    #!add a hot star (interpolate in hot_teff and [Z/H]
    tmp = (dh*dm*ingrid[:,vh+1,vm+1] + (1-dh)*dm*ingrid[:,vh,vm+1] + dh*(1-dm)*ingrid[:,vh+1,vm] + (1-dh)*(1-dm)*ingrid[:,vh,vm])  

    fy   = max(min(10**in_loghot, 1.0), 0.0)
    inspec = inspec + fy*tmp

    # ---- add in an M7 giant
    fy   = max(min(10**in_logm7g, 1.0), 0.0)
    inspec = (1.0-fy)*inspec + fy*inarr_m7g
    return inspec
# ---------------------------------------------------------------- #

def getmodel(pos, alfvar, mw = 0):
    """
    routine to produce a model spectrum (spec) for an input
    set of parameters (pos).  The optional flag 'mw' is used
    to force the IMF to be of the MW (Kroupa 2001) form
    """
    
    sspgrid = alfvar.sspgrid    
    msto_t0 = alfvar.msto_t0; msto_t1 = alfvar.msto_t1
    msto_z0 = alfvar.msto_z0; msto_z1 = alfvar.msto_z1; msto_z2=alfvar.msto_z2
    krpa_imf1, krpa_imf2, krpa_imf3 = alfvar.krpa_imf1, alfvar.krpa_imf2, alfvar.krpa_imf3
    nzmet = alfvar.nzmet
    nzmet3 = alfvar.nzmet3
    nimf = alfvar.nimf
    nage = alfvar.nage
    imfr1, imfr2,  imfr3 = alfvar.imfr1, alfvar.imfr2, alfvar.imfr3
        
    #emnormall = np.ones(alfvar.neml)
    #imfw = np.zeros(alfvar.nimfnp)
    hermite = np.zeros(2)

    #---------------------------------------------------------------!
    #---------------------------------------------------------------!
    # ---- set up interpolants for age, Line 25 in getmodel.f90
    vt, dt = get_dv(sspgrid.logagegrid, pos.logage, 1.2, -0.3, alfvar.nage-2, 0)

    # ---- set up interpolants for metallicity
    vm, dm = get_dv(sspgrid.logzgrid, pos.zh, 1.0, -1.0, nzmet-2, 0)

    # ---- compute the IMF-variable SSP
    if (alfvar.mwimf == 0) and (mw==0) and (alfvar.fit_type==0) and (alfvar.powell_fitting==0):
        vv1, dx1 = get_dv(sspgrid.imfx1, pos.imf1, 1.0, 0.0, nimf-2, 0)

        if alfvar.imf_type in [0, 2]:
            # ---- single power-law slope for IMF=0,2
            vv2 = vv1
            dx2 = dx1
        else:
            # ---- two-part power-law for IMF=1,3
            vv2, dx2 = get_dv(sspgrid.imfx2, pos.imf2, 1.0, 0.0, nimf-2, 0)
  
        if alfvar.imf_type in [2, 3]:
            vv3, dx3 = get_dv(sspgrid.imfx3, pos.imf3, 1.0, 0.0, alfvar.nmcut-2, 0)           

        if alfvar.imf_type in [2, 3]:
            vm3, dm3 = get_dv(sspgrid.logzgrid2, pos.zh, 1.5, -1.0, nzmet3-2,0)
            spec = cal_logsspm(sspgrid.logsspm, dx1, dx2, dx3, dt, dm3, vv1, vv2, vv3, vm3, vt)
            
        elif alfvar.imf_type in [0, 1]:
            spec = cal_logssp(sspgrid.logssp, dx1, dx2, dt, dm, vv1, vv2, vm, vt)
            
        elif alfvar.imf_type == 4:
            imfw = 10**np.array([pos.imf1, (pos.imf2+pos.imf1)/2., pos.imf2, (pos.imf3+pos.imf2)/2., 
                                 pos.imf3, (pos.imf4+pos.imf3)/2., pos.imf4, (alfvar.imf5+pos.imf4)/2., 
                                 alfvar.imf5])
            imfw[1-1] = 10**pos.imf1
            imfw[2-1] = 10**((pos.imf2+pos.imf1)/2.)
            imfw[3-1] = 10**pos.imf2
            imfw[4-1] = 10**((pos.imf3+pos.imf2)/2.)
            imfw[5-1] = 10**pos.imf3
            imfw[6-1] = 10**((pos.imf4+pos.imf3)/2.)
            imfw[7-1] = 10**pos.imf4
            imfw[8-1] = 10**((alfvar.imf5+pos.imf4)/2.)
            imfw[9-1] = 10**alfvar.imf5


            lam_length = len(sspgrid.lam)
            tmp1, tmp2, tmp3, tmp4 = np.zeros((4, lam_length))
            for i in range(alfvar.nimfnp):
                tmp1 += imfw[i]*sspgrid.sspnp[:,i, vt+1, vm+1]
                tmp2 += imfw[i]*sspgrid.sspnp[:,i, vt, vm+1]
                tmp3 += imfw[i]*sspgrid.sspnp[:,i, vt+1, vm]
                tmp4 += imfw[i]*sspgrid.sspnp[:,i, vt, vm]

            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt+1]) * 
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm+1]+
                            msto_z2*sspgrid.logzgrid[vm+1]**2),3.0),0.75)
            _, inorm = getmass(alfvar.imflo, msto, pos.imf1, pos.imf2, 
                                  krpa_imf3,pos.imf3,pos.imf4)
            tmp1 = tmp1/inorm

            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt]) *
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm+1]+
                            msto_z2*sspgrid.logzgrid[vm+1]**2),3.0),0.75)
            _, inorm = getmass(alfvar.imflo, msto, 
                                 pos.imf1, pos.imf2, 
                                 krpa_imf3, pos.imf3, pos.imf4)
            tmp2 = tmp2/inorm

            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt+1]) * 
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm]+ 
                            msto_z2*sspgrid.logzgrid[vm]**2),3.0),0.75)
            _, inorm = getmass(alfvar.imflo, msto, pos.imf1, pos.imf2, 
                                 krpa_imf3, pos.imf3, pos.imf4)
            tmp3 = tmp3/inorm

            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt]) * 
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm]+ 
                            msto_z2*sspgrid.logzgrid[vm]**2),3.0),0.75)
            _, inorm = getmass(alfvar.imflo, msto, pos.imf1, pos.imf2, 
                                 krpa_imf3, pos.imf3, pos.imf4)
            tmp4 = tmp4/inorm

            spec = 10**(dt*dm*np.log10(tmp1)+
                        (1-dt)*dm*np.log10(tmp2)+
                        dt*(1-dm)*np.log10(tmp3)+
                        (1-dt)*(1-dm)*np.log10(tmp4) )


    else:
        # ---- compute a Kroupa IMF, line196
        spec = fast_np_power(10, dt*dm*sspgrid.logssp[:,imfr1,imfr2,vt+1,vm+1] + 
                        (1-dt)*dm*sspgrid.logssp[:,imfr1,imfr2,vt,vm+1] + 
                        dt*(1-dm)*sspgrid.logssp[:,imfr1,imfr2,vt+1,vm] + 
                        (1-dt)*(1-dm)*sspgrid.logssp[:,imfr1,imfr2,vt,vm] )


    # ---- vary young population - both fraction and age
    # ---- only include these parameters in the "full" model
    if (alfvar.fit_type==0) and (alfvar.powell_fitting == 0) and (alfvar.fit_two_ages ==1):
        fy = max(min(10**pos.logfy, 1.0), 0.0)
        vy, dy = get_dv(sspgrid.logagegrid, pos.fy_logage, 1.0, -0.3, alfvar.nage-2, 0)
        
        yspec = (dy*dm*sspgrid.logssp[:,imfr1, imfr2, vy+1, vm+1] + 
                 (1-dy)*dm*sspgrid.logssp[:, imfr1, imfr2, vy, vm+1] + 
                 dy*(1-dm)*sspgrid.logssp[:, imfr1, imfr2, vy+1, vm] + 
                 (1-dy)*(1-dm)*sspgrid.logssp[:, imfr1, imfr2, vy, vm])
        spec = (1-fy)*spec + fy*10**yspec

    # ---- vary age in the response functions
    if alfvar.use_age_dep_resp_fcns == 0:
        # ---- force the use of the response fcn at age=fix_age_dep_resp_fcns
        vr, dr = get_dv(sspgrid.logagegrid_rfcn, 
                        math.log10(alfvar.fix_age_dep_resp_fcns), 
                        1.0, 0.0, alfvar.nage_rfcn-2,0)
    else:
        # ---- should be using mass-weighted age here
        vr, dr = get_dv(sspgrid.logagegrid_rfcn, 
                        pos.logage, 1.0, 0.0, 
                        alfvar.nage_rfcn-2, 0)

        
    # ---- vary metallicity in the response functions, line 221
    if alfvar.use_z_dep_resp_fcns == 0:
        vm2, dm2 = get_dv(sspgrid.logzgrid, 
                          alfvar.fix_z_dep_resp_fcns, 
                          1.0, 0.0, nzmet-2, 0)
    else:
        vm2 = vm
        dm2 = dm


    # ---- Only sigma, velz, logage, and [Z/H] are fit when either
    # ---- fitting in Powell mode or "super simple" mode
    # line 250
    if (alfvar.powell_fitting ==0) and (alfvar.fit_type != 2):
        #vary [Fe/H]
        spec *= add_response(pos.feh, 0.3, dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.fep,sspgrid.fem)
        #vary [O/H]
        spec *= add_response(pos.ah, 0.3, dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.ap)
        #vary [C/H]
        spec *= add_response(pos.ch, 0.15,dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.cp, sspgrid.cm)
        #vary [N/H]
        spec *= add_response(pos.nh, 0.3, dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.np, sspgrid.nm)
        #vary [Mg/H]
        spec *= add_response(pos.mgh, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.mgp, sspgrid.mgm)
        #vary [Si/H]
        spec *= add_response(pos.sih, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.sip, sspgrid.sim)
        #vary [Ca/H]
        spec *= add_response(pos.cah, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.cap, sspgrid.cam)
        #vary [Ti/H]
        spec *= add_response(pos.tih, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar, sspgrid.tip, sspgrid.tim)
        #vary [Na/H] (special case)
        if pos.nah < 0.3:
            spec *= add_response(pos.nah, 0.3, dr,vr,dm2,vm2, 
                                sspgrid.solar,sspgrid.nap,sspgrid.nam)
            
            
        elif pos.nah >= 0.3 and pos.nah < 0.6:
            spec *= add_na_03(pos.nah, 0.3, dr,vr,dm2,vm2, 
                              sspgrid.solar, sspgrid.nap, sspgrid.nap6)

                   
        elif pos.nah >= 0.6:
            spec *= add_na_03(pos.nah, 0.6, dr,vr,dm2,vm2, 
                              sspgrid.solar, sspgrid.nap6, sspgrid.nap9)


    # ---- only include these parameters in the "full" model, line325
    if alfvar.fit_type==0 and alfvar.powell_fitting==0:
                   
        # ---- vary Teff (special case - force use of the 13 Gyr model)
        spec *= add_response(pos.teff, 50.,1.0, alfvar.nage_rfcn-2, dm2, vm2,
                            sspgrid.solar, sspgrid.teffp, sspgrid.teffm)
         
        # ---- add a hot star (interpolate in hot_teff and [Z/H]
        #vh   = max(min(locate(sspgrid.teffarrhot, pos.hotteff), alfvar.nhot-2), 0)
        #dh   = (pos.hotteff-sspgrid.teffarrhot[vh])/(sspgrid.teffarrhot[vh+1]-sspgrid.teffarrhot[vh])
        vh, dh = get_dv(sspgrid.teffarrhot, pos.hotteff, np.nan, np.nan, alfvar.nhot-2, 0)
        
        spec = tmp_fy(dh, dm, vh, vm, sspgrid.hotspec, 
                      spec, sspgrid.m7g, pos.loghot, pos.logm7g)
        
        #tmp  = (dh*dm*sspgrid.hotspec[:,vh+1,vm+1] + 
        #       (1-dh)*dm*sspgrid.hotspec[:,vh,vm+1] + 
        #       dh*(1-dm)*sspgrid.hotspec[:,vh+1,vm] + 
        #       (1-dh)*(1-dm)*sspgrid.hotspec[:,vh,vm])
        #tmp = tmp_hotstar(dh, dm, vh, vm, sspgrid.hotspec)

        #fy   = max(min(10**pos.loghot, 1.0), 0.0)
        #spec = spec + fy*tmp

        ## ---- add in an M7 giant
        #fy   = max(min(10**pos.logm7g, 1.0), 0.0)
        #spec = (1.0-fy)*spec + fy*sspgrid.m7g

        # ---- vary [K/H] ---- #
        spec *= add_response(pos.kh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.kp)
        # ---- vary [V/H]
        spec *= add_response(pos.vh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.vp)
        # ---- vary [Cr/H]
        spec *= add_response(pos.crh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.crp)
        # ---- vary [Mn/H]
        spec *= add_response(pos.mnh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.mnp)
        # ---- vary [Co/H]
        spec *= add_response(pos.coh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.cop)
        # ---- vary [Ni/H]
        spec *= add_response(pos.nih,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.nip)
        # ---- vary [Cu/H]
        spec *= add_response(pos.cuh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.cup)
        # ---- vary [Sr/H]
        spec *= add_response(pos.srh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.srp)
        # ---- vary [Ba/H]
        spec *= add_response(pos.bah,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.bap,sspgrid.bam)
        # ---- vary [Eu/H]
        spec *= add_response(pos.euh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.eup)

        
        #add emission lines
        if alfvar.maskem==0:
            # ---- these line ratios come from Nell Byler's Cloudy lookup table
            emnormall = 10**np.array([pos.logemline_h, pos.logemline_h, pos.logemline_h, 
                                      pos.logemline_oiii, pos.logemline_oiii, pos.logemline_ni, 
                                      pos.logemline_nii, pos.logemline_h, pos.logemline_nii, 
                                      pos.logemline_sii, pos.logemline_sii, pos.logemline_oii, pos.logemline_oii, 
                                      pos.logemline_h, pos.logemline_h, pos.logemline_h, 
                                      pos.logemline_h, pos.logemline_h, pos.logemline_h])
            emnormall *= np.array([1./11.21, 1./6.16, 1./2.87, 1./3., 
                                   1., 1., 1./2.95, 1., 1., 1., 0.77, 
                                   1., 1.35, 1./65, 1./55, 1./45, 1./35, 
                                   1./25, 1./18])

            for i in range(alfvar.neml):
                # ---- allow the em lines to be offset in velocity from the continuum
                # ---- NB: velz2 is a *relative* shift between continuum and lines
                #ve   = alfvar.emlines[i]/(1+pos.velz2/clight*1e5)
                #lsig = max(ve*pos.sigma2/clight*1e5, 1.0)  #min dlam=1.0A
                #spec += emnormall[i] * np.exp(-(sspgrid.lam-ve)**2/lsig**2/2.0)
                spec += tmp_add_em(emnormall[i], alfvar.emlines[i], pos.velz2, pos.sigma2, sspgrid.lam)

    
    # velocity broaden the model   
    if pos.sigma > 5. and alfvar.fit_indices==0:
        if alfvar.fit_hermite == 1:
            hermite[0] = pos.h3
            hermite[1] = pos.h4
            spec = velbroad(sspgrid.lam, spec, pos.sigma, 
                            alfvar.l1[0]-100, alfvar.l2[alfvar.nlint-1]+100, 
                            hermite, velbroad_simple=1)
            
        else:
            spec = velbroad(sspgrid.lam, spec, pos.sigma, 
                            alfvar.l1[0]-100, alfvar.l2[alfvar.nlint-1]+100, 
                            velbroad_simple = 0)


    # ---- apply an atmospheric transmission function only in full mode
    # ---- note that this is done *after* velocity broadening
    if alfvar.fit_type== 0 and alfvar.powell_fitting==0 and alfvar.fit_trans==1:
        #applied in the observed frame
        tmp_ltrans     = sspgrid.lam / (1+pos.velz/clight*1e5)
        tmp_ftrans_h2o = linterp(xin=tmp_ltrans, yin=sspgrid.atm_trans_h2o, xout=sspgrid.lam)
        tmp_ftrans_o2  = linterp(xin=tmp_ltrans, yin=sspgrid.atm_trans_o2, xout=sspgrid.lam)
        spec = spec*(1+(tmp_ftrans_h2o-1)*fast_np_power(10, pos.logtrans))
        spec = spec*(1+(tmp_ftrans_o2-1)*fast_np_power(10, pos.logtrans))

    # ---- apply a template error function
    if alfvar.apply_temperrfcn==1:
        spec = spec / alfvar.temperrfcn
                             
    return spec                    
                             
                             
                             
                             
                             
