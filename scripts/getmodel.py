import pickle, copy, numpy as np

from alf_vars import *
from velbroad import *
from linterp import *
from add_response import add_response
from getmass import getmass

__all__ = ['getmodel']
        

# ---------------------------------------------------------------- #
def getmodel(pos, alfvar = None, mw = 0):
    """
    routine to produce a model spectrum (spec) for an input
    set of parameters (pos).  The optional flag 'mw' is used
    to force the IMF to be of the MW (Kroupa 2001) form
    """
    clight = 29979245800.0
    
    if alfvar is None:
        alfvar = pickle.load(open('alfvar_sspgrid_irldss3_imftype1.p', "rb" )) 
    sspgrid = copy.deepcopy(alfvar.sspgrid)
    
    msto_t0 = alfvar.msto_t0; msto_t1 = alfvar.msto_t1
    msto_z0 = alfvar.msto_z0; msto_z1 = alfvar.msto_z1; msto_z2=alfvar.msto_z2
    krpa_imf1, krpa_imf2, krpa_imf3 = alfvar.krpa_imf1, alfvar.krpa_imf2, alfvar.krpa_imf3
    nzmet = alfvar.nzmet
    nzmet3 = alfvar.nzmet3
    nimf = alfvar.nimf
    nage = alfvar.nage
    imfr1, imfr2,  imfr3 = alfvar.imfr1, alfvar.imfr2, alfvar.imfr3
        
    spec = np.empty(alfvar.nl) # output
    tmp, tmpr, yspec, tmp1, tmp2, tmp3, tmp4 = np.empty((7, alfvar.nl))
    tmp_ltrans, tmp_ftrans_h2o, tmp_ftrans_o2 = np.empty((3, alfvar.nl))
    emnormall = np.ones(alfvar.neml)
    imfw = np.zeros(alfvar.nimfnp)
    hermite = np.zeros(2)

    #---------------------------------------------------------------!
    #---------------------------------------------------------------!

    # set up interpolants for age, Line 25 in getmodel.f90
    vt = max(min(locate(sspgrid.logagegrid, pos.logage),alfvar.nage-2),0) 
    dt = (pos.logage - sspgrid.logagegrid[vt])/(sspgrid.logagegrid[vt+1]-sspgrid.logagegrid[vt])
    dt = max(min(dt, 1.2), -0.3)    # 0.5<age<14 Gyr

    #set up interpolants for metallicity
    vm = max(min(locate(sspgrid.logzgrid, pos.zh), nzmet-2), 0)
    dm = (pos.zh-sspgrid.logzgrid[vm])/(sspgrid.logzgrid[vm+1]-sspgrid.logzgrid[vm])
    dm = max(min(dm, 1.0),-1.0)    # -2.0<[Z/H]<0.25

    #compute the IMF-variable SSP
    if (alfvar.mwimf == 0) and (mw==0) and (alfvar.fit_type==0) and (alfvar.powell_fitting==0):
        #print("---- getting model for mwimf = ", alfvar.mwimf, ", fit_type=", alfvar.fit_type)
        vv1 = max(min(locate(sspgrid.imfx1, pos.imf1), nimf-2), 0)
        dx1 = (pos.imf1-sspgrid.imfx1[vv1])/(sspgrid.imfx1[vv1+1]-sspgrid.imfx1[vv1])
        dx1 = max(min(dx1, 1.0), 0.0)

        if alfvar.imf_type ==0 or alfvar.imf_type ==2:
            #print("getting model for imf_type=", alfvar.imf_type)
            #single power-law slope for IMF=0,2
            vv2 = vv1
            dx2 = dx1
        else:
            #print("getting model for imf_type=", alfvar.imf_type)
            #two-part power-law for IMF=1,3
            vv2 = max(min(locate(sspgrid.imfx2, pos.imf2), nimf-2),0)
            dx2 = (pos.imf2-sspgrid.imfx2[vv2])/(sspgrid.imfx2[vv2+1]-sspgrid.imfx2[vv2])
            dx2 = max(min(dx2, 1.0),0.0)

            
        if alfvar.imf_type == 2 or alfvar.imf_type == 3:
            #print("getting model for imf_type=", alfvar.imf_type)
            #variable low-mass cutoff for IMF=2,3, line 58 in getmodel.f90
            vv3 = max(min(locate(sspgrid.imfx3,pos.imf3), alfvar.nmcut-2),0)
            dx3 = (pos.imf3-sspgrid.imfx3[vv3])/(sspgrid.imfx3[vv3+1]-sspgrid.imfx3[vv3])
            dx3 = max(min(dx3, 1.0), 0.0)

            vm3 = max(min(locate(sspgrid.logzgrid2, pos.zh), nzmet3-2),0) 
            dm3 = (pos.zh-sspgrid.logzgrid2[vm3])/(sspgrid.logzgrid2[vm3+1]-sspgrid.logzgrid2[vm3])
            dm3 = max(min(dm3, 1.5), -1.0)
            

            tmp1 = ((1-dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2,vt+1,vv3,vm3+1] + 
                    (dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt+1,vv3,vm3+1] + 
                    (1-dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt+1,vv3,vm3+1] + 
                    (1-dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2,vt+1,vv3+1,vm3+1] + 
                    (dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt+1,vv3+1,vm3+1] + 
                    (1-dx1)*(dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt+1,vv3+1,vm3+1] + 
                    (dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2+1,vt+1,vv3,vm3+1] + 
                    dx1*dx2*dx3*sspgrid.logsspm[:,vv1+1,vv2+1,vt+1,vv3+1,vm3+1])

            tmp2 = ((1-dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2,vt,vv3,vm3+1] + 
                    (dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt,vv3,vm3+1] + 
                    (1-dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt,vv3,vm3+1] + 
                    (1-dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2,vt,vv3+1,vm3+1] + 
                    (dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt,vv3+1,vm3+1] + 
                    (1-dx1)*(dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt,vv3+1,vm3+1] + 
                    (dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2+1,vt,vv3,vm3+1] + 
                    dx1*dx2*dx3*sspgrid.logsspm[:,vv1+1,vv2+1,vt,vv3+1,vm3+1])

            tmp3 = ((1-dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2,vt+1,vv3,vm3] + 
                    (dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt+1,vv3,vm3] + 
                    (1-dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt+1,vv3,vm3] + 
                    (1-dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2,vt+1,vv3+1,vm3] + 
                    (dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt+1,vv3+1,vm3] + 
                    (1-dx1)*(dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt+1,vv3+1,vm3] + 
                    (dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2+1,vt+1,vv3,vm3] + 
                    dx1*dx2*dx3*sspgrid.logsspm[:,vv1+1,vv2+1,vt+1,vv3+1,vm3])

                
            tmp4 = ((1-dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2,vt,vv3,vm3] + \
                    (dx1)*(1-dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt,vv3,vm3] + \
                    (1-dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt,vv3,vm3] + \
                    (1-dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2,vt,vv3+1,vm3] + \
                    (dx1)*(1-dx2)*(dx3)*sspgrid.logsspm[:,vv1+1,vv2,vt,vv3+1,vm3] + \
                    (1-dx1)*(dx2)*(dx3)*sspgrid.logsspm[:,vv1,vv2+1,vt,vv3+1,vm3] + \
                    (dx1)*(dx2)*(1-dx3)*sspgrid.logsspm[:,vv1+1,vv2+1,vt,vv3,vm3] + \
                    dx1*dx2*dx3*sspgrid.logsspm[:,vv1+1,vv2+1,vt,vv3+1,vm3])

            spec = 10**(dt*dm3*tmp1 +(1.-dt)*dm3*tmp2 + dt*(1.-dm3)*tmp3 +(1.-dt)*(1.-dm3)*tmp4 )

            
        elif alfvar.imf_type==0 or alfvar.imf_type==1:
            #print("getting model for imf_type=", alfvar.imf_type)

            tmp1 = ((1.0-dx1)*(1.0-dx2)*sspgrid.logssp[:, vv1,vv2,vt+1,vm+1] + 
                    dx1*(1.0-dx2)*sspgrid.logssp[:, vv1+1,vv2,vt+1,vm+1] + 
                    (1.0-dx1)*dx2*sspgrid.logssp[:, vv1,vv2+1,vt+1,vm+1] + 
                    dx1*dx2*sspgrid.logssp[:, vv1+1,vv2+1,vt+1,vm+1])

            tmp2 = ((1.0-dx1)*(1.0-dx2)*sspgrid.logssp[:,vv1,vv2,vt,vm+1] + 
                    dx1*(1.0-dx2)*sspgrid.logssp[:,vv1+1,vv2,vt,vm+1] +
                    (1.0-dx1)*dx2*sspgrid.logssp[:,vv1,vv2+1,vt,vm+1] + 
                    dx1*dx2*sspgrid.logssp[:,vv1+1,vv2+1,vt,vm+1])

            tmp3 = ((1-dx1)*(1-dx2)*sspgrid.logssp[:,vv1,vv2,vt+1,vm] +  
                    dx1*(1-dx2)*sspgrid.logssp[:,vv1+1,vv2,vt+1,vm] + 
                    (1-dx1)*dx2*sspgrid.logssp[:,vv1,vv2+1,vt+1,vm] + 
                    dx1*dx2*sspgrid.logssp[:,vv1+1,vv2+1,vt+1,vm])

            tmp4 = ((1-dx1)*(1-dx2)*sspgrid.logssp[:,vv1,vv2,vt,vm] + 
                    dx1*(1-dx2)*sspgrid.logssp[:,vv1+1,vv2,vt,vm] + 
                    (1-dx1)*dx2*sspgrid.logssp[:,vv1,vv2+1,vt,vm] + 
                    dx1*dx2*sspgrid.logssp[:,vv1+1,vv2+1,vt,vm])

            spec = 10**(dt*dm*tmp1 + 
                        (1.0-dt)*dm*tmp2 + 
                        dt*(1.0-dm)*tmp3 + 
                        (1.0-dt)*(1.0-dm)*tmp4 )

            
        elif alfvar.imf_type == 4:
            #print("getting model for imf_type=", alfvar.imf_type)
            # non-parametric IMF, line 138 in getmodel.f90
            imfw[1-1] = 10**pos.imf1
            imfw[2-1] = 10**((pos.imf2+pos.imf1)/2.)
            imfw[3-1] = 10**pos.imf2
            imfw[4-1] = 10**((pos.imf3+pos.imf2)/2.)
            imfw[5-1] = 10**pos.imf3
            imfw[6-1] = 10**((pos.imf4+pos.imf3)/2.)
            imfw[7-1] = 10**pos.imf4
            imfw[8-1] = 10**((alfvar.imf5+pos.imf4)/2.)
            imfw[9-1] = 10**alfvar.imf5

            tmp1[:] = 0.0
            tmp2[:] = 0.0
            tmp3[:] = 0.0
            tmp4[:] = 0.0

            for i in range(alfvar.nimfnp):
                tmp1 += imfw[i]*sspgrid.sspnp[:,i, vt+1, vm+1]
                tmp2 += imfw[i]*sspgrid.sspnp[:,i, vt, vm+1]
                tmp3 += imfw[i]*sspgrid.sspnp[:,i, vt+1, vm]
                tmp4 += imfw[i]*sspgrid.sspnp[:,i, vt, vm]


            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt+1]) * 
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm+1]+
                            msto_z2*sspgrid.logzgrid[vm+1]**2),
                           3.0),0.75)
            mass, inorm = getmass(alfvar.imflo, msto, 
                                  pos.imf1, pos.imf2, 
                                  krpa_imf3,pos.imf3,pos.imf4)
            tmp1 = tmp1/inorm

            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt]) *
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm+1]+
                            msto_z2*sspgrid.logzgrid[vm+1]**2),
                           3.0),0.75)
            mass,inorm = getmass(alfvar.imflo, msto, 
                                 pos.imf1, pos.imf2, 
                                 krpa_imf3, pos.imf3, pos.imf4)
            tmp2 = tmp2/inorm

            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt+1]) * 
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm]+ 
                            msto_z2*sspgrid.logzgrid[vm]**2),
                           3.0),0.75)
            mass,inorm = getmass(alfvar.imflo, msto, pos.imf1, pos.imf2, 
                                 krpa_imf3, pos.imf3, pos.imf4)
            tmp3 = tmp3/inorm

            msto = max(min(10**(msto_t0+msto_t1*sspgrid.logagegrid[vt]) * 
                           (msto_z0+msto_z1*sspgrid.logzgrid[vm]+ 
                            msto_z2*sspgrid.logzgrid[vm]**2),
                           3.0),0.75)
            mass,inorm = getmass(alfvar.imflo, msto, pos.imf1, pos.imf2, 
                                 krpa_imf3, pos.imf3, pos.imf4)
            tmp4 = tmp4/inorm

            spec = 10**(dt*dm*np.log10(tmp1)+
                        (1-dt)*dm*np.log10(tmp2)+
                        dt*(1-dm)*np.log10(tmp3)+
                        (1-dt)*(1-dm)*np.log10(tmp4) )


    else:
        #compute a Kroupa IMF, line196
        print('get kroupa:', dt, dm, vt, vm)
        spec = 10**(dt*dm*sspgrid.logssp[:,imfr1,imfr2,vt+1,vm+1] + 
                    (1-dt)*dm*sspgrid.logssp[:,imfr1,imfr2,vt,vm+1] + 
                    dt*(1-dm)*sspgrid.logssp[:,imfr1,imfr2,vt+1,vm] + 
                    (1-dt)*(1-dm)*sspgrid.logssp[:,imfr1,imfr2,vt,vm] )
        return spec


    #vary young population - both fraction and age
    #only include these parameters in the "full" model
    if (alfvar.fit_type==0) and (alfvar.powell_fitting == 0) and (alfvar.fit_two_ages ==1):
        #print('vary young population')
        fy = max(min(10**pos.logfy, 1.0), 0.0)
        vy = max(min(locate(sspgrid.logagegrid, pos.fy_logage), nage-2),0)
        dy = (pos.fy_logage-sspgrid.logagegrid[vy])/(sspgrid.logagegrid[vy+1]-sspgrid.logagegrid[vy])
        dy = max(min(dy, 1.0), -0.3)    #!0.5<age<13.5 Gyr
        yspec = (dy*dm*sspgrid.logssp[:,imfr1, imfr2, vy+1, vm+1] + 
                 (1-dy)*dm*sspgrid.logssp[:, imfr1, imfr2, vy, vm+1] + 
                 dy*(1-dm)*sspgrid.logssp[:, imfr1, imfr2, vy+1, vm] + 
                 (1-dy)*(1-dm)*sspgrid.logssp[:, imfr1, imfr2, vy, vm])
        spec = (1-fy)*spec + fy*10**yspec

    #vary age in the response functions
    if alfvar.use_age_dep_resp_fcns == 0:
        #force the use of the response fcn at age=fix_age_dep_resp_fcns
        vr = max(min(locate(sspgrid.logagegrid_rfcn, 
                            np.log10(alfvar.fix_age_dep_resp_fcns)), alfvar.nage_rfcn-2),0)
        dr = (np.log10(alfvar.fix_age_dep_resp_fcns)-sspgrid.logagegrid_rfcn[vr])/(sspgrid.logagegrid_rfcn[vr+1]-sspgrid.logagegrid_rfcn[vr])
        dr = max(min(dr, 1.0), 0.0)

    else:
        #should be using mass-weighted age here
        vr = max(min(locate(sspgrid.logagegrid_rfcn,pos.logage),alfvar.nage_rfcn-2),0)
        dr = (pos.logage-sspgrid.logagegrid_rfcn[vr])/(sspgrid.logagegrid_rfcn[vr+1]-sspgrid.logagegrid_rfcn[vr])
        dr = max(min(dr, 1.0), 0.0)

        
    #vary metallicity in the response functions, line 221
    if alfvar.use_z_dep_resp_fcns == 0:
        vm2 = max(min(locate(sspgrid.logzgrid,fix_z_dep_resp_fcns),nzmet-2),0)
        dm2 = (alfvar.fix_z_dep_resp_fcns-sspgrid.logzgrid[vm2])/(sspgrid.logzgrid[vm2+1]-sspgrid.logzgrid[vm2])
        dm2 = max(min(dm2, 1.0), 0.0)
    else:
        vm2 = vm
        dm2 = dm


    #Only sigma, velz, logage, and [Z/H] are fit when either
    #fitting in Powell mode or "super simple" mode
    # line 250
    if (alfvar.powell_fitting ==0) and (alfvar.fit_type != 2):
        #vary [Fe/H]
        spec = add_response(spec, pos.feh, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.fep,sspgrid.fem)
        #vary [O/H]
        spec = add_response(spec, pos.ah, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.ap)
        #vary [C/H]
        spec = add_response(spec, pos.ch, 0.15,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.cp,sspgrid.cm)
        #vary [N/H]
        spec = add_response(spec, pos.nh, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.np,sspgrid.nm)
        #vary [Mg/H]
        spec = add_response(spec, pos.mgh, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.mgp,sspgrid.mgm)
        #vary [Si/H]
        spec = add_response(spec, pos.sih, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.sip,sspgrid.sim)
        #vary [Ca/H]
        spec = add_response(spec, pos.cah, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.cap,sspgrid.cam)
        #vary [Ti/H]
        spec = add_response(spec, pos.tih, 0.3,dr,vr,dm2,vm2, 
                            sspgrid.solar,sspgrid.tip,sspgrid.tim)
        
        #vary [Na/H] (special case)
        if pos.nah < 0.3:
            spec = add_response(spec, pos.nah, 0.3, dr,vr,dm2,vm2, 
                                sspgrid.solar,sspgrid.nap,sspgrid.nam)

            
        elif pos.nah >= 0.3 and pos.nah < 0.6:

            tmpr = (dr*dm2*sspgrid.nap[:,vr+1,vm2+1]/sspgrid.solar[:,vr+1,vm2+1] + 
                    (1-dr)*dm2*sspgrid.nap[:,vr,vm2+1]/sspgrid.solar[:,vr,vm2+1] + 
                    dr*(1-dm2)*sspgrid.nap[:,vr+1,vm2]/sspgrid.solar[:,vr+1,vm2] + 
                    (1-dr)*(1-dm2)*sspgrid.nap[:,vr,vm2]/sspgrid.solar[:,vr,vm2])

            tmp = (dr*dm2*(sspgrid.nap6[:,vr+1,vm2+1]-sspgrid.nap[:,vr+1,vm2+1])/sspgrid.solar[:,vr+1,vm2+1]+
                   (1-dr)*dm2*(sspgrid.nap6[:,vr,vm2+1]-sspgrid.nap[:,vr,vm2+1])/sspgrid.solar[:,vr,vm2+1]+
                   dr*(1-dm2)*(sspgrid.nap6[:,vr+1,vm2]-sspgrid.nap[:,vr+1,vm2])/sspgrid.solar[:,vr+1,vm2]+
                   (1-dr)*(1-dm2)*(sspgrid.nap6[:,vr,vm2]-sspgrid.nap[:,vr,vm2])/sspgrid.solar[:,vr,vm2])

            spec = spec * (tmpr+tmp*(pos.nah-0.3)/0.3 )

                   
        elif pos.nah >= 0.6:

            tmpr = (dr*dm2*sspgrid.nap6[:,vr+1,vm2+1]/sspgrid.solar[:,vr+1,vm2+1] + 
                    (1-dr)*dm2*sspgrid.nap6[:,vr,vm2+1]/sspgrid.solar[:,vr,vm2+1] + 
                    dr*(1-dm2)*sspgrid.nap6[:,vr+1,vm2]/sspgrid.solar[:,vr+1,vm2] + 
                    (1-dr)*(1-dm2)*sspgrid.nap6[:,vr,vm2]/sspgrid.solar[:,vr,vm2])

            tmp = (dr*dm2*(sspgrid.nap9[:,vr+1,vm2+1]-sspgrid.nap6[:,vr+1,vm2+1])/sspgrid.solar[:,vr+1,vm2+1]+
                   (1-dr)*dm2*(sspgrid.nap9[:,vr,vm2+1]-sspgrid.nap6[:,vr,vm2+1])/sspgrid.solar[:,vr,vm2+1]+
                   dr*(1-dm2)*(sspgrid.nap9[:,vr+1,vm2]-sspgrid.nap6[:,vr+1,vm2])/sspgrid.solar[:,vr+1,vm2]+
                   (1-dr)*(1-dm2)*(sspgrid.nap9[:,vr,vm2]-sspgrid.nap6[:,vr,vm2])/sspgrid.solar[:,vr,vm2])

            spec = spec * (tmpr+tmp*(pos.nah-0.6)/0.6 )



    #only include these parameters in the "full" model, line325
    if alfvar.fit_type==0 and alfvar.powell_fitting==0:
                   
        #vary Teff (special case - force use of the 13 Gyr model)
        spec = add_response(spec, pos.teff, 50.,
                            1.0, alfvar.nage_rfcn-2, dm2, vm2,
                            sspgrid.solar, sspgrid.teffp, sspgrid.teffm)
                   
        #add a hot star (interpolate in hot_teff and [Z/H]
        vh   = max(min(locate(sspgrid.teffarrhot, pos.hotteff), alfvar.nhot-2), 0)
        dh   = (pos.hotteff-sspgrid.teffarrhot[vh])/(sspgrid.teffarrhot[vh+1]-sspgrid.teffarrhot[vh])
        tmp  = (dh*dm*sspgrid.hotspec[:,vh+1,vm+1] + 
                (1-dh)*dm*sspgrid.hotspec[:,vh,vm+1] + 
                dh*(1-dm)*sspgrid.hotspec[:,vh+1,vm] + 
                (1-dh)*(1-dm)*sspgrid.hotspec[:,vh,vm])

        fy   = max(min(10**pos.loghot, 1.0), 0.0)
        spec = (1-fy)*spec + fy*tmp

        #add in an M7 giant
        fy   = max(min(10**pos.logm7g, 1.0), 0.0)
        spec = (1.0-fy)*spec + fy*sspgrid.m7g

        #vary [K/H]
        spec = add_response(spec,pos.kh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.kp)
        #vary [V/H]
        spec = add_response(spec,pos.vh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.vp)
        #vary [Cr/H]
        spec = add_response(spec,pos.crh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.crp)
        #vary [Mn/H]
        spec = add_response(spec,pos.mnh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.mnp)
        #vary [Co/H]
        spec = add_response(spec,pos.coh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.cop)
        #vary [Ni/H]
        spec = add_response(spec,pos.nih,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.nip)
        #vary [Cu/H]
        spec = add_response(spec,pos.cuh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.cup)
        #vary [Sr/H]
        spec = add_response(spec,pos.srh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.srp)
        #vary [Ba/H]
        spec = add_response(spec,pos.bah,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.bap,sspgrid.bam)
        #vary [Eu/H]
        spec = add_response(spec,pos.euh,0.3, dr,vr,dm2,vm2,sspgrid.solar,sspgrid.eup)

                
        #add emission lines
        if alfvar.maskem==0:
            #these line ratios come from Nell Byler's Cloudy lookup table
            emnormall[1-1] = 10**pos.logemline_h / 11.21   # Hy
            emnormall[2-1]  = 10**pos.logemline_h / 6.16    # Hd
            emnormall[3-1]  = 10**pos.logemline_h / 2.87    # Hb
            emnormall[4-1]  = 10**pos.logemline_oiii / 3.0  # [OIII]
            emnormall[5-1]  = 10**pos.logemline_oiii        # [OIII]
            emnormall[6-1]  = 10**pos.logemline_ni          # [NI]
            emnormall[7-1]  = 10**pos.logemline_nii / 2.95  # [NII]
            emnormall[8-1]  = 10**pos.logemline_h           # Ha
            emnormall[9-1]  = 10**pos.logemline_nii         # [NII]
            emnormall[10-1] = 10**pos.logemline_sii         # [SII]
            emnormall[11-1] = 10**pos.logemline_sii * 0.77  # [SII]
            emnormall[12-1] = 10**pos.logemline_oii         # [OII]
            emnormall[13-1] = 10**pos.logemline_oii * 1.35  # [OII]
            emnormall[14-1] = 10**pos.logemline_h / 65.0    # H12 (ratio is made up)
            emnormall[15-1] = 10**pos.logemline_h / 55.0    # H11 (ratio is made up)
            emnormall[16-1] = 10**pos.logemline_h / 45.0    # H10
            emnormall[17-1] = 10**pos.logemline_h / 35.0    # H9
            emnormall[18-1] = 10**pos.logemline_h / 25.0    # H8
            emnormall[19-1] = 10**pos.logemline_h / 18.0    # H7

            for i in range(alfvar.neml):
                #allow the em lines to be offset in velocity from the continuum
                #NB: velz2 is a *relative* shift between continuum and lines
                ve   = alfvar.emlines[i]/(1+pos.velz2/clight*1e5)
                lsig = max(ve*pos.sigma2/clight*1e5, 1.0)  #min dlam=1.0A
                spec += emnormall[i] * np.exp(-(sspgrid.lam-ve)**2/lsig**2/2.0)


    #velocity broaden the model
    if pos.sigma > 5. and alfvar.fit_indices==0:
        if alfvar.fit_hermite == 1:
            hermite[0] = pos.h3
            hermite[1] = pos.h4
            spec = velbroad(sspgrid.lam, spec, pos.sigma, alfvar.l1[0], alfvar.l2[alfvar.nlint-1], hermite)
        else:
            spec = velbroad(sspgrid.lam, spec, pos.sigma, alfvar.l1[0], alfvar.l2[alfvar.nlint-1])


    #apply an atmospheric transmission function only in full mode
    #note that this is done *after* velocity broadening
    if alfvar.fit_type== 0 and alfvar.powell_fitting==0 and alfvar.fit_trans==1:
        #applied in the observed frame
        tmp_ltrans     = sspgrid.lam / (1+pos.velz/clight*1e5)
        tmp_ftrans_h2o = linterp(tmp_ltrans, sspgrid.atm_trans_h2o, sspgrid.lam)
        tmp_ftrans_o2  = linterp(tmp_ltrans, sspgrid.atm_trans_o2, sspgrid.lam)
        spec = spec * (1.+(tmp_ftrans_h2o-1)*10**pos.logtrans)
        spec = spec * (1.+(tmp_ftrans_o2-1)*10**pos.logtrans)

    #apply a template error function
    if alfvar.apply_temperrfcn==1:
        spec = spec / alfvar.temperrfcn

                             
    return spec                    
                             
                             
                             
                             
                             
