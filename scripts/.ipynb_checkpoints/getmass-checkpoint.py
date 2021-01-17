import copy, numpy as np
from alf_vars import *

def getmass(mlo, mto, imf1, imf2, imfup, imf3=None, imf4=None, timfnorm = None):
    """
    !compute mass in stars and remnants (normalized to 1 Msun at t=0)
    !assume an IMF that runs from 0.08 to 100 Msun.
    !turnoff mass
    
    REAL(DP), INTENT(in) :: mlo,mto,imf1,imf2,imfup
    REAL(DP), INTENT(in), OPTIONAL :: imf3,imf4
    REAL(DP), INTENT(inout), OPTIONAL :: timfnorm
    INTEGER :: i
    REAL(DP) :: imfnorm, getmass
    REAL(DP), PARAMETER :: bhlim=40.0,nslim=8.5
    REAL(DP) :: m2=0.5,m3=1.0,alpha
    REAL(DP), DIMENSION(nimfnp) :: imfw=0.0, alpha2
    """

    
    #!---------------------------------------------------------------!
    #!---------------------------------------------------------------!
    m2=0.5; m3=1.0
    if mlo > m2:
        print('GETMASS ERROR: mlo>m2: %.2e, %.2e' %(mlo, m2))
        return getmass, np.nan
    
    alfvar = ALFVAR()
    imflo, imfhi = alfvar.imflo, alfvar.imfhi
    imf5 = alfvar.imf5
    mbin_nimf9 = np.copy(alfvar.mbin_nimf9)
    npi_alphav = np.copy(alfvar.npi_alphav)
    
    bhlim = 40.0; nslim = 8.5
    m2 = 0.5; m3 = 1.0
    
    nimfnp = 9
    imfw = np.zeros(nimfnp)
    alpha2 = np.empty(nimfnp)

    # ---------------------------------------------------------------!
    # ---------------------------------------------------------------!
        
    getmass = 0.0

    if imf4 is None:
        # ---- normalize the weights so that 1 Msun formed at t=0 ---- #
        imfnorm = (m2**(-imf1+2)-mlo**(-imf1+2))/(-imf1+2) + \
                   m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) + \
                   m2**(-imf1+imf2)*(imfhi**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)

        # ---- stars still alive ---- #
        getmass = (m2**(-imf1+2)-mlo**(-imf1+2))/(-imf1+2)
        
        if mto < m3:
            getmass += m2**(-imf1+imf2)*(mto**(-imf2+2)-m2**(-imf2+2))/(-imf2+2)
        else:
            getmass += (m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) + \
                        m2**(-imf1+imf2)*(mto**(-imfup+2)-m3**(-imfup+2))/(-imfup+2))
            
        getmass = getmass/imfnorm
    
        # ---- BH remnants ---- #
        # ---- 40<M<imf_up leave behind a 0.5*M BH ---- #
        getmass += 0.5*m2**(-imf1+imf2)*(imfhi**(-imfup+2)-bhlim**(-imfup+2))/(-imfup+2)/imfnorm

        # ---- NS remnants ---- #
        # ---- 8.5<M<40 leave behind 1.4 Msun NS ---- #
        getmass += 1.4*m2**(-imf1+imf2)*(bhlim**(-imfup+1)-nslim**(-imfup+1))/(-imfup+1)/imfnorm

        # ---- WD remnants ---- #
        # ---- M<8.5 leave behind 0.077*M+0.48 WD ---- #
        if mto < m3:
            getmass += 0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-m3**(-imfup+1))/(-imfup+1)/imfnorm
            
            getmass += 0.48*m2**(-imf1+imf2)*(m3**(-imf2+1)-mto**(-imf2+1))/(-imf2+1)/imfnorm
            
            getmass += 0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)/imfnorm
            
            getmass += 0.077*m2**(-imf1+imf2)*(m3**(-imf2+2)-mto**(-imf2+2))/(-imf2+2)/imfnorm
    
        else:
            getmass += 0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-mto**(-imfup+1))/(-imfup+1)/imfnorm
            
            getmass += 0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-mto**(-imfup+2))/(-imfup+2)/imfnorm


    else:
        #non-parametric IMF
        #mbin_nimf = (/0.2,0.4,0.6,0.8,1.0/)
        #mbin_nimf = np.array([0.2,0.4,0.6,0.8,1.0])
        alpha2 = 2.0 - alfvar.npi_alphav
        imfw[1-1] = 10**imf1 
        imfw[2-1] = 10**((imf2+imf1)/2.)
        imfw[3-1] = 10**imf2
        imfw[4-1] = 10**((imf3+imf2)/2.)
        imfw[5-1] = 10**imf3
        imfw[6-1] = 10**((imf4+imf3)/2.)
        imfw[7-1] = 10**imf4
        imfw[8-1] = 10**((imf5+imf4)/2.)
        imfw[9-1] = 10**imf5
        
        for i in range(nimfnp):
            imfw[i] *= alfvar.npi_renorm[i]
            
        imfnorm = 0.0
        for i in range(nimfnp):
            imfnorm += imfw[i]/alpha2[i]*(mbin_nimf9[i+1]**alpha2[i]-mbin_nimf9[i]**alpha2[i])

        imfnorm += imfw[8]/(-imfup+2)/(mbin_nimf9[9]**(-imfup)) * \
                    (imfhi**(-imfup+2)-mbin_nimf9[9]**(-imfup+2)) 
        
        if mto > mbin_nimf9[9]:
            # ---- MSTO > 1.0 ---- #
            for i in range(nimfnp):
                getmass += imfw[i]/alpha2[i]*(mbin_nimf9[i+1]**alpha2[i] - mbin_nimf9[i]**alpha2[i])
            
            getmass += imfw[8]/(-imfup+2)/(mbin_nimf9[9]**(-imfup))*(mto**(-imfup+2)-mbin_nimf9[9]**(-imfup+2)) 
                
            
            # ---- remnants from MTO-8.5 ---- #
            getmass += 0.48*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+1)-mto**(-imfup+1))/(-imfup+1)
            getmass += 0.077*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+2)-mto**(-imfup+2))/(-imfup+2)
            
            
        elif np.logical_and(mto > mbin_nimf9[8], mto <= mbin_nimf9[9]):
            
            # ---- 0.9 < MSTO < 1.0 ---- #
            for i in range(nimfnp-1):
                getmass += imfw[i]/alpha2[i]*(mbin_nimf9[i+1]**alpha2[i]-mbin_nimf9[i]**alpha2[i])
            getmass += imfw[8]/alpha2[i]*(mto**alpha2[i]-mbin_nimf9[8]**alpha2[i])
            
            # ---- WD remnants from 1.0-8.5 ---- #
            getmass += 0.48*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+1)-mbin_nimf9[9]**(-imfup+1))/(-imfup+1)
            getmass += 0.077*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+2)-mbin_nimf9[9]**(-imfup+2))/(-imfup+2)
                
            # ---- WD remnants from MSTO-1.0 ---- #
            getmass += 0.48*imfw[8]/(1-npi_alphav[i]) * (mbin_nimf9[9]**(1-npi_alphav[i])-mto**(1-npi_alphav[i]))
            getmass += 0.077*imfw[8]/alpha2[i]*(mbin_nimf9[9]**alpha2[i]-mto**alpha2[i])
        
                
        elif np.logical_and(mto>mbin_nimf9[7], mto<=mbin_nimf9[8]):
            
            # ---- 0.8 < MSTO < 0.9 ---- #
            for i in range(nimfnp-2):
                getmass += imfw[i]/alpha2[i]*(mbin_nimf9[i+1]**alpha2[i]-mbin_nimf9[i]**alpha2[i])
                
            getmass += imfw[7]/alpha2[i]*(mto**alpha2[i]-mbin_nimf9[7]**alpha2[i]) 
        
            # ---- WD remnants from 1.0-8.5 ---- #
            getmass += 0.48*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+1)-mbin_nimf9[9]**(-imfup+1))/(-imfup+1)
            getmass += 0.077*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+2)-mbin_nimf9[9]**(-imfup+2))/(-imfup+2)

            # ---- WD remnants from 0.9-1.0 ---- #
            getmass += (0.48*imfw[8]/(1-npi_alphav[i]) * 
                        (mbin_nimf9[9]**(1-npi_alphav[i])-mbin_nimf9[8]**(1-npi_alphav[i])))
            getmass += (0.077*imfw[8]/alpha2[i] * 
                        (mbin_nimf9[9]**alpha2[i]-mbin_nimf9[8]**alpha2[i]))
        
            # ---- WD remnants from MSTO-0.9 ---- #
            getmass += (0.48*imfw[7]/(1-npi_alphav[i]) * 
                        (mbin_nimf9[8]**(1-npi_alphav[i])-mto**(1-npi_alphav[i])))
            getmass += 0.077*imfw[7]/alpha2[i] * (mbin_nimf9[8]**alpha2[i]-mto**alpha2[i])
            
            
        elif np.logical_and(mto>mbin_nimf9[6], mto<=mbin_nimf9[7]):
            # ---- 0.7 < MSTO < 0.8 ---- #
            for i in range(nimfnp-3):
                getmass += imfw[i]/alpha2[i]*(mbin_nimf9[i+1]**alpha2[i] - 
                                              mbin_nimf9[i]**alpha2[i])
                
            getmass += imfw[6]/alpha2[i]*(mto**alpha2[i]-mbin_nimf9[6]**alpha2[i]) 
        
            # ---- WD remnants from 1.0-8.5 ---- #
            getmass += 0.48*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+1)-
                                                               mbin_nimf9[9]**(-imfup+1))/(-imfup+1)
            
            getmass += 0.077*imfw[8]/(mbin_nimf9[9]**(-imfup))*(nslim**(-imfup+2)-
                                                                mbin_nimf9[9]**(-imfup+2))/(-imfup+2)

            # ---- WD remnants from 0.9-1.0 ---- #
            getmass += 0.48*imfw[8]/(1-npi_alphav[i])*(mbin_nimf9[10-1]**(1-npi_alphav[i])-
                                                       mbin_nimf9[9-1]**(1-npi_alphav[i]))
            
            getmass += 0.077*imfw[8]/alpha2[i]*(mbin_nimf9[9]**alpha2[i]-
                                                mbin_nimf9[8]**alpha2[i])
 
            # ---- WD remnants from 0.8-0.9 ---- #
            getmass += 0.48*imfw[7]/(1-npi_alphav[i]) * (mbin_nimf9[8]**(1-npi_alphav[i])-
                                                         mbin_nimf9[7]**(1-npi_alphav[i]))
        
            getmass += 0.077*imfw[7]/alpha2[i] * (mbin_nimf9[8]**alpha2[i]-
                                                  mbin_nimf9[7]**alpha2[i])
        
            # ---- WD remnants from MSTO-0.8 ---- #
            getmass += 0.48*imfw[6]/(1-npi_alphav[i]) * (mbin_nimf9[7]**(1-npi_alphav[i])-
                                                         mto**(1-npi_alphav[i]))
            
            getmass += 0.077*imfw[6]/alpha2[i] * (mbin_nimf9[7]**alpha2[i]-
                                                  mto**alpha2[i])   
        else:
            print('GETMASS ERROR, msto<0.7',mto)
        # ---- BH remnants ---- #
        getmass += 0.5*10**imf5/(mbin_nimf9[9]**(-imfup))*(imfhi**(-imfup+2)-bhlim**(-imfup+2))/(-imfup+2)
        # ---- NS remnants ---- #
        getmass += 1.4*10**imf5/(mbin_nimf9[9]**(-imfup))*(bhlim**(-imfup+1)-nslim**(-imfup+1))/(-imfup+1)
        getmass = getmass / imfnorm

        
    if timfnorm is not None:
        timfnorm = imfnorm
        return getmass, imfnorm, timfnorm
        
    else:   
        return getmass, imfnorm


