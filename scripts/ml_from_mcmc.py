import numpy as np
from alf_vars import ALFVAR,ALFPARAM
from str2arr import str2arr
from linterp import linterp, tsum
from getmodel import getmodel
from getmass import getmass
from alf_constants import ALF_HOME,mypi,clight,lsun,pc2cm

def ml_from_mcmc(filename, alfvar=None):
    """
    #! takes an *mcmc file as input and returns M/L in many filters
    #! note that M/L is returned in the observed frame
    """
    #!----------------------------------------------------------------------!
    #!----------------------------------------------------------------------!
  
    nfil2 = 27
    #mspec, lam, zmspe = np.zeros((3, nl))
    d1=0.; oneplusz=0.; mass=0.; msto=0.


    pos = ALFPARAM()
    if alfvar is None:
        alfvar = ALFVAR()
        
    npar = alfvar.npar
    nstart, nl = alfvar.nstart, alfvar.nl
    msto_t0 = alfvar.msto_t0; msto_t1 = alfvar.msto_t1
    msto_z0 = alfvar.msto_z0; msto_z1 = alfvar.msto_z1; msto_z2=alfvar.msto_z2
    krpa_imf1, krpa_imf2, krpa_imf3 = alfvar.krpa_imf1, alfvar.krpa_imf2, alfvar.krpa_imf3
    imflo,imfhi = alfvar.imflo,alfvar.imfhi
        
        
    posarr = np.zeros(alfvar.npar)
    filters2 = np.zeros((alfvar.nl, nfil2))
    m2l, mag, magsun = np.zeros((3, nfil2))
    mlx2 = np.zeros(6)
    #!----------------------------------------------------------------------!
    #!----------------------------------------------------------------------!
    nlint = 1
    l1, l2 = alfvar.l1, alfvar.l2
    l1[0] = 0.0
    l2[nlint-1] = 1.5e4
    

    try:
        f11 = np.loadtxt("{0}results/{1}.sum".format(ALF_HOME, filename), )
    except:
        print('ERROR, file not found: ', filename)
      
    
    #read in the header to set the relevant parameters
    char='#'
    with open("{0}results/{1}.sum".format(ALF_HOME, filename), "r") as myfile:
        temdata = myfile.readlines()
    for iline in temdata:
        if iline.split(' ')[0] == char:
            temline = np.array(iline.split()[1:])
            #print(temline)
            if 'mwimf' in temline:
                mwimf = int(temline[-1].split('\n')[0])
            if 'imf_type' in temline:
                imf_type = int(temline[-1].split('\n')[0])
            if 'fit_type' in temline:
                fit_type = int(temline[-1].split('\n')[0])
            if 'fit_two_ages' in temline:
                fit_two_ages = int(temline[-1].split('\n')[0]) 
            if 'fit_hermite' in temline:
                fit_hermite = int(temline[-1].split('\n')[0])
            if 'nonpimf_alpha' in temline:
                nonpimf_alpha = int(temline[-1].split('\n')[0])
            if 'Nwalkers' in temline:
                nwalkers = int(temline[-1].split('\n')[0])
            if 'Nchain' in temline:
                nchain = int(temline[-1].split('\n')[0])
    print('mwimf=', mwimf, ', imf_type=', imf_type, ', nwalkers=', nwalkers, ', nchain=', nchain)
                

    #setup the models
    #CALL SETUP()
    lam = np.loadtxt(ALF_HOME+'models/sspgrid.lam.dat')

    #read in filter transmission curves (they are already normalized)
    f15 = np.loadtxt('{0}infiles/filters2_py.dat'.format(ALF_HOME))
    #READ(15,*) magsun2
    filter2 = np.copy(f15[nstart:nstart+nl, 1:])

    with open('{0}infiles/filters2_py.dat'.format(ALF_HOME), "r") as myfile:
        temdata = myfile.readlines()
    for iline in temdata:
        if iline.split(' ')[0] == '#':
            tem = iline.split()[1:]
            magsun2 = np.array([float(i) for i in tem])
    
    try:
        f11 = np.loadtxt("{0}results/{1}.mcmc".format(ALF_HOME, filename), )
    except:
        print('ERROR, file not found: ', filename, '.mcmc')    
    
    
    
    #!----------------------------------------------------------------------!    
    #!open file for output
    f12 = open("{0}models/{1}test.ml_all".format(ALF_HOME, filename), "a")
    f12.writelines('# UBVRI, SDSS ugriz, ACS F435W F475W F555W F606W F625W F775W '+
       'F814W F850LP, UVIS F336W F390W F438W F475W F555W F606W F775W F814W F850LP')
        
    #!----------------------------------------------------------------------!
    #!----------------------------------------------------------------------!
    d1_arr, posarr_arr, mlx2_arr = f11[:,0], f11[:,1:47], f11[:,-6:]    
    ii = 0
    for j in range(nchain):
        for k in range(nwalkers):
            #!read the parameters from the mcmc file
            posarr = posarr_arr[ii]
            if (j+1*k+1) == nchain*nwalkers/4:
                print('---- 25\% done')
            elif (j+1*k+1) == nchain*nwalkers/2:
                print('---- 50\% done')
            elif (j+1*k+1) == nchain*nwalkers/4*3:
                print('---- 75\% done')
            ii+=1
            #copy the input parameter array into the structure
            pos = str2arr(switch=2, inarr=posarr) #arr->str

            #turn off the emission lines
            pos.logemline_h    = -8.0
            pos.logemline_oii  = -8.0
            pos.logemline_oiii = -8.0
            pos.logemline_nii  = -8.0
            pos.logemline_sii  = -8.0
            pos.logemline_ni   = -8.0

            #get the model spectrum
            mspec = getmodel(pos, alfvar)
        
            #redshift the spectrum
            oneplusz = (1+pos.velz/clight*1E5)
            zmspec   = linterp(lam*oneplusz, mspec, lam)
            zmspec[zmspec<0]=0.
            #convert to proper units for mags
            zmspec  = zmspec*lsun/1e6*lam**2/clight/1e8/4/mypi/pc2cm**2

            #main sequence turn-off mass
            msto = 10**(msto_t0 + msto_t1 * pos.logage) * ( msto_z0 + msto_z1*pos.zh + msto_z2*pos.zh**2 )

            #compute normalized stellar mass
            if (imf_type == 0):
                mass, imfnorm = getmass(imflo,msto,pos.imf1,pos.imf1, krpa_imf3)
            elif (imf_type == 1):
                mass, imfnorm = getmass(imflo, msto, pos.imf1, pos.imf2, krpa_imf3)
            elif (imf_type == 2):
                mass, imfnorm = getmass(pos.imf3,msto,pos.imf1,pos.imf1, krpa_imf3)
            elif (imf_type == 3):
                mass, imfnorm = getmass(pos.imf3,msto,pos.imf1,pos.imf2, krpa_imf3)
            elif (imf_type == 4):
                mass, imfnorm = getmass(imflo,msto,pos.imf1,pos.imf2,krpa_imf3,pos.imf3,pos.imf4)
                
                
            #loop over the filters
            for i in range(nfil2):
                mag[i] = tsum(lam, zmspec*filters2[:,i]/lam)
                if (mag[i] <= 0.0):
                    m2l[i] = 0.0
                else:
                    mag[i] = -2.5*np.log10(mag[i])-48.60
                    m2l[i] = mass/10**(2./5*(magsun2[i]-mag[i]))
                    if (m2l[i] > 100.):
                         m2l[i]=0.0
                            
            f12.writelines((' ').join([str(i) for i in m2l]))
            f12.writelines('\n')
    #write M/L values to file
    f12.close()

