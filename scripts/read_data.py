import os, numpy as np
from alf_vars import *
from astropy.io import ascii as astro_ascii


__all__ = ['read_data']

def read_data(alfvar, sigma=None, velz=None):
    """
    #routine to read in the data that will be used in the fit
    #returns a structure for the data and an integer specifying
    #the length of the data array
    """

    filename = alfvar.filename
    
    if alfvar.fit_indices == 1:
        try:
            f10 = astro_ascii.read("{0}indata/{1}.indx".format(ALF_HOME, filename))
        except:
            print('READ_DATA ERROR: file not found')
            print("{0}indata/{1}.indx".format(ALF_HOME, filename))

        alfvar.indx2fit = f10['col1']
        ivelz = f10['col2'] 
        isig = f10['col3']


    else:
        try:
            f10 = astro_ascii.read("{0}indata/{1}.dat".format(ALF_HOME, filename))
        except:
            print('READ_DATA ERROR: file not found')
            print("{0}indata/{1}.dat".format(ALF_HOME, filename))


    #-----Read in the wavelength boundaries, which are in the header-----!
    char='#'
    with open ('{0}indata/{1}.dat'.format(ALF_HOME, filename), "r") as myfile:
        temdata=myfile.readlines()

    header = []
    for iline in temdata:
        if iline.split(' ')[0] == char:
            temline = np.array(iline.split(' ')[1:])
            header_item = []
            for iitem in temline:
                if len(iitem.split('\n'))>1:
                    header_item.append(float(iitem.split('\n')[0]))
                else:
                    header_item.append(float(iitem))
            header.append(np.array(header_item))
            
    spec = astro_ascii.read('{0}indata/{1}.dat'.format(ALF_HOME, filename))

    if len(header) == 0:
        header = [np.array(0.40, 0.47), np.array(0.47, 0.55)]
       
    #convert from um to A.
    header = np.asarray(header)
    header *= 1e4
    alfvar.l1 = np.asarray(np.copy(header[:,0]))
    alfvar.l2 = np.asarray(np.copy(header[:,1]))

    #--------now read in the input spectrum, errors, and weights----------!
    alfvar.nlint = header.shape[0]
    alfvar.datmax = spec['col1'].size
    if (alfvar.nlint > alfvar.nlint_max):
        print('READ_DATA ERROR: number of wavelength\n'+
             'intervals exceeds nlint_max')

    alfvar.data = ALFTDATA(alfvar.ndat)
    alfvar.data.lam = np.copy(spec['col1'].data)
    alfvar.data.flx = np.copy(spec['col2'].data)
    alfvar.data.err = np.copy(spec['col3'].data)
    alfvar.data.wgt = np.copy(spec['col4'].data)
    alfvar.data.ires = np.copy(spec['col5'].data)


    #read in the tabulated M/L prior
    if alfvar.extmlpr == 1:
        try:
            f11 = astro_ascii.read("{0}indata/{1}.mlpr".format(ALF_HOME, filename))
            alfvar.mlprtab = np.copy(f11)
            alfvar.nmlprtabmax = len(f11['col1'])
        except:
            print('READ_DATA ERROR: tabulated M/L prior file not found')
            print("{0}indata/{1}.mlpr".format(ALF_HOME, filename))
        

      #ensure that the prior is normalized to unity
      #mlprtab(1:nmlprtabmax,2) = mlprtab(1:nmlprtabmax,2) / MAXVAL(mlprtab(1:nmlprtabmax,2))
      #set a non-zero floor so that initial walker locations are not out of bounds
      #mlprtab(1:nmlprtabmax,2) = MAX(mlprtab(1:nmlprtabmax,2),1E-6)
        
    if np.logical_and(sigma is not None, velz is not None):
        return alfvar, isig, ivelz
    
    else:
        return alfvar
