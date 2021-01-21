import os, numpy as np
from alf_vars import *
from astropy.io import ascii as astro_ascii
from linterp import *
from alf_constants import *


__all__ = ['read_data']


def read_data(alfvar, sigma=None, velz=None):
    """
    # routine to read in the data that will be used in the fit
    # returns a structure for the data and an integer specifying
    # the length of the data array
    - update alfvar.data
    - updates:
        - include the sky term, since it's only called in alf.f90? 
    """

    filename = alfvar.filename
    lam = np.copy(alfvar.sspgrid.lam)
    
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
            spec = astro_ascii.read("{0}indata/{1}.dat".format(ALF_HOME, filename))
        except:
            print('READ_DATA ERROR: file not found')
            print("{0}indata/{1}.dat".format(ALF_HOME, filename))


    # ---- Read in the wavelength boundaries, which are in the header-----!
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
            
    #spec = astro_ascii.read('{0}indata/{1}.dat'.format(ALF_HOME, filename))
    header = np.asarray(header)
    nlint = header.shape[0]
    if nlint == 0:
        header = [np.array(0.40, 0.47), np.array(0.47, 0.55)]
        nlint = 2
       
    # --- convert from um to A.
    header *= 1e4
    alfvar.l1 = np.asarray(np.copy(header[:,0]))
    alfvar.l2 = np.asarray(np.copy(header[:,1]))

    #--------now read in the input spectrum, errors, and weights----------!
    alfvar.nlint = nlint
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
    
    
    # ---- the following part is moved from alf.py ---- #
    # ---- interpolate the sky emission model onto the observed wavelength grid
    if alfvar.observed_frame == 1:
        alfvar.data.sky = linterp(alfvar.lsky, alfvar.fsky, alfvar.data.lam)
        alfvar.data.sky[alfvar.data.sky<0] = 0.
    else:
        alfvar.data.sky[:] = tiny_number
    alfvar.data.sky[:] = tiny_number  # ?? check why?

    # ---- we only compute things up to 500A beyond the input fit region
    alfvar.nl_fit = min(max(locate(lam, alfvar.l2[-1]+500.0),0),alfvar.nl-1)

    #define the log wavelength grid used in velbroad.f90
    alfvar.dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-
                     np.log(alfvar.sspgrid.lam[0]))/alfvar.nl_fit

    for i in range(alfvar.nl_fit):
        alfvar.lnlam[i] = i*alfvar.dlstep + np.log(alfvar.sspgrid.lam[0])

    # ---- masked regions have wgt=0.0.  We'll use wgt as a pseudo-error
    # ---- array in contnormspec, so turn these into large numbers
    alfvar.data.wgt = 1./(alfvar.data.wgt+tiny_number)
    alfvar.data.wgt[alfvar.data.wgt>huge_number] = huge_number
    # ---- fold the masked regions into the errors
    alfvar.data.err = alfvar.data.err * alfvar.data.wgt
    alfvar.data.err[alfvar.data.err>huge_number] = huge_number
            
    if np.logical_and(sigma is not None, velz is not None):
        return alfvar, isig, ivelz
    
    else:
        return alfvar
