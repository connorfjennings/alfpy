import os, numpy as np
from itertools import takewhile
import re

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
    with open('{0}indata/{1}.dat'.format(ALF_HOME, filename), 'r') as fobj:
        headiter = takewhile(lambda s: s.startswith('#'), fobj)
        header = list(headiter)
        
    header = np.array([list(filter(None, re.split('#|\s+|\n', i))) for i in header])
    header = header.astype(float)
    nlint = header.shape[0]
    if nlint == 0:
        header = [np.array(0.40, 0.47), np.array(0.47, 0.55)]
        nlint = 2
       
    # --- convert from um to A.
    header *= 1e4
    alfvar.l1 = np.asarray(np.copy(header[:,0]))
    alfvar.l2 = np.asarray(np.copy(header[:,1]))
    print('l1, l2', alfvar.l1, alfvar.l2)

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
    alfvar.data.sky = np.zeros_like(alfvar.data.lam)

    
    if np.logical_and(sigma is not None, velz is not None):
        return alfvar, isig, ivelz
    
    else:
        return alfvar
