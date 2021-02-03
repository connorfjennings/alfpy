import matplotlib.pyplot as plt
from astropy.io import ascii as astro_ascii
import pickle, scipy, copy, numpy as np, pandas as pd
from dynesty import plotting as dyplot
import sys
alfpydir = '../scripts/'
sys.path.insert(1, alfpydir)

from spec_from_sum import *
from linterp import locate
from getmodel import *
import math
from func import func
from str2arr import *
from str2arr import alfobj
from contnormspec import *
from alf_constants import *
from linterp import *
from read_data import *
from getm2l import *
from post_process import calm2l_dynesty, worker_m2l

alfvar = pickle.load(open('../pickle/alfvar_sspgrid_ldss3_dr247_n1600_Re4_wave6e_1.p', "rb" ))
pos, _, mspec_ = spec_from_sum('ldss3_dr247_n1600_Re4_wave6e_imf3hernoatm', 
                               alfvar, getsum = 'minchi2', returnspec=True)
alfvar.filename = 'ldss3_dr247_n1600_Re4_wave6e'
alfvar = read_data(alfvar)

from post_process import calm2l_mcmc
infile = '{0}/{1}'.format('/Users/menggu/work/massive/alfresults', 
                          'ldss3_dr247_n1600_Re4_wave6e_imf3hernoatm.mcmc')
calm2l_mcmc(infile, alfvar, ncpu=8, outname = 'test3')