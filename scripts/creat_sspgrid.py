import pickle
from alf_vars import ALFVAR
from setup import setup
from read_data import read_data

alfvar = ALFVAR()
alfvar.filename = 'input'
alfvar = read_data(alfvar)
alfvar.imf_type = 3
alfvar = setup(alfvar, onlybasic = True, ncpu=16)
pickle.dump(alfvar, open('../pickle/alfmodel.p', "wb" ) )
