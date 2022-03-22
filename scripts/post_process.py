import os, math, numpy as np, pandas as pd
from functools import partial
from dynesty import utils as dyfunc
from getm2l import *
from getmodel import *
from str2arr import *
from tofit_parameters import tofit_params

import time
import h5py

import multiprocessing
#from multiprocessing import Pool
from schwimmbad import MultiPool
#from mpi4py.futures import MPIPoolExecutor
from joblib import Parallel, delayed
from tqdm import tqdm


__all__ = ['calm2l_dynesty']
ALFPY_HOME = os.environ['ALFPY_HOME']

key_list = [ikey for ikey in tofit_params.keys() if tofit_params[ikey].fit is True]

# ---------------------------------------------------------------- #
def worker_m2l(alfvar, use_keys, inarr):
    tem_posarr = fill_param(inarr, usekeys = use_keys)
    tem_pos = str2arr(2, inarr = tem_posarr)
    # ---- turn off various parameters for computing M/L
    tem_pos.logemline_h    = -8.0
    tem_pos.logemline_oii  = -8.0
    tem_pos.logemline_oiii = -8.0
    tem_pos.logemline_nii  = -8.0
    tem_pos.logemline_sii  = -8.0
    tem_pos.logemline_ni   = -8.0
    tem_pos.logtrans       = -8.0
    tem_pos.sigma          = 4.0
    
    tem_mspec = getmodel(tem_pos, alfvar=alfvar)
    tem_mspec_mw = getmodel(tem_pos, alfvar=alfvar, mw=1)
    m2l = getm2l(alfvar.sspgrid.lam, tem_mspec, tem_pos, 
                 alfvar.nstart, alfvar.nend, alfvar.nfil, alfvar.imf_type, )
    m2lmw = getm2l(alfvar.sspgrid.lam, tem_mspec_mw, tem_pos,
                   alfvar.nstart, alfvar.nend, alfvar.nfil, alfvar.imf_type, mw=1)
    return np.append(m2l,m2lmw)


# ---------------------------------------------------------------- #
def calm2l_dynesty(in_res, alfvar, use_keys, outname, ncpu=1):
    print('creating results file:\n {0}results/res_dynesty_{1}.hdf5'.format(ALFPY_HOME, outname))
    f1 = h5py.File("{0}results_dynesty/res_dynesty_{1}.hdf5".format(ALFPY_HOME, outname), "w")
    for ikey in ['samples', 'logwt', 'logl', 'logvol', 'logz', 'logzerr', 'information']:
        dset = f1.create_dataset(ikey, dtype=np.float16, data=getattr(in_res, ikey))

    samples, weights = in_res.samples, np.exp(in_res.logwt - in_res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    samples = dyfunc.resample_equal(in_res.samples, weights)

    dset = f1.create_dataset('samples_eq', dtype=np.float16, data=samples)
    dset = f1.create_dataset('mean', dtype=np.float16, data=mean)
    dset = f1.create_dataset('cov', dtype=np.float16, data=cov)
    dset = f1.create_dataset('use_keys', data=use_keys)

    nspec = samples.shape[0]
    select_ind = np.random.choice(np.arange(nspec), size=1000)
    samples = np.copy(samples[select_ind,:])

    tstart = time.time()
    pwork = partial(worker_m2l, alfvar, use_keys)
    nspec = samples.shape[0]
    m2l_res = Parallel(n_jobs=ncpu)(delayed(pwork)(samples[i]) for i in tqdm(range(nspec)))
    #m2l_res = executor.map(pwork, [ispec for ispec in samples])
    #pool = multiprocessing.Pool(ncpu)
    #m2l_res = pool.map(pwork, [ispec for ispec in samples])
    #pool.close()
    #pool.join()
    ndur = time.time() - tstart
    print('\npost processing dynesty results: {:.2f}minutes'.format(ndur/60.))

    dset = f1.create_dataset('m2l', dtype=np.float16, data=m2l_res)
    f1.close()


# ---------------------------------------------------------------- #
def calm2l_emcee(in_res, alfvar, use_keys, ncpu):
    return None




# ---------------------------------------------------------------- #
def calm2l_mcmc(infile, alfvar, ncpu, outname):
    """
    - for test purpose
    """
    ALFPY_HOME = os.environ['ALFPY_HOME']
    samples = np.array(pd.read_csv(infile, delim_whitespace=True,
                                header=None, comment='#'))[:,1:47]
    tstart = time.time()

    with MultiPool() as pool:
        pwork = partial(worker_m2l, alfvar, key_list)
        ml_res = pool.map(pwork, samples)
    ndur = time.time() - tstart
    pool.close()
    print('\ncalculating m2l in .mcmc file: {:.2f}minutes'.format(ndur/60.))
    np.savez('{0}results_dynesty/{1}_mcmcm2l_b.npz'.format(ALFPY_HOME, outname), m2l=ml_res)
    return np.array(ml_res)
