import os, sys, copy, pickle, numpy as np
import emcee, time
import dynesty
from dynesty import NestedSampler
from tofit_parameters import tofit_params
from func import func
from alf_vars import *
from alf_constants import *
from priors import TopHat,ClippedNormal
from read_data import *
from linterp import *
from str2arr import *
from setup import *
from set_pinit_priors import *

from post_process import calm2l_dynesty

# -------------------------------------------------------- #
def log_prob(inarr):
    res_ = func(global_alfvar, 
                inarr, use_keys,
                prhiarr=global_prhiarr,
                prloarr=global_prloarr)
    return -0.5*res_

# -------------------------------------------------------- #
def log_prob_nested(posarr):
    ln_prior = lnprior(posarr)
    if not np.isfinite(ln_prior):
        return -np.infty
    res_ = func(global_alfvar, posarr, 
                     usekeys=use_keys, prhiarr=global_prhiarr,
                     prloarr=global_prloarr)
    return ln_prior-0.5*res_

# -------------------------------------------------------- #
def prior_transform(unit_coords):
    all_key_list = list(tofit_params.keys())
    return np.array([global_all_prior[all_key_list.index(ikey)].unit_transform(unit_coords[i]) for i, ikey in enumerate(use_keys)])


# -------------------------------------------------------- #
def lnprior(in_arr):
    """
    - only used for dynesty
    - INPUT: npar arr (same length as use_keys)
    - GLOBAL VARIABLES:
        - global_all_priors: priors for all 46 parameter
    """
    full_arr = fill_param(in_arr, use_keys)
    lnp = sum([global_all_prior[i_].lnp(iarr_) for i_, iarr_ in enumerate(full_arr)])
    if np.isfinite(lnp):
        return 0.0
    return lnp


# -------------------------------------------------------- #
def func_2min(inarr):
    """
    only optimize the first 4 parameters before running the sampler
    """
    return func(global_alfvar, inarr, use_keys[:len(inarr)], 
                prhiarr=global_prhiarr,
                prloarr=global_prloarr,
               )


# -------------------------------------------------------- #
def alf(filename, tag='', run='dynesty', pool_type='mpi'):
    """
    - based on alf.f90
    - `https://github.com/cconroy20/alf/blob/master/src/alf.f90`
    Master program to fit the absorption line spectrum, or indices,
    #  of a quiescent (>1 Gyr) stellar population
    # Some important points to keep in mind:
    # 1. The prior bounds on the parameters are specified in set_pinit_priors.
    #    Always make sure that the output parameters are not hitting a prior.
    # 2. Make sure that the chain is converged in all relevant parameters
    #    by plotting the chain trace (parameter vs. chain step).
    # 3. Do not use this code blindly.  Fitting spectra is a
    #    subtle art and the code can easily fool you if you don't know
    #    what you're doing.  Make sure you understand *why* the code is
    #    settling on a particular parameter value.
    # 4. Wavelength-dependent instrumental broadening is included but
    #    will not be accurate in the limit of modest-large redshift b/c
    #    this is implemented in the model restframe at code setup time
    # 5. The code can fit for the atmospheric transmission function but
    #    this will only work if the input data are in the original
    #    observed frame; i.e., not de-redshifted.
    # 6. I've found that Nwalkers=1024 and Nburn=~10,000 seems to
    #    generically yield well-converged solutions, but you should test
    #    this yourself by fitting mock data generated with write_a_model
    # To Do: let the Fe-peak elements track Fe in simple mode
    """
    ALFPY_HOME = os.environ['ALFPY_HOME']
    pickle_model_name = '{0}alfvar_models/alfvar_model_{1}_{2}.p'.format(ALFPY_HOME, filename, tag)
    global use_keys
    use_keys = [k for k, (v1, v2) in tofit_params.items() if v1 == True]
    
    global global_alfvar
    global global_all_prior
    global global_prloarr, global_prhiarr
    try:
        [global_alfvar, global_prloarr, global_prhiarr, global_all_prior, optimize_res_x] = pickle.load(open(pickle_model_name, "rb" ))
    except:
        print('cannot find model_arr at {0}. \n Please run alf_build_model first'.format(pickle_model_name))
        return 

    nmcmc = 100    # -- number of chain steps to print to file
    # -- inverse sampling of the walkers for printing
    # -- NB: setting this to >1 currently results in errors in the *sum outputs
    nsample = 1
    nburn = 50000    # -- length of chain burn-in
    nwalkers = 512    # -- number of walkers
    npar = len(use_keys)
    all_key_list = list(tofit_params.keys())
    
    # ==== works should have all info ===== #
    # ========== initialize pool ========== #
    if pool_type == 'multiprocessing':
        import multiprocessing
        from multiprocessing import Pool as to_use_pool        
    else:
        from schwimmbad import MPIPool as to_use_pool 
    with to_use_pool() as pool:
        if pool_type == 'mpi':
            ncpu = pool.size
            print('pool size', pool.size)
            if not pool.is_master():
                pool.wait()
                sys.exit(0) 
        else:
            ncpu = multiprocessing.cpu_count()
        print('ncpu=', ncpu)
        # ---------------------------------------------------------------- #
        if run == 'emcee':
            pos_emcee_in = np.zeros(shape=(nwalkers, npar))
            prrange = [10, 10, 0.1, 0.1]
            for i in range(npar):
                if i <4:
                    min_ = max(global_prloarr[i], np.array(optimize_res_x)[i]-prrange[i])
                    max_ = min(global_prhiarr[i], np.array(optimize_res_x)[i]+prrange[i])
                    pos_emcee_in[:, i] = np.array([np.random.uniform(min_, max_, nwalkers)])                
                else:
                    tem_prior = np.take(
                        global_all_prior, 
                        all_key_list.index(use_keys[i])
                    )
                    pos_emcee_in[:, i] = np.array(
                        [np.random.uniform(tem_prior.range[0], 
                                           tem_prior.range[1], 
                                           nwalkers)])
                
            print('Initializing emcee with nwalkers=%.0f, npar=%.0f' %(nwalkers, npar))
            tstart = time.time()
            sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob, pool=pool)
            sampler.run_mcmc(pos_emcee_in, nburn + nmcmc, progress=True, skip_initial_state_check=True)
            print('mean acc fraction %.3f' %np.nanmean(sampler.acceptance_fraction))
            ndur = time.time() - tstart
            print('\n Total time for emcee {:.2f}min'.format(ndur/60))
            res = sampler.get_chain(discard = nburn) # discard: int, burn-in
            prob = sampler.get_log_prob(discard = nburn)
            pickle.dump(res, open('{0}results_emcee/res_emcee_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ) )
            pickle.dump(prob, open('{0}results_emcee/prob_emcee_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ) )

        # ---------------------------------------------------------------- #
        elif run == 'dynesty':
            dsampler = dynesty.NestedSampler(log_prob_nested, prior_transform, 
                                             npar, nlive = int(50*npar),
                                             sample='rslice', bootstrap=0, 
                                             pool=pool, queue_size=ncpu)
            ncall = dsampler.ncall
            niter = dsampler.it - 1
            tstart = time.time()
            dsampler.run_nested(dlogz=0.5)
            ndur = time.time() - tstart
            print('\n Total time for dynesty {:.2f}hrs'.format(ndur/60./60.))

            results = dsampler.results
            pickle.dump(results, open('{0}results_dynesty/res_dynesty_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ))

            results = pickle.load(open('{0}results_dynesty/res_dynesty_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "rb" ))
            # ---- post process ---- #
            calm2l_dynesty(results, alfvar, use_keys=use_keys, outname=filename+'_'+tag, pool=pool)

# -------------------------------- #
# ---- command line arguments ---- #
# -------------------------------- #
if __name__ == "__main__":
    ncpu = os.getenv('SLURM_NTASKS')
    ncpu_pertask = os.getenv('SLURM_CPUS_PER_TASK')
    os.environ["OMP_NUM_THREADS"] = "1"
    if ncpu is None:
        import multiprocessing
        pool_type = 'multiprocessing'
        ncpu = multiprocessing.cpu_count()
    elif ncpu_pertask>ncpu:
        pool_type = 'multiprocessing'
        ncpu = ncpu_pertask
    else:
        pool_type = 'mpi'

    print('\npool type:', pool_type)      
    print('ncpu=', ncpu)
    argv_l = sys.argv
    n_argv = len(argv_l)
    filename = argv_l[1]
    tag = ''
    if n_argv >= 3:
        tag = argv_l[2]
    run = 'dynesty'
    print('\nrunning alf:\ninput spectrum:{0}.dat'.format(filename))
    print('sampler = {0}'.format(run))
    alf(filename, tag, run=run, pool_type = pool_type)
