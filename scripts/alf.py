import os, sys, copy, pickle, numpy as np
import matplotlib.pyplot as plt
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
def alf(filename, tag='', run='dynesty', pool_type='multiprocessing', save_chains=True):
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

            if save_chains:
                sample_file = '{0}results_emcee/{1}_{2}.h5'.format(ALFPY_HOME, filename, tag)
                backend = emcee.backends.HDFBackend(sample_file, name = 'burn in')
                backend.reset(nwalkers, npar)
            else:
                backend = None


            # ---------------- run initial burn-in ----------------- #
            sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob, pool=pool,
                                            moves=[emcee.moves.StretchMove(a=1.5)],
                                            backend=backend)

            p_mean_last = pos_emcee_in.mean(0)
            log_prob_mean_last = -np.inf
            for j, sample in enumerate(sampler.sample(pos_emcee_in, iterations=3000, progress=True)):
                it = j+1
                if (it % 100): continue
                ndur = (time.time() - tstart)/60
                p_mean = sampler.get_chain(flat=True, thin=1, discard = it-100).mean(0)
                log_prob_mean = sampler.get_log_prob(discard=it-100).mean()
                # delta log probability
                dlogP = np.log10(np.abs((log_prob_mean - log_prob_mean_last) / log_prob_mean))
                # mean change in each parameter
                dmean = np.abs(np.mean((p_mean-p_mean_last)/p_mean))
                print(f'iter = {it}, ' +
                      f'acceptance fraction = {sampler.acceptance_fraction.mean():.2f}, ' +
                      f'd(logP) = {dlogP:0.2f}, ' +
                      f'd(mean) = {dmean:0.4f}, ' +
                      f'time={ndur:.2f}min',
                      flush=True)

                if (it>500)&(dmean < 0.01)&(dlogP<-4): break 

                p_mean_last = p_mean
                log_prob_mean_last = log_prob_mean


            # ---------------- run burn-in ----------------- #
            # take best walker and re-initialize there
            last_state = sampler.get_last_sample()
            best_walker = last_state.coords[last_state.log_prob.argmax()]

            # reinitialize from best walker (within +/- 5% of total prior range)
            for i in range(npar):
                tem_prior = np.take(
                    global_all_prior, 
                    all_key_list.index(use_keys[i])
                )
                min_ = max(global_prloarr[i], np.array(best_walker)[i]-0.05*np.diff(tem_prior.range))
                max_ = min(global_prhiarr[i], np.array(best_walker)[i]+0.05*np.diff(tem_prior.range))
                pos_emcee_in[:, i] = np.array(
                    [np.random.uniform(tem_prior.range[0], 
                                    tem_prior.range[1], 
                                    nwalkers)])

            if save_chains:
                backend = emcee.backends.HDFBackend(sample_file, name=f"samples")
                backend.reset(nwalkers, npar)
            
            
            sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob, pool=pool,
                                            moves=[emcee.moves.StretchMove(a=1.1)],
                                            backend=backend)

            old_tau = np.inf
            num = it
            converged = False
            for j, sample in enumerate(sampler.sample(pos_emcee_in, iterations=60000, progress=True)):
                it = j+1
                if it%100: continue
                ndur = (time.time() - tstart)/60
                # use the tau (without discarding) to determine how much to remove
                tau = sampler.get_autocorr_time(discard=int(np.max(old_tau)) if \
                                                np.all(np.isfinite(old_tau)) else 0,
                                                tol=0) 

                print(f'iter = {it}, ' +
                      f"tau = {np.max(tau):.0f}, " +
                      f"acceptance fraction = {sampler.acceptance_fraction.mean():.2f}, " +
                      f"dtau = {np.max((tau-old_tau)/tau):.2f}, " +
                      f"it/tau = {np.min(it/tau):.1f}, "+
                      f"time={ndur:.2f}min",
                      flush=True)

                converged = np.all(tau * 20 < it)
                converged &= np.all((tau-old_tau)/tau < 0.01)
                old_tau = tau

                if converged: break

                # plot chain update!
                if it % 1000:
                    continue
                samples = sampler.get_chain(thin=int(np.max(tau) / 2))
                nlabels = len(use_keys)
                fig, axes = plt.subplots(nlabels, figsize=(10, 20), sharex=True)
                for i in range(samples.shape[2]):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_ylabel(use_keys[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")
                plt.savefig(f'{ALFPY_HOME}results_emcee/check_chains_{filename}_{tag}.png',dpi=300)
                plt.close('all')


            ndur = time.time() - tstart
            print('\n Total time for emcee {:.2f}min'.format(ndur/60))
            res = sampler.get_chain(discard = nburn) # discard: int, burn-in
            prob = sampler.get_log_prob(discard = nburn)
            pickle.dump(res, open('{0}results_emcee/res_emcee_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ) )
            pickle.dump(prob, open('{0}results_emcee/prob_emcee_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ) )
            
            # get model spectra for best walker
            idx_min_prob = np.where(prob == np.amin(prob))
            _, min_prob_spec = func(global_alfvar, res[idx_min_prob][0], prhiarr=global_prhiarr,
                                prloarr=global_prloarr, usekeys=use_keys,
                                    funit=True)
            
            mcmc = res.reshape(res.shape[0]*res.shape[1], res.shape[2])
            pcs = np.percentile(mcmc,[5,16,50,84,95],axis=0)
            specs = {'wave':global_alfvar.data.lam,
                    'flux':global_alfvar.data.flx,
                    'err':global_alfvar.data.err,
                    'min_prob':min_prob_spec}
            for p in pcs:
                _, spec = func(global_alfvar, pcs[0], prhiarr=global_prhiarr,
                                prloarr=global_prloarr, usekeys=use_keys,
                                    funit=True)
                specs[str(p)] = spec

            pickle.dump(specs, open(f'{ALFPY_HOME}results_emcee/specs_emcee_{filename}.p', "wb" ) )


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
            calm2l_dynesty(results, global_alfvar, use_keys=use_keys, outname=filename+'_'+tag, pool=pool)

# -------------------------------- #
# ---- command line arguments ---- #
# -------------------------------- #
if __name__ == "__main__":
    import multiprocessing

    ncpu = os.getenv('SLURM_CPUS_PER_TASK')
    os.environ["OMP_NUM_THREADS"] = "1"
    if ncpu is None:
        pool_type = 'multiprocessing'
        ncpu = multiprocessing.cpu_count()
    else:
        pool_type = 'multiprocessing'

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
