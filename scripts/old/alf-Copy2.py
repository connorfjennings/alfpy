import os, sys, copy, pickle, numpy as np
import emcee, time
#import multiprocessing
#from multiprocessing import Pool
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

from scipy.optimize import differential_evolution
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
def alf(filename, tag='', run='dynesty', model_arr = None, ncpu=1):
    """
    - based on alf.f90
    - `https://github.com/cconroy20/alf/blob/master/src/alf.f90`
    - use use_keys to define parameters to fit.  Others will remain
      as set_pinit_priors().
    - works fine for 4 parameters so far
    - emcee: run='emcee'
    - dynesty: run = 'dynesty'

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

    if model_arr is not None:
        print('\nPickle loading alf model array: '+model_arr+'\n')
        alfvar = pickle.load(open(model_arr, "rb" ))
    else:
        pickle_model_name = '{0}pickle/alfvar_sspgrid_{1}.p'.format(ALFPY_HOME, filename)
        print('No existing model array.  We will create one and pickle dump it to \n'+pickle_model_name)
        alfvar = ALFVAR()


    nmcmc = 200    # -- number of chain steps to print to file
    # -- inverse sampling of the walkers for printing
    # -- NB: setting this to >1 currently results in errors in the *sum outputs
    nsample = 1
    nburn = 0    # -- length of chain burn-in
    nwalkers = 512    # -- number of walkers

    # -- check
    minchi2 = huge_number
    npar = alfvar.npar

    #---------------------------------------------------------------!
    #---------------------------Setup-------------------------------!
    #---------------------------------------------------------------!
    # ---- flag specifying if fitting indices or spectra
    alfvar.fit_indices = 0  #flag specifying if fitting indices or spectra

    # ---- flag determining the level of complexity
    # ---- 0=full, 1=simple, 2=super-simple.  See sfvars for details
    alfvar.fit_type = 0  # do not change; use use_keys to specify parameters

    # ---- fit h3 and h4 parameters
    alfvar.fit_hermite = 0

    # ---- type of IMF to fit
    # ---- 0=1PL, 1=2PL, 2=1PL+cutoff, 3=2PL+cutoff, 4=non-parametric IMF
    alfvar.imf_type = 1

    # ---- are the data in the original observed frame?
    alfvar.observed_frame = 0
    alfvar.mwimf = 0  #force a MW (Kroupa) IMF

    # ---- fit two-age SFH or not?  (only considered if fit_type=0)
    alfvar.fit_two_ages = 1

    # ---- IMF slope within the non-parametric IMF bins
    # ---- 0 = flat, 1 = Kroupa, 2 = Salpeter
    alfvar.nonpimf_alpha = 2

    # ---- turn on/off the use of an external tabulated M/L prior
    alfvar.extmlpr = 0

    # ---- set initial params, step sizes, and prior ranges
    _, prlo,prhi = set_pinit_priors(alfvar.imf_type)
    
    # ---- change the prior limits to kill off these parameters
    #pos, prlo, prhi = set_pinit_priors(alfvar)
    prhi.logm7g = -5.0
    prhi.teff   =  2.0
    prlo.teff   = -2.0

    # ---- mass of the young component should always be sub-dominant
    prhi.logfy = -0.5

    # ---------------------------------------------------------------!
    # --------------Do not change things below this line-------------!
    # ---------------unless you know what you are doing--------------!
    # ---------------------------------------------------------------!

    # ---- regularize non-parametric IMF (always do this)
    alfvar.nonpimf_regularize = 1

    # ---- dont fit transmission function in cases where the input
    # ---- spectrum has already been de-redshifted to ~0.0
    if alfvar.observed_frame == 0 or alfvar.fit_indices == 1:
        alfvar.fit_trans = 0
        prhi.logtrans = -5.0
        prhi.logsky   = -5.0
    else:
        alfvar.fit_trans = 1
    """
    # ---- extra smoothing to the transmission spectrum.
    # ---- if the input data has been smoothed by a gaussian
    # ---- in velocity space, set the parameter below to that extra smoothing
    """
    alfvar.smooth_trans = 0.0

    if (alfvar.ssp_type == 'cvd'):
        # ---- always limit the [Z/H] range for CvD since
        # ---- these models are actually only at Zsol
        prhi.zh =  0.01
        prlo.zh = -0.01
        if (alfvar.imf_type > 1):
            print('ALF ERROR, ssp_type=cvd but imf>1')

    if alfvar.fit_type in [1,2]:
        alfvar.mwimf=1

    #---------------------------------------------------------------!

    if filename is None:
        print('ALF ERROR: You need to specify an input file')
        teminput = input("Name of the input file: ")
        if len(teminput.split(' '))==1:
            filename = teminput
        elif len(teminput.split(' '))>1:
            filename = teminput[0]
            tag = teminput[1]


    # ---- write some important variables to screen
    print(" ************************************")
    if alfvar.fit_indices == 1:
        print(" ***********Index Fitter*************")
    else:
        print(" **********Spectral Fitter***********")
    print(" ************************************")
    print("   ssp_type  =", alfvar.ssp_type)
    print("   fit_type  =", alfvar.fit_type)
    print("   imf_type  =", alfvar.imf_type)
    print(" fit_hermite =", alfvar.fit_hermite)
    print("fit_two_ages =", alfvar.fit_two_ages)
    if alfvar.imf_type == 4:
        print("   nonpimf   =", alfvar.nonpimf_alpha)
    print("  obs_frame  =",  alfvar.observed_frame)
    print("      mwimf  =",  alfvar.mwimf)
    print("  age-dep Rf =",  alfvar.use_age_dep_resp_fcns)
    print("    Z-dep Rf =",  alfvar.use_z_dep_resp_fcns)
    print("  Nwalkers   = ",  nwalkers)
    print("  Nburn      = ",  nburn)
    print("  Nchain     = ",  nmcmc)
    #print("  Ncores     = ",  ntasks)
    print("  filename   = ",  filename, ' ', tag)
    print(" ************************************")
    #print('\n\nStart Time ',datetime.now())

    #---------------------------------------------------------------!
    # ---- read in the data and wavelength boundaries
    alfvar.filename = filename
    alfvar.tag = tag


    alfvar = read_data(alfvar)
    print("  Fitting ",alfvar.nlint," wavelength intervals")
    nlint = alfvar.nlint
    l1, l2 = alfvar.l1, alfvar.l2
    print('wavelength bourdaries: ', l1, l2)
    
    # ---- read in the SSPs and bandpass filters
    # ------- setting up model arry with given imf_type ---- #
    global global_alfvar, global_prloarr, global_prhiarr
    global use_keys
    global global_all_prior  # ---- note it's for all parameters
    # ---- convert the structures into their equivalent arrays
    prloarr = str2arr(switch=1, instr = prlo)
    prhiarr = str2arr(switch=1, instr = prhi)
    global_prloarr = prloarr
    global_prhiarr = prhiarr 
    use_keys = [k for k, (v1, v2) in tofit_params.items() if v1 == True]
    npar = len(use_keys)
    global_alfvar = alfvar # ? need to double check this
      
    from schwimmbad import MPIPool
    pool = MPIPool()
    if model_arr is None:
        print('\nsetting up model arry with given imf_type and input data\n')
        tstart = time.time()
        alfvar = setup(alfvar, onlybasic = False, pool_type = 'MPI')
        pickle_model_name = '{0}pickle/alfvar_sspgrid_{1}.p'.format(ALFPY_HOME, filename)
        pickle.dump(alfvar, open(pickle_model_name, "wb" ))
        ndur = time.time() - tstart
        print('\n Total time for setup {:.2f}min'.format(ndur/60))


    ## ---- This part requires alfvar.sspgrid.lam ---- ##
    lam = np.copy(alfvar.sspgrid.lam)
    # ---- we only compute things up to 500A beyond the input fit region
    alfvar.nl_fit = min(max(locate(lam, alfvar.l2[-1]+500.0),0),alfvar.nl-1)
    ## ---- define the log wavelength grid used in velbroad.f90
    alfvar.dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-
                         np.log(alfvar.sspgrid.lam[0]))/alfvar.nl_fit
    for i in range(alfvar.nl_fit):
        alfvar.lnlam[i] = i*alfvar.dlstep + np.log(alfvar.sspgrid.lam[0])
    if l2[-1]>np.nanmax(lam) or l1[0]<np.nanmin(lam):
        print('ERROR: wavelength boundaries exceed model wavelength grid')
        print(l2[nlint-1],lam[nl-1],l1[0],lam[0])
        

    # ---- this is the master process ---- #
    global_alfvar = alfvar
    # -------- optimize the first four parameters -------- #
    # ==== turn off differential_evolution to compare with alf fortran ==== #
    len_optimize = 4
    prior_bounds = list(zip(global_prloarr[:len_optimize], global_prhiarr[:len_optimize]))
    print('prior_bounds:', prior_bounds)
    optimize_res = differential_evolution(func_2min, bounds = prior_bounds, disp=True,
                                          polish=False, updating='deferred', workers=-1)
    print('optimized parameters', optimize_res)

    # -------- getting priors for the sampler -------- #
    all_key_list = list(tofit_params.keys())
    # ---------------- update priors ----------------- #
    prrange = [10, 10, 0.1, 0.1]
    global_all_prior = [ClippedNormal(np.array(optimize_res.x)[i], prrange[i],
                                      global_prloarr[i], 
                                      global_prhiarr[i]) for i in range(len_optimize)] + \
                       [TopHat(global_prloarr[i+len_optimize], 
                               global_prhiarr[i+len_optimize]) for i in range(len(all_key_list)-len_optimize)]
    # ---------------- update priors ----------------- #
    print('\nWe are going to fit ', npar, 'parameters\nThey are', use_keys)
    
    # ---------------------------------------------------------------- #
    if run == 'emcee':
        print('Initializing emcee with nwalkers=%.0f, npar=%.0f' %(nwalkers, npar))
        tstart = time.time()
        pos_emcee_in = np.zeros(shape=(nwalkers, npar))
        for i in range(npar):
            if i <4:
                min_ = max(prloarr[i], np.array(optimize_res.x)[i]-prrange[i])
                max_ = min(prhiarr[i], np.array(optimize_res.x)[i]+prrange[i])
                pos_emcee_in[:, i] = np.array([np.random.uniform(min_, max_, nwalkers)])                
            else:
                tem_prior = np.take(global_all_prior, all_key_list.index(use_keys[i]))
                pos_emcee_in[:, i] = np.array(
                    [np.random.uniform(tem_prior.range[0], tem_prior.range[1], nwalkers)])
        if pool.is_master():
            sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob, pool=pool)
            sampler.run_mcmc(pos_emcee_in, nburn + nmcmc, progress=True, skip_initial_state_check=True)
            pool.close()
        else:
            pool.wait()
        #with multiprocessing.Pool(ncpu) as pool:
        #    sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob, pool=pool)
        #    sampler.run_mcmc(pos_emcee_in, nburn + nmcmc, progress=True, skip_initial_state_check=True)

        os.system('mkdir -p {0}results_emcee'.format(ALFPY_HOME))
        ndur = time.time() - tstart
        print('\n Total time for emcee {:.2f}min'.format(ndur/60))
        res = sampler.get_chain(discard = nburn) # discard: int, burn-in
        prob = sampler.get_log_prob(discard = nburn)
        pickle.dump(res, open('{0}results_emcee/res_emcee_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ) )
        pickle.dump(prob, open('{0}results_emcee/prob_emcee_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ) )


    # ---------------------------------------------------------------- #
    elif run == 'dynesty':
        with multiprocessing.Pool(ncpu) as pool:
            dsampler = dynesty.NestedSampler(log_prob_nested, prior_transform, 
                                             npar, nlive = int(50*npar),
                                             sample='rslice', bootstrap=0,pool=pool, 
                                             queue_size = ncpu)
            ncall = dsampler.ncall
            niter = dsampler.it - 1
            tstart = time.time()
            dsampler.run_nested(dlogz=0.5)
            ndur = time.time() - tstart
            print('\n Total time for dynesty {:.2f}hrs'.format(ndur/60./60.))
        pool.close()
        results = dsampler.results
        os.system('mkdir -p {0}results_dynesty'.format(ALFPY_HOME))
        pickle.dump(results, open('{0}results_dynesty/res_dynesty_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "wb" ))

        results = pickle.load(open('{0}results_dynesty/res_dynesty_{1}_{2}.p'.format(ALFPY_HOME, filename, tag), "rb" ))
        # ---- post process ---- #
        calm2l_dynesty(results, alfvar, use_keys=use_keys, outname=filename+'_'+tag, ncpu=ncpu)


# -------------------------------- #
# ---- command line arguments ---- #
# -------------------------------- #


if __name__ == "__main__":
    import multiprocessing
    #ncpu = multiprocessing.cpu_count()
    ncpu = 8
    print('Number of available cpu:', ncpu)
    argv_l = sys.argv
    n_argv = len(argv_l)
    filename = argv_l[1]
    tag = ''
    if n_argv >= 3:
        tag = argv_l[2]
    run = 'emcee'
    print('\nrunning alf:\ninput spectrum:{0}.dat'.format(filename))
    print('sampler = {0}'.format(run))

    dir0 = '{0}/pickle/'.format(os.environ['ALFPY_HOME'])
    alf(filename, 
        tag,
        model_arr =  dir0+'alfvar_sspgrid_zsol_na+0.0.p',
        run=run, ncpu=ncpu)







