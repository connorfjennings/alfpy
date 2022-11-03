# ---- build alfvar model and pickle dump it

import os, sys, copy, pickle, numpy as np
import time
from tofit_parameters import tofit_params
from func import func
from alf_vars import *
from alf_constants import *
from priors import TopHat,ClippedNormal
from read_data import *
from linterp import *
from str2arr import *
from getvelz import getvelz
from setup import *
from set_pinit_priors import *
from scipy.optimize import differential_evolution
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
def build_alf_model(filename, tag='', pool_type='multiprocessing'):
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
    for ifolder in ['alfvar_models', 'results_emcee', 'results_dynesty', 'subjobs']:
        if os.path.exists(ALFPY_HOME+ifolder) is not True:
            os.makedirs(ALFPY_HOME+ifolder)
    
    pickle_model_name = '{0}alfvar_models/alfvar_model_{1}_{2}.p'.format(ALFPY_HOME, filename, tag)
    print('We will create one and pickle dump it to \n'+pickle_model_name)
    alfvar = ALFVAR()

    global use_keys
    use_keys = [k for k, (v1, v2) in tofit_params.items() if v1 == True]
    
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
    alfvar.imf_type = 3

    # ---- are the data in the original observed frame?
    alfvar.observed_frame = 0
    alfvar.mwimf = 0  #force a MW (Kroupa) IMF

    if alfvar.mwimf:
        alfvar.imf_type = 1

    # ---- fit two-age SFH or not?  (only considered if fit_type=0)
    alfvar.fit_two_ages = 1

    # ---- IMF slope within the non-parametric IMF bins
    # ---- 0 = flat, 1 = Kroupa, 2 = Salpeter
    alfvar.nonpimf_alpha = 2

    # ---- turn on/off the use of an external tabulated M/L prior
    alfvar.extmlpr = 0

    # ---- set initial params, step sizes, and prior ranges
    _, prlo, prhi = set_pinit_priors(alfvar.imf_type)
    
    # ---- change the prior limits to kill off these parameters
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

    # ---- extra smoothing to the transmission spectrum.
    # ---- if the input data has been smoothed by a gaussian
    # ---- in velocity space, set the parameter below to that extra smoothing
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
    #print("  Ncores     = ",  ntasks)
    print("  filename   = ",  filename, ' ', tag)
    print(" ************************************")
    #print('\n\nStart Time ',datetime.now())

    #---------------------------------------------------------------!
    
    # ---- read in the data and wavelength boundaries
    alfvar.filename = filename
    alfvar.tag = tag

    if alfvar.fit_indices == 0:
        alfvar = read_data(alfvar)
        # ---- read in the SSPs and bandpass filters
        # ------- setting up model arry with given imf_type ---- #

        if pool_type == 'multiprocessing':
            from multiprocessing import Pool as to_use_pool        
        else:
            from schwimmbad import MPIPool as to_use_pool 
            
        pool = to_use_pool()
        if pool_type == 'mpi':
            print('pool size', pool.size)
            if not pool.is_master():
                pool.wait()
                sys.exit(0) 
                
        print('\nsetting up model arry with given imf_type and input data\n')
        tstart = time.time()
        #alfvar = setup(alfvar, onlybasic = False, pool = pool)
        alfvar = setup(alfvar, onlybasic = True, pool = pool)
        ndur = time.time() - tstart
        print('\n Total time for setup {:.2f}min'.format(ndur/60))


        ## ---- This part requires alfvar.sspgrid.lam ---- ##
        lam = np.copy(alfvar.sspgrid.lam)
        # ---- interpolate the sky emission model onto the observed wavelength grid
        # ---- moved to read_data
        if alfvar.observed_frame == 1:
            alfvar.data.sky = linterp(alfvar.lsky, alfvar.fsky, alfvar.data.lam)
            alfvar.data.sky[alfvar.data.sky<0] = 0.
        else:
            alfvar.data.sky[:] = tiny_number
        alfvar.data.sky[:] = tiny_number  # ?? why?

        # ---- we only compute things up to 500A beyond the input fit region
        alfvar.nl_fit = min(max(locate(lam, alfvar.l2[-1]+500.0),0),alfvar.nl-1)
        ## ---- define the log wavelength grid used in velbroad.f90
        alfvar.dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-
                         np.log(alfvar.sspgrid.lam[0]))/(alfvar.nl_fit+1)

        for i in range(alfvar.nl_fit):
            alfvar.lnlam[i] = i*alfvar.dlstep + np.log(alfvar.sspgrid.lam[0])


    # ---- convert the structures into their equivalent arrays
    prloarr = str2arr(switch=1, instr = prlo)
    prhiarr = str2arr(switch=1, instr = prhi)

    # ---- this is the master process
    # ---- estimate velz ---- #
    print("  Fitting ",alfvar.nlint," wavelength intervals")
    nlint = alfvar.nlint
    l1, l2 = alfvar.l1, alfvar.l2
    print('wavelength bourdaries: ', l1, l2)
    if l2[-1]>np.nanmax(lam) or l1[0]<np.nanmin(lam):
        print('ERROR: wavelength boundaries exceed model wavelength grid')
        print(l2[nlint-1],lam[nl-1],l1[0],lam[0])
        
    global global_alfvar, global_prloarr, global_prhiarr
    global_alfvar = copy.deepcopy(alfvar)
    global_prloarr = copy.deepcopy(prloarr)
    global_prhiarr = copy.deepcopy(prhiarr)
    # -------- optimize the first four parameters -------- #    
    len_optimize = 4
    all_key_list = list(tofit_params.keys())
    prloarr_usekeys = np.array([global_prloarr[i_] for i_, k_ in enumerate(all_key_list) if k_ in use_keys])
    prhiarr_usekeys = np.array([global_prhiarr[i_] for i_, k_ in enumerate(all_key_list) if k_ in use_keys])
    
    print('will narrow prior for the following parameters: \n', use_keys[:len_optimize])
    prior_bounds = list(zip(prloarr_usekeys[:len_optimize], prhiarr_usekeys[:len_optimize]))
    print('prior_bounds:\n', prior_bounds)
    
    if ~alfvar.observed_frame:
        prior_bounds[0] = (-200,200)
    optimize_res = differential_evolution(func_2min, bounds = prior_bounds, disp=True,
                                          polish=False, updating='deferred', workers=1)
    print('optimized parameters', optimize_res)
    
    # -------- getting priors for the sampler -------- #
    global global_all_prior  # ---- note it's for all parameters
    
    # ---------------- update priors ----------------- #
    prrange = [10, 10, 0.1, 0.1]
    global_all_prior = [ClippedNormal(np.array(optimize_res.x)[i], prrange[i],
                                      global_prloarr[i], 
                                      global_prhiarr[i]) for i in range(len_optimize)] + \
                       [TopHat(global_prloarr[i+len_optimize], 
                               global_prhiarr[i+len_optimize]) for i in range(len(all_key_list)-len_optimize)]
    
    # ---- update on 11/2/2022 ---- #
    for i_, k_ in enumerate(all_key_list):
        if i_ <=3:
            continue
        if k_ in use_keys:
            prrange_ = global_prhiarr[i_] - global_prloarr[i_]
            global_all_prior[i_] = TopHat(
                max(global_prloarr[i_], optimize_res.x[use_keys.index(k_)]-0.25*prrange_), 
                min(global_prhiarr[i_], optimize_res.x[use_keys.index(k_)]+0.25*prrange_))
     
    pickle.dump([alfvar, prloarr, prhiarr, global_all_prior, optimize_res.x], open(pickle_model_name, "wb" ))
    pool.close()

    
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
    print(ncpu, 'cores')
    argv_l = sys.argv
    n_argv = len(argv_l)
    filename = argv_l[1]
    tag = ''
    if n_argv >= 3:
        tag = argv_l[2]
    print('\nrunning alf:\ninput spectrum:{0}.dat'.format(filename))

    build_alf_model(filename, tag, pool_type = pool_type)








