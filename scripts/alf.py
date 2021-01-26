import os, sys, copy, pickle, numpy as np
import emcee, time
from multiprocessing import Pool
import dynesty
from dynesty import NestedSampler, DynamicNestedSampler
from contextlib import closing

#import mpi4py
#from mpi4py import MPI
#from schwimmbad import MPIPool
from schwimmbad import MultiPool

from func import func
from alf_vars import *
from alf_constants import *
from priors import TopHat
from read_data import *
from linterp import *
from str2arr import *
from getvelz import getvelz
from setup import *
from set_pinit_priors import *


# -------------------------------------------------------- #
global key_list
global use_keys
use_keys = ['velz', 'sigma', 'logage', 'zh', 'feh', 
            'ah','ch','nh','mgh']

# -------------------------------------------------------- #
def log_prob(posarr):
    ln_prior = lnprior(posarr, nested = False)
    if not np.isfinite(ln_prior):
        return -np.inf

    return ln_prior - 0.5*func(global_alfvar, posarr, prhiarr=global_prhiarr,
                               prloarr=global_prloarr, usekeys=use_keys)


# -------------------------------------------------------- #
def log_prob_nested(posarr):
    ln_prior = lnprior(posarr, nested = True)
    if not np.isfinite(ln_prior):
        return -np.infty

    return ln_prior - 0.5*func(global_alfvar, posarr, prhiarr=global_prhiarr,
                               prloarr=global_prloarr, usekeys=use_keys)


# -------------------------------------------------------- #
def prior_transform(unit_coords):
    return np.array([global_all_prior[key_list.index(ikey)].unit_transform(unit_coords[i]) for i, ikey in enumerate(use_keys)])


# -------------------------------------------------------- #
def lnprior(in_arr, nested=False):
    """
    - INPUT: npar arr (same length as use_keys)
    - GLOBAL VARIABLES:
        - global_all_priors: priors for all 46 parameter
    """
    in_pos_arr = fill_param(in_arr, use_keys)
    lnp = np.nansum([global_all_prior[i].lnp(in_pos_arr[i]) for i in range(in_pos_arr.size)])
    if nested and np.isfinite(lnp):
        return 0.0
    return lnp


# -------------------------------------------------------- #
def alf(filename, alfvar=None, tag='', run='dynesty',
        model_arr = None, ncpu=4, outname='test'):
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

    if alfvar is None:
        if model_arr is not None:
            print('\nPickle loading alf model array: '+model_arr+'\n')
            alfvar = pickle.load(open(model_arr, "rb" ))
        else:
            pickle_model_name = 'pickle/alfvar_sspgrid_'+filename+'.p'
            print('No existing model array.  We will create one and pickle dump it to '+pickle_model_name)
            alfvar = ALFVAR()

    nmcmc = 300    # -- number of chain steps to print to file
    # -- inverse sampling of the walkers for printing
    # -- NB: setting this to >1 currently results in errors in the *sum outputs
    nsample = 1
    nburn = 50000    # -- length of chain burn-in
    nwalkers = 512    # -- number of walkers
    print_mcmc = 1; print_mcmc_spec = 0    # -- save the chain outputs to file and the model spectra

    dopowell = 0  # -- start w/ powell minimization?
    ftol = 0.1    # -- Powell iteration tolerance
    # -- if set, will print to screen timing of likelihood calls
    test_time = 0
    # -- number of Monte Carlo realizations of the noise for index errors
    nmcindx = 1000

    # -- check
    totacc = 0; iter_ = 30
    minchi2 = huge_number
    bret = huge_number

    nl = alfvar.nl
    npar = alfvar.npar
    nfil = alfvar.nfil

    mspec, mspecmw, lam = np.zeros((3, nl))
    m2l, m2lmw = np.zeros((2, nfil))
    oposarr, bposarr = np.zeros((2, npar))
    #runtot = np.zeros((3,npar+2*nfil))
    #cl2p5,cl16,cl50,cl84,cl97p5 = np.zeros((5, npar+2*nfil))

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
    alfvar.observed_frame = 1

    alfvar.mwimf = 0  #force a MW (Kroupa) IMF

    # ---- fit two-age SFH or not?  (only considered if fit_type=0)
    alfvar.fit_two_ages = 1

    # ---- IMF slope within the non-parametric IMF bins
    # ---- 0 = flat, 1 = Kroupa, 2 = Salpeter
    alfvar.nonpimf_alpha = 2

    # ---- turn on/off the use of an external tabulated M/L prior
    alfvar.extmlpr = 0

    # ---- change the prior limits to kill off these parameters
    pos, prlo, prhi = set_pinit_priors(alfvar)
    prhi.logm7g = -5.0
    prhi.teff   =  2.0
    prlo.teff   = -2.0


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


    if alfvar.fit_indices == 0:
        alfvar = read_data(alfvar)
        # ---- read in the SSPs and bandpass filters
        # ------- setting up model arry with given imf_type ---- #

        if model_arr is None:
            print('\nsetting up model arry with given imf_type and input data\n')
            tstart = time.time()
            alfvar = setup(alfvar, onlybasic = False, ncpu=ncpu)
            os.system('mkdir -p ../pickle')
            pickle_model_name = '../pickle/alfvar_sspgrid_'+filename+'.p'
            pickle.dump(alfvar, open(pickle_model_name, "wb" ))
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
        ##define the log wavelength grid used in velbroad.f90
        alfvar.dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-
                         np.log(alfvar.sspgrid.lam[0]))/alfvar.nl_fit

        for i in range(alfvar.nl_fit):
            alfvar.lnlam[i] = i*alfvar.dlstep + np.log(alfvar.sspgrid.lam[0])
            
        # ---- test 01/25/21 ---- #
        alfvar.data.flx[alfvar.data.wgt==0] = np.nan
        alfvar.data.err[alfvar.data.wgt==0] = np.nan
        # ---- masked regions have wgt=0.0.  We'll use wgt as a pseudo-error
        # ---- array in contnormspec, so turn these into large numbers
        alfvar.data.wgt = 1./(alfvar.data.wgt+tiny_number)
        alfvar.data.wgt[alfvar.data.wgt>huge_number] = huge_number
        # ---- fold the masked regions into the errors
        #alfvar.data.err = alfvar.data.err * alfvar.data.wgt
        #alfvar.data.err[alfvar.data.err>huge_number] = huge_number


    # ---- set initial params, step sizes, and prior ranges
    opos,prlo,prhi = set_pinit_priors(alfvar)
    # ---- convert the structures into their equivalent arrays
    prloarr = str2arr(switch=1, instr = prlo)
    prhiarr = str2arr(switch=1, instr = prhi)


    # ---- The worker's only job is to calculate the value of a function
    # ---- after receiving a parameter vector.

    # ---- this is the master process
    # ---- estimate velz ---- #
    print("  Fitting ",alfvar.nlint," wavelength intervals")
    nlint = alfvar.nlint
    l1, l2 = alfvar.l1, alfvar.l2
    print('wavelength bourdaries: ', l1, l2)
    if l2[-1]>np.nanmax(lam) or l1[0]<np.nanmin(lam):
        print('ERROR: wavelength boundaries exceed model wavelength grid')
        print(l2[nlint-1],lam[nl-1],l1[0],lam[0])

    # ---- make an initial estimate of the redshift
    print(' Fitting cz...')
    velz = getvelz(alfvar)
    if velz < prlo.velz or velz > prhi.velz:
        print('cz', velz,' out of prior bounds, setting to 0.0')
        velz = 0.0
    opos.velz = velz
    print("    cz= ",opos.velz," (z=",opos.velz/1e5,")")

    oposarr = str2arr(switch=1, instr=opos)
    global global_alfvar, global_prloarr, global_prhiarr
    global_alfvar = copy.deepcopy(alfvar)
    global_prloarr = copy.deepcopy(prloarr)
    global_prhiarr = copy.deepcopy(prhiarr)
    global global_all_prior # note it's for all parameters
    global_all_prior = [TopHat(prloarr[i], prhiarr[i]) for i in range(len(key_list))]

    npar = len(use_keys)
    print('\nWe are going to fit ', npar, 'parameters\nThey are', use_keys)


    # ---------------------------------------------------------------- #
    if run == 'emcee':
        print('Initializing emcee with nwalkers=%.0f, npar=%.0f' %(nwalkers, npar))
        tstart = time.time()
        npar = len(use_keys)
        np.random.seed(42)
        pos_emcee_in = np.empty((nwalkers, npar))

        for i, ikeys in enumerate(use_keys):
            if ikeys[:4] == 'velz' or ikeys[:4] == 'sigm':
                pos_emcee_in[:,i] = oposarr[i] + 10.0*(2.*np.random.rand(nwalkers, npar)[:,i]-1.0)
            else:
                pos_emcee_in[:,i] = oposarr[i] + 0.1*(2.*np.random.rand(nwalkers, npar)[:,i]-1.0)

        print('initial values of the first walker:\n', pos_emcee_in[0], '\n')

        #with closing(Pool(processes = ncpu)) as pool:
        with MultiPool(ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob, pool=pool)
            state = sampler.run_mcmc(pos_emcee_in, nburn + nmcmc, progress=True)

        ndur = time.time() - tstart
        print('\n Total time for emcee {:.2f}min'.format(ndur/60))
        res = sampler.get_chain(discard = nburn) # discard: int, burn-in
        prob = sampler.get_log_prob(discard = nburn)
        pickle.dump(res, open('../chain_'+outname+'.p', "wb" ) )
        pickle.dump(prob, open('../prob_'+outname+'.p', "wb" ) )

        
    # ---------------------------------------------------------------- #
    elif run == 'dynesty':
        # ---------------------------------------------------------------- #
        # based on prospector
        # ... need to learn how to optimize these parameters

        ndim = len(use_keys)

        with MultiPool(ncpu) as pool:
        #with closing(Pool(processes = ncpu)) as pool:
        #with MPIPool() as pool:
        #    if not pool.is_master():
        #        pool.wait()
        #        sys.exit(0)
        #    nprocs = pool.size
            dsampler = dynesty.DynamicNestedSampler(log_prob_nested, prior_transform, ndim,
                                                    bound='multi', nlive=int(50*ndim), sample='rwalk', 
                                                    wt_kwargs={'pfrac': 1.0}, bootstrap=0,
                                                    walks=25, pool=pool)

            # generator for initial nested sampling
            ncall = dsampler.ncall
            niter = dsampler.it - 1
            tstart = time.time()
            dsampler.run_nested(dlogz_init=0.2)
            ndur = time.time() - tstart
            print('\n Total time for dynesty {:.2f}min'.format(ndur/60))

        results = dsampler.results
        pickle.dump(results, open('../'+outname+'.p', "wb" ) )


# -------------------------------- #
# ---- command line arguments ---- #
# -------------------------------- #
argv_l = sys.argv
n_argv = len(argv_l)

ncpu = 8
tag = ''
run = 'emcee'
modelname = None

filename = argv_l[1]    
if n_argv >= 3:
    run = argv_l[2]

for il, label in enumerate(['run', 'ncpu', 'tag', 'model']):
    label_ind = [argv_l.index(s) for s in argv_l if label in str(s)]
    if len(label_ind) > 0:
        newval = argv_l[int(label_ind[0])].split(label)[1].split('=')[1]
        if il==0:
            run = str(newval)
        elif il==1:
            ncpu = int(newval)
        elif il==2:
            tag = str(newval)
        elif il==3:
            modelname = str(newval)
        
out_name = 'res_'+run+'_'+filename+'_'+tag
print('\nrunning alf:')
print('input spectrum:', filename+'.dat')
print('sampler =',run)
print('ncpu =',ncpu)
print('model_arr =',modelname)
print('output:', out_name,'\n')

alf(filename='ldss3_test1', tag='', run='dynesty',
    model_arr = '../pickle/alfvar_sspgrid_ldss3_test1.p',
    ncpu = 8, out_name=outname)    







