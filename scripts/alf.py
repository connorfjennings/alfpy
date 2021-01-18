import os, copy, pickle, numpy as np
from func import func
from alf_vars import *
from set_pinit_priors import *
from alf_constants import *
import emcee, time
from multiprocessing import Pool
import dynesty
from dynesty import NestedSampler, DynamicNestedSampler
from priors import TopHat, lnprior
from read_data import *
from linterp import *
from str2arr import *
from getvelz import getvelz
#import h5py


def alf(filename, alfvar=None, tag='', run='emcee'):
    """
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
        alfvar = pickle.load(open('../../alfvar_sspgrid_irldss3_imftype3.p', "rb" )) 
        
    nmcmc = 100    # -- number of chain steps to print to file
    # -- inverse sampling of the walkers for printing
    # -- NB: setting this to >1 currently results in errors in the *sum outputs
    nsample = 1
    nburn = 100    # -- length of chain burn-in
    nwalkers = 128    # -- number of walkers
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
    mpiposarr = np.zeros((npar,nwalkers))
    runtot = np.zeros((3,npar+2*nfil))
    cl2p5,cl16,cl50,cl84,cl97p5 = np.zeros((5, npar+2*nfil))
    #mcmcpar = np.zeros((npar+2*nfil, nwalkers*nmcmc/nsample))

    #---------------------------------------------------------------!
    #---------------------------Setup-------------------------------!
    #---------------------------------------------------------------!
    # ---- flag specifying if fitting indices or spectra
    alfvar.fit_indices = 0  #flag specifying if fitting indices or spectra

    # ---- flag determining the level of complexity
    # ---- 0=full, 1=simple, 2=super-simple.  See sfvars for details
    alfvar.fit_type = 0

    # ---- fit h3 and h4 parameters
    alfvar.fit_hermite = 0

    # ---- type of IMF to fit
    # ---- 0=1PL, 1=2PL, 2=1PL+cutoff, 3=2PL+cutoff, 4=non-parametric IMF
    alfvar.imf_type = 1

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
    if (alfvar.observed_frame == 0.) or (alfvar.fit_indices == 1):
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
        #alfvar = setup(alfvar)
        lam = np.copy(alfvar.sspgrid.lam)

        # ---- interpolate the sky emission model onto the observed wavelength grid
        if alfvar.observed_frame == 1:
            alfvar.data.sky = linterp(alfvar.lsky, alfvar.fsky, alfvar.data.lam)
            alfvar.data.sky[alfvar.data.sky<0] = 0.                               
        else:
            alfvar.data.sky[:] = tiny_number
        alfvar.data.sky[:] = tiny_number

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


    # ---- set initial params, step sizes, and prior ranges
    opos,prlo,prhi = set_pinit_priors(alfvar)
    # ---- convert the structures into their equivalent arrays
    prloarr = str2arr(switch=1, instr = prlo)
    prhiarr = str2arr(switch=1, instr = prhi)
    

    # ---- The worker's only job is to calculate the value of a function
    # ---- after receiving a parameter vector.

    # ---- this is the master process
    # ---- estimate velz ---- #
    if (alfvar.fit_indices == 0):
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
        
        global use_keys
        use_keys = ['velz', 'sigma', 'logage', 'zh',]
        print('\nWe are going to fit ', use_keys, '\n')
        npar = len(use_keys)

        #def log_prob(posarr):
        #    return -0.5*func(global_alfvar, posarr, 
        #                     prhiarr=global_prhiarr, 
        #                     prloarr=global_prloarr, 
        #                     usekeys=use_keys)
        def log_prob(posarr):
            ln_prior = lnprior(posarr, usekeys = use_keys, 
                               prhiarr=global_prhiarr, prloarr=global_prloarr, 
                               nested = False)
            if not np.isfinite(ln_prior):
                return -np.infty
    
            return ln_prior - 0.5*func(global_alfvar, posarr, 
                             prhiarr=global_prhiarr, 
                             prloarr=global_prloarr, 
                             usekeys=use_keys)

           
        print('Initializing emcee with nwalkers=%.0f, npar=%.0f' %(nwalkers, npar))
        
        nwalkers = 128
        pos_emcee_in = np.empty((nwalkers, npar))
        pos_emcee_in[:,:2] = oposarr[:2] + 10.0*(2.*np.random.rand(nwalkers, npar)[:,:2]-1.0)
        pos_emcee_in[:,2:npar] = oposarr[2:npar] +  0.1*(2.*np.random.rand(nwalkers, npar)[:,2:npar]-1.0)
        
        #backend = emcee.backends.HDFBackend("../test.h5")
        #backend.reset(nwalkers, npar)

        if run == 'emcee':
            nmcmc = 1000
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob, pool=pool)
                start = time.time()
                state = sampler.run_mcmc(pos_emcee_in, nmcmc, progress=True)
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))  
            
            samples = sampler.get_chain()
            pickle.dump(samples, open('../test_samples.p', "wb" ) )
            
        elif run == 'dynesty':
            def prior_transform(unit_coords, usekeys = use_keys,
                    prhiarr=global_prhiarr, prloarr=global_prloarr):    
                theta = np.empty((len(unit_coords)))
                key_arr = np.array(['velz', 'sigma', 'logage', 'zh', 'feh', 
                        'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
                        'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
                            'teff','imf1','imf2','logfy','sigma2','velz2',
                            'logm7g','hotteff','loghot','fy_logage',
                            'logemline_h','logemline_oii','logemline_oiii',
                            'logemline_sii','logemline_ni','logemline_nii',
                            'logtrans','jitter','logsky', 'imf3','imf4','h3','h4'])
    
                for i, ikey in enumerate(usekeys):
                    ind = np.where(key_arr==ikey)
                    a = TopHat(prloarr[ind].item(), prhiarr[ind].item())
                    theta[i] = a.unit_transform(unit_coords[i])
                return theta    
            
            dsampler = dynesty.DynamicNestedSampler(log_prob, prior_transform, 
                                        len(use_keys), nlive=1000)
            dsampler.run_nested(maxiter = 1000)
            res1 = dsampler.results
 