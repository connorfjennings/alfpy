import os, numpy as np
from datetime import datetime
import alf_vars import *
from set_pinit_priors import *


def alf(alfvar, filename = None):
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

    nmcmc = 100    #number of chain steps to print to file
    #inverse sampling of the walkers for printing
    #NB: setting this to >1 currently results in errors in the *sum outputs
    nsample = 1
    nburn = 10000    #length of chain burn-in
    walkers=512    #number of walkers
    print_mcmc=1; print_mcmc_spec=0    #save the chain outputs to file and the model spectra

    dopowell = 0  #start w/ powell minimization?
    ftol = 0.1    #Powell iteration tolerance
    #if set, will print to screen timing of likelihood calls
    test_time = 0
    #number of Monte Carlo realizations of the noise for index errors
    nmcindx = 1000

    # check
    tiny_number, huge_number = 1e-33, 1e33
    totacc = 0; iter_ = 30
    minchi2 = huge_number
    bret = huge_number
    
    nl = alfvar.nl
    npar = alfvar.npar
    nfil = alfvar.nfil

    mspec, mspecmw, lam = numpy.zeros((3, nl))
    m2l, m2lmw = numpy.zeros((2, nfil))
    oposarr, bposarr = numpy.zeros((2, npar))
    mpiposarr = numpy.zeros((npar,nwalkers))
    runtot = numpy.zeros((3,npar+2*nfil))
    cl2p5,cl16,cl50,cl84,cl97p5 = numpy.zeros((5, npar+2*nfil))
    xi = numpy.zeros((npar, npar))
    mcmcpar = numpy.zeros((npar+2*nfil, nwalkers*nmcmc/nsample))

    sortpos = numpy.zeros(nwalkers*nmcmc/nsample)

    dumt = numpy.empty(2)
    file=''; tag=''
    bpos,tpos = PARAMS(),PARAMS(),

    #REAL(DP)      :: sigma_indx,velz_indx
    gdev, tflx = numpy.empty((2, ndat))
    tmpindx = numpy.zeros((nmcindx,nindx))
    #mspec_mcmc = numpy.zeros((nmcmc*nwalkers/nsample+1,nl))

    #variables for emcee
    pos_emcee_in, pos_emcee_out = numpy.empty((2, npar, nwalkers))
    lp_emcee_in, lp_emcee_out, lp_mpi = numpy.empty((3, nwalkers))
    accept_emcee = numpy.zeros(nwalkers, dtype='i4')

    #variables for MPI
    #INTEGER :: ierr,taskid,ntasks,received_tag,status(MPI_STATUS_SIZE)
    KILL=99; BEGIN=0
    wait = True
    masterid=0

    #---------------------------------------------------------------!
    #---------------------------Setup-------------------------------!
    #---------------------------------------------------------------!
    #flag specifying if fitting indices or spectra
    alfvar.fit_indices = 0  #flag specifying if fitting indices or spectra

    #flag determining the level of complexity
    #0=full, 1=simple, 2=super-simple.  See sfvars for details
    alfvar.fit_type = 1

    #fit h3 and h4 parameters
    alfvar.fit_hermite = 0

    #type of IMF to fit
    #0=1PL, 1=2PL, 2=1PL+cutoff, 3=2PL+cutoff, 4=non-parametric IMF
    alfvar.imf_type = 1

    #are the data in the original observed frame?
    alfvar.observed_frame = 1

    alfvar.mwimf = 0  #force a MW (Kroupa) IMF

    #fit two-age SFH or not?  (only considered if fit_type=0)
    alfvar.fit_two_ages = 1

    #IMF slope within the non-parametric IMF bins
    #0 = flat, 1 = Kroupa, 2 = Salpeter
    alfvar.nonpimf_alpha = 2

    #turn on/off the use of an external tabulated M/L prior
    alfvar.extmlpr = 0

    #change the prior limits to kill off these parameters
    pos, prlo, prhi = set_pinit_priors(alfvar)
    prhi.logm7g = -5.0
    prhi.teff   =  2.0
    prlo.teff   = -2.0


    #---------------------------------------------------------------!
    #--------------Do not change things below this line-------------!
    #---------------unless you know what you are doing--------------!
    #---------------------------------------------------------------!

    #regularize non-parametric IMF (always do this)
    alfvar.nonpimf_regularize = 1

    #dont fit transmission function in cases where the input
    #spectrum has already been de-redshifted to ~0.0
    if (alfvar.observed_frame == 0.) or (alfvar.fit_indices == 1):
        alfvar.fit_trans = 0
        prhi.logtrans = -5.0
        prhi.logsky   = -5.0
    else
        alfvar.fit_trans = 1
        #extra smoothing to the transmission spectrum.
        #if the input data has been smoothed by a gaussian
        #in velocity space, set the parameter below to that extra smoothing
        alfvar.smooth_trans = 0.0

    if (alfvar.ssp_type == 'cvd'):
        #always limit the [Z/H] range for CvD since
        #these models are actually only at Zsol
        prhi.zh =  0.01
        prlo.zh = -0.01
        if (alfvar.imf_type > 1):
            print('ALF ERROR, ssp_type=cvd but imf>1')

    if alfvar.fit_type in [1,2]:
        alfvar.mwimf=1

    #---------------------------------------------------------------!
    # CHECK
    # Initialize MPI, and get the total number of processes and
    # your process number
    CALL MPI_INIT( ierr )
    CALL MPI_COMM_RANK( MPI_COMM_WORLD, taskid, ierr )
    CALL MPI_COMM_SIZE( MPI_COMM_WORLD, ntasks, ierr )

    #initialize the random number generator
    #set each task to sleep for a different length of time
    #so that each task has its own unique random number seed
    CALL SLEEP(taskid)
    CALL INIT_RANDOM_SEED()
    # CHECK
    #---------------------------------------------------------------!

    
    if filename is None:
        print('ALF ERROR: You need to specify an input file')
        teminput = input("Name of the input file: ")
        if len(teminput.split(' '))==1:
            filename = teminput
        elif len(teminput.split(' '))>1:
            filename = teminput[0]
            tag = teminput[1]


    if (taskid == masterid):
        #write some important variables to screen
        print(" ************************************")
        if alfvar.fit_indices == 1:
            print(" ***********Index Fitter*************")
        else:
            print(" **********Spectral Fitter***********")

        print(" ************************************")
        print("   ssp_type  =", alfvar.ssp_type)
        print("   fit_type  =", alfvar.fit_type)
        print("   imf_type  =", alfvar.imf_type)
        print(" fit_hermite =", fit_hermite)
        print("fit_two_ages =", fit_two_ages)
        if alfvar.imf_type == 4:
            print("   nonpimf   =", nonpimf_alpha)
        print("  obs_frame  =",  observed_frame)
        print("      mwimf  =",  mwimf)
        print("  age-dep Rf =",  use_age_dep_resp_fcns)
        print("    Z-dep Rf =",  use_z_dep_resp_fcns)
        print("  Nwalkers   = ",  nwalkers)
        print("  Nburn      = ",  nburn)
        print("  Nchain     = ",  nmcmc)
        print("  Ncores     = ",  ntasks)
        print("  filename   = ",  filename, ' ', tag)
        print(" ************************************")
        print('\n\nStart Time ',datetime.now())


    #---------------------------------------------------------------!
    #read in the data and wavelength boundaries
    alf.filename = filename
    alf.tag = tag    


    #if alfvar.fit_indices == 1:
    #    alfvar, sigma_indx, velz_indx = read_data(alfvar, sigma_indx, velz_indx)
    #    #fold in the approx data sigma into the "instrumental"
    #    alfvar.data.ires = np.sqrt(data.ires**2 + sigma_indx**2)
    #    #read in the SSPs and bandpass filters
    #    alfvar = setup(alfvar)
    #    lam = sspgrid.lam
    #    prhi.logemline_h    = -5.0
    #    prhi.logemline_oii  = -5.0
    #    prhi.logemline_oiii = -5.0
    #    prhi.logemline_nii  = -5.0
    #    prhi.logemline_sii  = -5.0
    #    prhi.logemline_ni   = -5.0
    #    prhi.loghot         = -5.0
    #    prhi.logm7g         = -5.0
    #    prhi.teff           =  2.0
    #    prlo.teff           = -2.0
    #    #we dont use velocities or dispersions here, so this
    #    #should be unnecessary, but haven't tested turning them off yet.
    #    prlo.velz           = -10.
    #    prhi.velz           =  10.
    #    prlo.sigma          = sigma_indx-10.
    #    prhi.sigma          = sigma_indx+10.#

    #    #de-redshift, monte carlo sample the noise, and compute indices
    #    #NB: need to mask bad pixels!
    #    for j in range()
    #    DO j=1,nmcindx
    #       CALL GASDEV(gdev(1:datmax))
    #       tflx(1:datmax) = linterp(data(1:datmax)%lam/(1+velz_indx),&
    #            data(1:datmax)%flx+gdev(1:datmax)*data(1:datmax)%err,&
    #            data(1:datmax)%lam)
    #       CALL GETINDX(data(1:datmax)%lam,tflx(1:datmax),tmpindx(j,:))
    #     ENDDO

    #    #compute mean indices and errors
    #    DO j=1,nindx
    #       IF (indx2fit(j).EQ.1) THEN
    #          data_indx(j)%indx = SUM(tmpindx(:,j))/nmcindx
    #          data_indx(j)%err  = SQRT( SUM(tmpindx(:,j)**2)/nmcindx - &
    #               (SUM(tmpindx(:,j))/nmcindx)**2 )
    #          !write(*,'(I2,2F6.2)') j,data_indx(j)%indx,data_indx(j)%err
    #       ELSE
    #          data_indx(j)%indx = 0.0
    #          data_indx(j)%err  = 999.
    #       ENDIF
    #    ENDDO
    #    nl_fit = nl


    if fit_indices == 0:
        alfvar = read_data(alfvar)
        #read in the SSPs and bandpass filters
        alfvar = setup(alfvar)
        lam = np.copy(alfvar.sspgrid.lam)

        #interpolate the sky emission model onto the observed wavelength grid
        if observed_frame == 1:
            alfvar.data.sky = linterp(alfvar.lsky, alfvar.fsky, alfvar.data.lam)
            alfvar.data.sky[alfvar.data.sky<0] = 0.
                                            
        else:
            alfvar.data.sky[:] = tiny_number
        alfvar.data.sky[:] = tiny_number

        #we only compute things up to 500A beyond the input fit region
        alfvar.nl_fit = min(max(locate(lam, 
                                       alfvar.l2[alfvar.nlint-1]+500.0),
                                1),alfvar.nl)

        #define the log wavelength grid used in velbroad.f90
        dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-
                  np.log(alfvar.sspgrid.lam[0]))/alfvar.nl_fit
        
        for i in range(alfvar.nl_fit):
            alfvar.lnlam[i] = i*dlstep + np.log(alfvar.sspgrid.lam[0])

        #masked regions have wgt=0.0.  We'll use wgt as a pseudo-error
        #array in contnormspec, so turn these into large numbers
        alfvar.data.wgt = 1./(alfvar.data.wgt+tiny_number)
        alfvar.data.wgt[alfvar.data.wgt>huge_number] = huge_number
        #fold the masked regions into the errors
        alfvar.data.err = alfvar.data.err * alfvar.data.wgt
        alfvar.data.err[alfvar.data.err>huge_number] = huge_number


    #set initial params, step sizes, and prior ranges
    opos,prlo,prhi = set_pinit_priors(alfvar)
    #convert the structures into their equivalent arrays
    prloarr = str2arr(switch=1, instr = prlo)
    prhiarr = str2arr(switch=1, instr = prhi)
    

    # The worker's only job is to calculate the value of a function
    # after receiving a parameter vector.
    if taskid != masterid:
        # Start event loop
        while wait:
           # Look for data from the master. This call can accept up
           # to ``nwalkers`` paramater positions, but it expects
           # that the actual number of positions is smaller and is
           # given by the MPI_TAG.  This call does not return until
           # a set of parameter vectors is received
            CALL MPI_RECV(npos, 1, MPI_INTEGER, &
                 masterid, MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)
            received_tag = status(MPI_TAG)
            IF ((received_tag.EQ.KILL).OR.(npos.EQ.0)) EXIT
            CALL MPI_RECV(mpiposarr(1,1), npos*npar, MPI_DOUBLE_PRECISION, &
                 masterid, MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)

            if (taskid==1 and test_time==1):
               CALL DATE_AND_TIME(TIME=time)
               WRITE(*,*) '1 Time '//time(1:2)//':'//time(3:4)//':'&
                    //time(5:9),npos,taskid
            ENDIF

            #Calculate the probability for these parameter positions
            for k range(npos)
               lp_mpi(k) = -0.5*func(mpiposarr(:,k))
            ENDDO

            if (taskid == 1 and test_time==1) THEN
              CALL DATE_AND_TIME(TIME=time)
              WRITE(*,*) '2 Time '//time(1:2)//':'//time(3:4)//':'&
                   //time(5:9),npos,taskid


            #Send it back to the master
            CALL MPI_SEND(lp_mpi(1), npos, MPI_DOUBLE_PRECISION, &
                 masterid, BEGIN, MPI_COMM_WORLD, ierr)

        ENDDO


    #this is the master process
    if taskid == masterid:
        #for testing
        if (1 == 0):
            tpos.logage = 1.143
            tpos.imf1   = 3.32
            tpos.imf2   = 2.76
            tpos.imf3   = 0.08
            #CALL GETMODEL(tpos,mspecmw,mw=1)     #get spectrum for MW IMF
            #CALL GETM2L(lam,mspecmw,tpos,m2lmw,mw=1) #compute M/L_MW
            #write(*,'(A10,2F7.2)') 'M/L(MW)=', m2lmw(1:2)
            #CALL GETMODEL(tpos,mspec)
            #CALL GETM2L(lam,mspec,tpos,m2l)
            #write(*,'(A10,2F7.2)') 'M/L=', m2l(1:2)
            #CALL FREE_WORKERS(ntasks-1)
            #CALL MPI_FINALIZE(ierr)


        if (fit_indices == 0):
            print("  Fitting ",nlint," wavelength intervals")
            if l2[nlint-1]>lam[nl-1] or l1[0]>lam[0]:
                print('ERROR: wavelength boundaries exceed model wavelength grid')
                print(l2[nlint-1],lam[nl-1],l1[0],lam[0])

            #make an initial estimate of the redshift
            if (filename[:4] == 'cdfs' or filename[:5] == 'legac'):
                print('Setting initial cz to 0.0')
                velz = 0.0
            else
                print(' Fitting cz...')
                velz = getvelz()
                if velz > prlo.velz or velz > prhi.velz:
                    print('cz out of prior bounds, setting to 0.0')
                    velz = 0.0

            opos.velz = velz
            print("    cz= ",opos.velz," (z=",opos.velz/e5,")")


        oposarr = str2arr(switch=1, instr=opos)
        #initialize the random number generator
        #why is this being done here again?
        CALL INIT_RANDOM_SEED()

        #---------------------------------------------------------------!
        #---------------------Powell minimization-----------------------!
        #---------------------------------------------------------------!

        if (dopowell == 1):
            print(' Running Powell...')
            alfvarl.powell_fitting = 1
            for j in range(10):
                xi=0.0
                for i in range(npar):
                    xi[i,i] = 1e-2

                fret = huge_number
                opos,prlo,prh = set_pinit_priors(velz=velz)
                #CALL SET_PINIT_PRIORS(opos,prlo,prhi,velz=velz)
                oposarr = str2arr(switch=1, instr=opos)
                #CALL STR2ARR(1,opos,oposarr) !str->arr
                CALL POWELL(oposarr(1:npowell),xi(1:npowell,1:npowell),ftol,iter,fret)
                opos = str2arr(switch=2, inarr = oposarr)
                if fret > bret:
                    bposarr = oposarr
                    bpos    = opos
                    bret    = fret

            alfvar.powell_fitting = 0
            #use the best-fit Powell position for the first MCMC position
            opos = str2arr(switch=2, inarr=bposarr)
            print("    best velocity: ",opos.velz) 
            print("    best sigma:    ",opos.sigma) 
            print("    best age:      ",10**opos.logage)
            print("    best [Z/H]:    ",opos.zh)


        if maskem == 1:
           #now that we have a good guess of the redshift and velocity dispersion,
           #mask out regions where emission line contamination may be a problem
           #In full mode, the default is to actually *fit* for emissions lines.
           CALL MASKEMLINES(opos%velz,opos%sigma)

        #---------------------------------------------------------------!
        #-----------------------------MCMC------------------------------!
        #---------------------------------------------------------------!
        print(' Running emcee...')
        CALL FLUSH()

        #initialize the walkers
        for j in range(nwalkers):
            opos,prlo,prhi = set_pinit_priors(velz=velz)
            pos_emcee_in[:,j] = str2arr(switch=1, instr=opos)
            if dopowell == 1:
                #use the best-fit position from Powell, with small
                #random offsets to set up all the walkers, but only
                #do this for the params actually fit in Powell!
                #the first two params are velz and sigma so give them
                #larger variation.
                for i in range(npowell):
                    if i<=2:  wdth = 10.0
                    if i>2: wdth = 0.1
                    pos_emcee_in(i,j) = bposarr(i) + wdth*(2.*myran()-1.0)
                    if pos_emcee_in[i,j] <= prloarr[i]:
                        pos_emcee_in[i,j]=prloarr[i]+wdth
                    if pos_emcee_in[i,j] >= prhiarr[i]:
                        pos_emcee_in[i,j]=prhiarr[i]-wdth
                        

            #Compute the initial log-probability for each walker
            lp_emcee_in[j] = -0.5*func(pos_emcee_in[:, j])

            #check for initialization errors
            if -2.*lp_emcee_in[j] >= huge_number/2.:
                print('ALF ERROR: initial lnp out of bounds!', j)
                for i in range(npar):
                    if pos_emcee_in[i,j] > prhiarr[i] or pos_emcee_in[i,j] < prloarr[i]:
                        print(i, pos_emcee_in(i,j), prloarr(i), prhiarr(i))


        #burn-in
        print('   burning in...')
        print('      Progress:')
        for i in range(nburn):
            CALL EMCEE_ADVANCE_MPI(npar,nwalkers,2.d0,pos_emcee_in,&
             lp_emcee_in,pos_emcee_out,lp_emcee_out,accept_emcee,ntasks-1)
            pos_emcee_in = pos_emcee_out
            lp_emcee_in  = lp_emcee_out
            if i == nburn/4.*1:
                print(' ...25%')
                CALL FLUSH()
            if i == nburn/4.*2:
                print( '...50%')
                CALL FLUSH()
            if i == nburn/4.*3:
                print( '...75%')
                CALL FLUSH()

        print('...100%')
        CALL FLUSH()

        #Run a production chain
        print('   production run...')

        if (print_mcmc == 1):
            #open output file
            OPEN(12,FILE=TRIM(ALF_HOME)//TRIM(OUTDIR)//&
                 TRIM(file)//TRIM(tag)//'.mcmc',STATUS='REPLACE')

        for i in range(nmcmc):
            CALL EMCEE_ADVANCE_MPI(npar,nwalkers,2.d0,pos_emcee_in,&
             lp_emcee_in,pos_emcee_out,lp_emcee_out,accept_emcee,ntasks-1)
            pos_emcee_in = pos_emcee_out
            lp_emcee_in  = lp_emcee_out
            totacc       = totacc + SUM(accept_emcee)
            
            
            for j in range(0, nwalkers, nsample):
                opos = str2arr(switch=2, inarr=pos_emcee_in[:,j])


                #turn off various parameters for computing M/L
                opos.logemline_h    = -8.0
                opos.logemline_oii  = -8.0
                opos.logemline_oiii = -8.0
                opos.logemline_nii  = -8.0
                opos.logemline_sii  = -8.0
                opos.logemline_ni   = -8.0
                opos.logtrans       = -8.0

                #compute the main sequence turn-off mass vs. t and Z
                CALL GETMODEL(opos, mspecmw, mw=1)       #get spectrum for MW IMF
                CALL GETM2L(lam, mspecmw, opos, m2lmw, mw=1)   #compute M/L_MW

                if alfvar.mwimf ==0:
                    CALL GETMODEL(opos,mspec)
                    CALL GETM2L(lam,mspec,opos,m2l) # compute M/L
                else:
                    m2l   = m2lmw
                    mspec = mspecmw


                #save each model spectrum
                #mspec_mcmc(1+j+(i-1)*nwalkers/nsample,:) = mspec

                #these parameters aren't actually being updated
                if (fit_indices==1):
                    pos_emcee_in[0,j] = 0.0
                    pos_emcee_in[1,j] = sigma_indx

                if (fit_type==1):
                    pos_emcee_in[nparsimp:,j] = 0.0
                elif fit_type ==2:
                    pos_emcee_in[npowell:,j] = 0.

                    
                if (print_mcmc==1):
                #!write the chain element to file
                    WRITE(12,'(ES12.5,1x,99(F11.4,1x))') &
                       -2.0*lp_emcee_in(j),pos_emcee_in(:,j),m2l,m2lmw

                #keep the model with the lowest chi2
                if (-2.0*lp_emcee_in[j]<minchi2):
                    bposarr = pos_emcee_in[:,j]
                    minchi2 = -2.0*lp_emcee_in[j]


                CALL UPDATE_RUNTOT(runtot,pos_emcee_in(:,j),m2l,m2lmw)

                #save each chain element
                mcmcpar(1:npar,j+(i-1)*nwalkers/nsample) = pos_emcee_in(:,j)
                mcmcpar(npar+1:npar+nfil,j+(i-1)*nwalkers/nsample)        = m2l
                mcmcpar(npar+nfil+1:npar+2*nfil,j+(i-1)*nwalkers/nsample) = m2lmw



        IF (print_mcmc==1) CLOSE(12)

        #save the best position to the structure
        bpos = str2arr(switch=2, inarr=bposarr)
        bpos.chi2 = minchi2

        #compute CLs
        for i in range(npar+2*nfil):
            sortpos = mcmcpar[i,:]
            CALL SORT(sortpos)
            cl2p5(i)  = sortpos(INT(0.025*nwalkers*nmcmc/nsample))
            cl16(i)   = sortpos(INT(0.160*nwalkers*nmcmc/nsample))
            cl50(i)   = sortpos(INT(0.500*nwalkers*nmcmc/nsample))
            cl84(i)   = sortpos(INT(0.840*nwalkers*nmcmc/nsample))
            cl97p5(i) = sortpos(INT(0.975*nwalkers*nmcmc/nsample))


        CALL DATE_AND_TIME(TIME=time)
        CALL DTIME(dumt,time2)
        print( 'End Time   '//time(1:2)//':'//time(3:4))
        print(" Elapsed Time: ",time2/3600.," hr")
        print("  facc: ",REAL(totacc)/REAL(nmcmc*nwalkers))


        #---------------------------------------------------------------!
        #--------------------Write results to file----------------------!
        #---------------------------------------------------------------!

       #write a binary file of the production chain spectra
       #if (print_mcmc_spec==1):
      #  mspec_mcmc(1,:) = lam
      #  OPEN(11,FILE=TRIM(ALF_HOME)//TRIM(OUTDIR)//&
      #       TRIM(file)//TRIM(tag)//'.spec',FORM='UNFORMATTED',&
      #       STATUS='REPLACE',access='DIRECT',&
      #       recl=(1+nmcmc*nwalkers/nsample)*nl*4)
      #  WRITE(11,rec=1) mspec_mcmc
      #  CLOSE(11)
    
    bposarr = str2arr(switch =1, instr=bpos)
    fret = func(bposarr, spec=mspefc, alfvar=alfvar, funit=13)
    f13name = '{0}results/{1}_{2}.bestspec'.format(ALF_HOME, file, tag)
    np.savetxt(alfname, np.transpose([lam, zmspec]), 
               delimiter="     ", 
               fmt='   %12.4f   %12.4E')

     OPEN(13,FILE=TRIM(ALF_HOME)//TRIM(OUTDIR)//&
          TRIM(file)//TRIM(tag)//'.bestspec',STATUS='REPLACE')
     CALL STR2ARR(1,bpos,bposarr)
     #NB: the model written to file has the lowest chi^2
     fret = func(bposarr,spec=mspec,funit=13)
     CLOSE(13)

     #write mean of the posterior distributions
     OPEN(14,FILE=TRIM(ALF_HOME)//TRIM(OUTDIR)//&
          TRIM(file)//TRIM(tag)//'.sum',STATUS='REPLACE')
     WRITE(14,'("#   Elapsed Time: ",F6.2," hr")') time2/3600.
     WRITE(14,'("#    ssp_type  =",A4)') alfvar.ssp_type
     WRITE(14,'("#    fit_type  =",I2)') alfvar.fit_type
     WRITE(14,'("#    imf_type  =",I2)') alfvar.imf_type
     WRITE(14,'("#  fit_hermite =",I2)') fit_hermite
     WRITE(14,'("# fit_two_ages =",I2)') fit_two_ages
     WRITE(14,'("#     nonpimf  =",I2)') nonpimf_alpha
     WRITE(14,'("#   obs_frame  =",I2)') observed_frame
     WRITE(14,'("#    fit_poly  =",I2)') fit_poly
     WRITE(14,'("#       mwimf  =",I2)') mwimf
     WRITE(14,'("#   age-dep Rf =",I2)') use_age_dep_resp_fcns
     WRITE(14,'("#     Z-dep Rf =",I2)') use_z_dep_resp_fcns
     WRITE(14,'("#   Nwalkers   = ",I6)') nwalkers
     WRITE(14,'("#   Nburn      = ",I6)') nburn
     WRITE(14,'("#   Nchain     = ",I6)') nmcmc
     WRITE(14,'("#   Nsample    = ",I6)') nsample
     WRITE(14,'("#   Nwave      = ",I6)') nl
     WRITE(14,'("#   Ncores     = ",I6)') ntasks
     WRITE(14,'("#   facc: ",F6.3)') REAL(totacc)/REAL(nmcmc*nwalkers)
     WRITE(14,'("#   rows: mean posterior, pos(chi^2_min), 1 sigma errors, '//&
          '2.5%, 16%, 50%, 84%, 97.5% CL, lower priors, upper priors ")')

     #write mean of posteriors
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') bpos%chi2,runtot(2,:)/runtot(1,:)

     #write position where chi^2=min
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') bpos%chi2,bposarr,m2l*0.0,m2lmw*0.0

     #write 1 sigma errors
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0, &
          SQRT( runtot(3,:)/runtot(1,:) - runtot(2,:)**2/runtot(1,:)**2 )

     #write 2.5%, 16%, 50%, 84%, 97.5% CL
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0, cl2p5
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0, cl16
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0, cl50
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0, cl84
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0, cl97p5
     #write lower/upper priors
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0,prloarr,m2l*0.0,m2lmw*0.0
     WRITE(14,'(ES12.5,1x,99(F11.4,1x))') 0.0,prhiarr,m2l*0.0,m2lmw*0.0

     CLOSE(14)

     WRITE(*,*)
     WRITE(*,'(" ************************************")')

     #break the workers out of their event loops so they can close
     CALL FREE_WORKERS(ntasks-1)

  ENDIF

  CALL MPI_FINALIZE(ierr)
