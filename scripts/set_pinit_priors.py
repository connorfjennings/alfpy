#from alf_vars import *
from str2arr import alfobj, str2arr
import copy, random, numpy as np

__all__ = ['set_pinit_priors']

def set_pinit_priors(alfvar, velz=None):
    """
    ---- define the first position (pos), and the lower and upper bounds 
    on the priors (prlo, prhi).  The priors are defined in such a way
    that if the user defines a prior limit that is **different from
    the default parameter set**, then that value overrides the defaults below
    ---- TYPE(PARAMS), INTENT(inout) :: pos,prlo,prhi
    ---- TYPE(PARAMS) :: test, tprlo, tprhi
    """
    
    imf_type = alfvar.imf_type
    fit_type = alfvar.fit_type
    fit_two_ages = alfvar.fit_two_ages
    npar = alfvar.npar
    
    prloarr1, prhiarr1, tprloarr1 = np.zeros((3, npar))
    testarr1, posarr1, tprhiarr1 = np.zeros((3, npar))
    
    # --------------------------------------------------------------- #
    pos, prlo, prhi =  alfobj(),  alfobj(),  alfobj()  
    
    # setup the first position
    # example: pos.logage = random.random*0.4+0.6
    pos_dict = {'logage':(0.4, 0.6), 'zh': (1.0, -1.0), 'feh': (0.4, -0.2), 
                'ah': (0.4, -0.2), 'ch': (0.4, -0.2), 
                'nh': (0.4, -0.2), 'nah': (0.4, -0.2), 'mgh': (0.4, -0.2),
                'sih': (0.4, -0.2), 'kh': (0.4, -0.2), 'cah': (0.4, -0.2),
                'tih': (0.4, -0.2), 'vh': (0.4, -0.2), 'crh': (0.4, -0.2),
                'mnh': (0.4, -0.2), 'coh': (0.4, -0.2), 'nih': (0.4, -0.2),
                'cuh': (0.4, -0.2), 'srh': (0.4, -0.2), 'bah': (0.4, -0.2),
                'euh': (0.4, -0.2), 'teff': (80., -40.), 'logfy':(2., -4.), 
                'fy_logage': (0.3, 0.), 'logm7g': (1., -4.), 'hotteff':(5., 15.), 
                'loghot':(1., -4.), 'chi2':(0, 1e33), 'sigma':(100., 100.),
                'sigma2':(100., 100.), 'velz2':(10., -5.), 'logtrans':(4., -4), 
                'logemline_h': (2., -4), 'logemline_oii': (2., -4), 'logemline_oiii': (2., -4),
                'logemline_ni': (2., -4), 'logemline_nii': (2., -4), 'logemline_sii': (2., -4),
                'jitter': (0.5, 0.75), 'logsky':(3., -6), 'h3':(0.02, -0.01), 
                 'h4':(0.02, -0.01)
               }
    
    for i, ikey in enumerate(list(pos_dict.keys())):
        tem = pos_dict[ikey]
        pos.__setattr__(ikey, random.random()*tem[0] + tem[1])
        
    if imf_type <= 3:
        pos.imf1 = random.random()*1.0-0.3 + 1.3
        pos.imf3 = random.random()*0.1 + 0.1
    else:
        pos.imf1 = random.random()*1.0 + 0.5
        pos.imf3 = random.random()*1.0   
        
    if imf_type in [0, 1, 3]:
        pos.imf2 = random.random()*1.5-0.75 + 2.0
    elif imf_type == 2:
        pos.imf2 = random.random()*0.1 + 0.1
    elif imf_type == 4:
        pos.imf2 = random.random()*0.5+0.5

    pos.imf4 = random.random()*0.5
    if imf_type == 4:
        pos.imf4 = random.random()*0.5
        pos.imf3 = pos.imf4 + random.random()*0.5
        pos.imf2 = pos.imf3 + random.random()*0.5
        pos.imf1 = pos.imf2 + random.random()*0.5        

    # ---- these pr=test statements allow the user to pre-set
    # ---- specific priors at the beginning of alf.f90; those
    # ---- choices are then not overwritten below
    # ----   priors (low)
    
    if fit_type == 0:
        if fit_two_ages == 0:
            # ---- !in this case we have a single age model, so it needs tocover the full range
            prlo.logage = np.log10(0.5)
        else:
            # ---- in this case we're fitting a two component model so dont allow them to overlap in age
            prlo.logage = np.log10(3.0)  
    else:
        # ---- in this case we have a single age model, so it needs to cover the full range
        prlo.logage = np.log10(0.5) 
            
    prhi.logage = np.log10(14.0)
            
    prior_dict = {'zh': (-1.8, 0.3), 'feh': (-0.3, 0.5), 'ah': (-0.3, 0.5), 
                  'ch': (-0.3, 0.5), 'nh': (-0.3, 1.0), 
                  'nah': (-0.3, 1.0), 'mgh': (-0.3, 0.5), 'sih': (-0.3, 0.5), 
                  'kh': (-0.3, 0.5), 'cah': (-0.3, 0.5), 'tih': (-0.3, 0.5), 
                  'vh': (-0.3, 0.5), 'crh': (-0.3, 0.5), 'mnh': (-0.3, 0.5), 
                  'coh': (-0.3, 0.5), 'nih': (-0.3, 0.5), 'cuh': (-0.3, 0.5), 
                  'srh': (-0.3, 0.5), 'bah': (-0.6, 0.5), 'euh': (-0.5, 0.5), 
                  'teff': (-50., 50.), 'logfy':(-6., -0.1), 'fy_logage': (np.log10(0.5), np.log10(3.0)), 
                  'logm7g': (-6., -1.0), 'hotteff':(8., 30.), 'loghot':(-8.0, -1.0), 
                  'sigma':(10., 1e3), 'sigma2':(10., 1e3), 'chi2':(0, 2e33),
                  'velz':(-1e3, 5e4), 'velz2':(-1e3, 5e4), 
                  'logtrans':(-6., 1.0), 'logemline_h': (-6., 1.0), 'logemline_oii': (-6., 1.0), 
                  'logemline_oiii': (-6., 1.0),'logemline_ni': (-6., 1.0), 'logemline_nii': (-6., 1.0), 
                  'logemline_sii': (-6., 1.0),
                  'jitter': (0.1, 10.0), 'logsky':(-9., 2.0), 
                  'h3':(-0.4, 0.4), 'h4':(-0.4, 0.4)
               } 
    
    
    for i, ikey in enumerate(list(prior_dict.keys())):
        tem = prior_dict[ikey]
        prlo.__setattr__(ikey, tem[0]) 
        prhi.__setattr__(ikey, tem[1]) 
                                                        
        
    if imf_type <= 3:
        prlo.imf1 = 0.5
        prlo.imf3 = 0.08
        prhi.imf1 = 3.5  
        prhi.imf3 = 0.4
            
    else:
        prlo.imf1 = -5.0
        prlo.imf3 = -5.0
        prhi.imf1 = 3.0
        prhi.imf3 = 3.0
        
    if imf_type in [0, 1, 3]:
        prlo.imf2 = 0.5
        prhi.imf2 = 3.5
            
    elif imf_type == 2:
        prlo.imf2 = -5.0
        prhi.imf2 = 0.5
            
    elif imf_type == 4:
        prlo.imf2 = -4.0
        prhi.imf2 = 3.0

    prlo.imf4 = -5.0
    prhi.imf4 = 3.0

    #--------------------------------------------------------------------------!
    #-------reset the initial parameters if the priors have been altered-------!
    #--------------------------------------------------------------------------!
    
    # ---- str->arr
    #prloarr = str2arr(switch = 1, instr = prlo)
    #prhiarr = str2arr(switch = 1, instr = prhi)
    #posarr = str2arr(switch = 1, instr = pos)

    # ---- test the priors and if the priors have been altered then 
    # ---- re-initialize the parameters within the prior range
    # ---- NB: why not simply always initialize the starting position this way?
    
    #for i in range(npar):
    #    if prhiarr1[i] <= prloarr1[i]:
    #        print('SET_PINIT_PRIORS ERROR: prhi <= prlo!', i)
    #    # ---- reset the initial parameters randomly within the prior range
    #    # ---- if both the initial priors have changed *and* the position is outside
    #    # ---- of the new priors
    #    if np.logical_or(tprhiarr1[i] != testarr1[i], tprloarr1[i] != testarr1[i]):
    #        if posarr1[i] >= prhiarr1[i] or posarr1[i] <= prloarr1[9]:
    #            posarr1[i] = random.random()*(prhiarr1[i] - prloarr1[i]) + prloarr1[i]
                
    #if velz is not None:
    #    posarr1[0] = velz + 50*(random.random()*2.-1)   
    # ---- arr -> str
    #pos = str2arr(switch=2, inarr=posarr1)
    return pos, prlo, prhi
    

