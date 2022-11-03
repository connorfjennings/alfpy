import numpy as np
from linterp import *
from str2arr import *
from getmodel import *
from getm2l import *
from alf_constants import *

def spec_from_pos(pos, alfvar):
    """
    !takes a *sum file as input and returns the corresponding 
    !model spectrum associated with min(chi^2)
    USE alf_vars; USE alf_utils
    USE nr, ONLY : gasdev,locate,powell,ran1
    USE ran_state, ONLY : ran_seed,ran_init
    """

    # ---- !setup the models
    lam = alfvar.sspgrid.lam

    # ---- !we turn off the emission lines, since they are often highly
    # ---- !unconstrained if they are not included in the wavelength range
    pos.logemline_h    = -6.0
    pos.logemline_oii  = -6.0
    pos.logemline_oiii = -6.0
    pos.logemline_nii  = -6.0
    pos.logemline_sii  = -6.0
    pos.logemline_ni   = -6.0

    # ------------------------------------------------------------!    
    mspec = getmodel(pos, alfvar=alfvar)

    # -- redshift the spectrum to observed frame
    oneplusz = (1+pos.velz/clight*1e5)
    zmspec   = linterp(lam*oneplusz, mspec, lam)
    m2l = getm2l(lam, zmspec, pos, 
                 other_filter=['sdss_r0', 'sdss_i0', 'twomass_Ks', 'wfc3_ir_f110w']
                )
    m2lmw = getm2l(lam, zmspec, pos,  mw=1, 
                  other_filter=['sdss_r0', 'sdss_i0', 'twomass_Ks', 'wfc3_ir_f110w']
                  )
    
    return [lam, zmspec], m2l+m2lmw