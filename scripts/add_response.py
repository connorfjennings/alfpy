import alf_vars
from numba import jit
import math
__all__ = ['add_response']


# ---------------------------------------------------------------- #
@jit(nopython=True, fastmath=True)
def add_response(pos, range_, 
                 dr, vr, dm, vm, 
                 solar, plus, minus=None):

    """
    # perform bilinear interpolation over the age and 
    # metallicity-dependent response functions
    # updated to remove INPUT spec
    """
    # ---- in this case we have both + and - response functions
    if minus is None or pos>0.:
        tmpr = dr*dm*plus[:,vr+1,vm+1]/solar[:,vr+1,vm+1] + \
               (1-dr)*dm* plus[:,vr,vm+1]/solar[:,vr,vm+1] + \
               dr*(1-dm)* plus[:,vr+1,vm]/solar[:,vr+1,vm] + \
               (1-dr)*(1-dm)*plus[:,vr,vm]/solar[:,vr,vm]
        return 1.+(tmpr-1.)*pos/range_
            

    else:
        tmpr = dr*dm*minus[:,vr+1,vm+1]/solar[:,vr+1,vm+1] + \
               (1-dr)*dm*minus[:,vr,vm+1]/solar[:,vr,vm+1] + \
               dr*(1-dm)*minus[:,vr+1,vm]/solar[:,vr+1,vm] + \
               (1-dr)*(1-dm)*minus[:,vr,vm]/solar[:,vr,vm]
                    
        return 1+(tmpr-1)*math.fabs(pos)/range_

        

        
# ---------------------------------------------------------------- #
@jit(nopython=True, fastmath=True)
def add_na_03(pos, base_, dr, vr, dm, vm, 
                solar, plus, plus2):
    
    tmpr =(dr*dm*plus[:,vr+1,vm+1]/solar[:,vr+1,vm+1] + \
           (1-dr)*dm*plus[:,vr,vm+1]/solar[:,vr,vm+1] + \
           dr*(1-dm)*plus[:,vr+1,vm]/solar[:,vr+1,vm] + \
           (1-dr)*(1-dm)*plus[:,vr,vm]/solar[:,vr,vm])

    tmp = (dr*dm*(plus2[:,vr+1,vm+1]-plus[:,vr+1,vm+1])/solar[:,vr+1,vm+1]+ \
           (1-dr)*dm*(plus2[:,vr,vm+1]-plus[:,vr,vm+1])/solar[:,vr,vm+1]+ \
           dr*(1-dm)*(plus2[:,vr+1,vm]-plus[:,vr+1,vm])/solar[:,vr+1,vm]+ \
           (1-dr)*(1-dm)*(plus2[:,vr,vm]-plus[:,vr,vm])/solar[:,vr,vm])
    
    return tmpr+tmp*(pos-base_)/base_