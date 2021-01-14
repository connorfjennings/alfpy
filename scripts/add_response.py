import alf_vars
import numpy as np

__all__ = ['add_response']
def add_response(spec, pos, range_, 
                 dr, vr, dm, vm, 
                 solar, plus, minus=None):

    """
    # perform bilinear interpolation over the age and 
    # metallicity-dependent response functions
    """
    #alfvar = ALFVAR()
    #in this case we have both + and - response functions
    #print('dr=',dr, ', vr=',vr, ',dm=',dm, ', vm=',vm)
    if minus is not None:
        if pos > 0.0:
            tmpr = (dr * dm * plus[:,vr+1,vm+1]/solar[:,vr+1,vm+1] + 
                    (1-dr)*dm* plus[:,vr,vm+1]/solar[:,vr,vm+1] + 
                    dr*(1-dm)* plus[:,vr+1,vm]/solar[:,vr+1,vm] + 
                    (1-dr)*(1-dm)*plus[:,vr,vm]/solar[:,vr,vm])
            
            spec *= 1+(tmpr-1)*pos/range_

        else:
            tmpr = (dr*dm*minus[:,vr+1,vm+1]/solar[:,vr+1,vm+1] + 
                    (1-dr)*dm*minus[:,vr,vm+1]/solar[:,vr,vm+1] + 
                    dr*(1-dm)*minus[:,vr+1,vm]/solar[:,vr+1,vm] + 
                    (1-dr)*(1-dm)*minus[:,vr,vm]/solar[:,vr,vm])
            spec *= 1+(tmpr-1)*np.abs(pos)/range_


    elif minus is None:
        tmpr = (dr*dm*plus[:,vr+1,vm+1]/solar[:,vr+1,vm+1] + 
                (1-dr)*dm*plus[:,vr,vm+1]/solar[:,vr,vm+1] + 
                dr*(1-dm)*plus[:,vr+1,vm]/solar[:,vr+1,vm] + 
                (1-dr)*(1-dm)*plus[:,vr,vm]/solar[:,vr,vm])

        spec *= 1+(tmpr-1)*pos/range_
        
    return spec
