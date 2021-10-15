#from alf_vars import *
import math, numpy as np
from numba import jit

__all__ = ['str2arr', 'fill_param', 'get_default_keylist', 'get_default_arr']


# ---------------------------------------------------------------- #
@jit(nopython=True)
def get_default_arr():
    return np.array([0.0, 11.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 1.3, 2.3, -4.0, 10.1, 0.0,-5.5, 20.0, 
                     -4., 0.3,-5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
                     -5.5, 1.0, -8.5, 0.08, 0.0, 0.0, 0.0])

# ---------------------------------------------------------------- #
@jit(nopython=True)
def get_default_keylist():
    key_list = ['velz', 'sigma', 'logage', 'zh', 'feh',
            'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
            'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
            'teff','imf1','imf2','logfy','sigma2','velz2',
            'logm7g','hotteff','loghot','fy_logage',
            'logemline_h','logemline_oii','logemline_oiii',
            'logemline_sii','logemline_ni','logemline_nii',
            'logtrans','jitter','logsky', 'imf3','imf4','h3','h4']
    return key_list



# ---------------------------------------------------------------- #
class alfobj(object):
    def __init__(self):
        self.__dict__ = dict(zip(get_default_keylist(), get_default_arr()))

# ---------------------------------------------------------------- #
def str2arr(switch, instr=None, inarr=None, usekeys =None):
    """
    routine to dump the information in the parameteter
    structure (str) into an array (arr), or visa versa, depending
    on the value of the switch

    this is actually the location where the parameters to be fit
    in the full model are specified.  If a parameter is left out
    of this list (e.g., [Rb/H]) then it does not affect Chi^2
    - 1. str->arr
    - 2. arr->str
    """
    if usekeys is None:
        usekeys = get_default_keylist()
        
    if switch == 1 and instr is not None:
        res = get_default_arr()
        key_list = get_default_keylist()
        for ikey in usekeys:
            res[key_list.index(ikey)] = getattr(instr, ikey)
        return res


    elif switch==2 and inarr is not None:
        res = alfobj()
        res.__dict__ = dict(zip(get_default_keylist(), inarr))
        return res



# ---------------------------------------------------------------- #
#@jit(nopython=True)
def fill_param(inarr, usekeys):
    
    key_list = get_default_keylist()
    default_arr = get_default_arr()
    
    if len(inarr) < len(key_list):
        """
        - short inarr, output is full size
        """
        res = get_default_arr()
        for i, ikey in enumerate(usekeys):
            res[key_list.index(ikey)] = inarr[i]

    else:
        res = get_default_arr()
        for ikey in usekeys:
            res[key_list.index(ikey)] = inarr[key_list.index(ikey)]
    return res