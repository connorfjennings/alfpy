#from alf_vars import *
import math, numpy as np
from numba import jit

__all__ = ['str2arr', 'fill_param', 'get_default_keylist', 'get_default_arr', 
           'get_default_value_usekeys', 'fill_param_lnprior']


# ---------------------------------------------------------------- #
@jit(nopython=True)
def get_default_arr():
    return np.array([0.0, 11.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 1.3, 2.3, -4.0, 10.1, 0.0,-5.5, 20.0, 
                     -4., 0.3,-5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
                     -5.5, 1.0, -8.5, 0.08, 0.0, 0.0, 0.0])+1e-4

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
def get_default_value_usekeys(usekeys):
    outarr = np.empty(len(usekeys))
    fullist = get_default_keylist()
    fullarr = get_default_arr()
    for i, ikey in enumerate(usekeys):
        outarr[i] = fullarr[fullist.index(ikey)]
    return outarr


# ---------------------------------------------------------------- #
class alfobj(object):
    def __init__(self):
        self.__dict__ = dict(zip(get_default_keylist(), get_default_arr()))

# ---------------------------------------------------------------- #
def str2arr(switch, instr=None, inarr=None):
    """
    - 1. str->arr
    - 2. arr->str
    """
    
    if switch == 1 and instr is not None:
        alllist = get_default_keylist()
        usekeys = [vi for vi in instr.__dict__.keys() if vi in alllist]
        res = np.zeros(len(usekeys))
        for ikey in usekeys:
            res[usekeys.index(ikey)] = getattr(instr, ikey)
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


# ---------------------------------------------------------------- #
def fill_param_lnprior(inarr, usekeys):
    key_list = get_default_keylist()
    res = get_default_arr()
    for i in range(len(usekeys)):
        res[key_list.index(usekeys[i])] = inarr[i]
    return res




