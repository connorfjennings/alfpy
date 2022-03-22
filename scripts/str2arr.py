#from alf_vars import *
import math, numpy as np
from numba import jit
from tofit_parameters import tofit_params

__all__ = ['str2arr', 'fill_param']
tofit_params_keys = list(tofit_params.keys())


# ---------------------------------------------------------------- #
class alfobj(object):
    def __init__(self, inarr=None):
        if inarr is None:
            for ikey in tofit_params_keys:
                self.__dict__[ikey] = tofit_params[ikey].default_val
        else:
            for i_, ikey_ in enumerate(tofit_params_keys):
                self.__dict__[ikey_] = inarr[i_]            
        
# ---------------------------------------------------------------- #
def str2arr(switch, instr=None, inarr=None):
    """
    - 1. str->arr
    - 2. arr->str, keys in in arr has to have the same order
    """ 
    if switch == 1 and instr is not None:
        return np.array(list(instr.__dict__.values()))

    elif switch==2 and inarr is not None:
        return alfobj(inarr)


# ---------------------------------------------------------------- #
def fill_param(inarr, usekeys):
    """
    fill the parameters not included in fitting with default values
    """
    res_ = np.array([tofit_params[ikey].default_val for ikey in tofit_params_keys])
    for i_, ikey_ in enumerate(usekeys):
        res_[tofit_params_keys.index(ikey_)] = inarr[i_]  
    return res_       




