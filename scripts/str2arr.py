import numpy as np
from tofit_parameters import tofit_params

__all__ = ['str2arr', 'fill_param']

#tofit_params_keys = list(tofit_params.keys())
tofit_params_keys, tofit_params_values = zip(*[(key, v2) for key, (_, v2) in tofit_params.items()])
tofit_params_keys = list(tofit_params_keys) 
tofit_params_values = np.array(tofit_params_values) 
# ---------------------------------------------------------------- #
class alfobj(object):
    def __init__(self, inarr=None):
        if inarr is None:
            for key, value in zip(tofit_params_keys, tofit_params_values):
                self.__dict__[key] = value
        else:
            for key, value in zip(tofit_params_keys, inarr):
                self.__dict__[key] = value
            #for ikey in tofit_params_keys:
            #    self.__dict__[ikey] = tofit_params[ikey][1]
        #else:
        #    for i_, ikey_ in enumerate(tofit_params_keys):
        #        self.__dict__[ikey_] = inarr[i_]            
        
        
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
    res_ = np.copy(tofit_params_values)
    key_to_index = {key: idx for idx, key in enumerate(tofit_params_keys)}
    for i_, ikey_ in enumerate(usekeys):
        res_[key_to_index[ikey_]] = inarr[i_]
    return res_    