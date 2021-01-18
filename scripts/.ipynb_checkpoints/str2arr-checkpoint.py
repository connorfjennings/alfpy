#from alf_vars import *
import numpy as np

__all__ = ['str2arr']


key_list = ['velz', 'sigma', 'logage', 'zh', 'feh', 
            'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
            'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
                'teff','imf1','imf2','logfy','sigma2','velz2',
                'logm7g','hotteff','loghot','fy_logage',
                'logemline_h','logemline_oii','logemline_oiii',
                'logemline_sii','logemline_ni','logemline_nii',
                'logtrans','jitter','logsky', 'imf3','imf4','h3','h4']

default_arr = np.array([0.0, 11.0, 1.0, 0.0, 0.0,0.0,0.0,0.0, 0.0, 0.0,  
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 1.3, 2.3, -4.0, 
                        11.0, 0.0, -4.0, 20.0, -4.0, 0.3, 
                        -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, 
                        -4.0, 1.0, -4.0, 0.10, 0.0, 0.0, 0.0])


class alfobj(object):
    def __init__(self):
        self.__dict__ = dict(zip(key_list, default_arr))
    
    
def str2arr(switch, instr=None, inarr=None, usekeys = key_list):
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
    
    if switch == 1 and instr is not None:
        key_arr = np.array(key_list)
        res = np.copy(default_arr)
        for i, ikey in enumerate(usekeys):
            # 0-3: the super-simple and Powell-mode parameters
            # 4=13: end of the simple model parameters
            #ind = np.where(key_list==ikey)
            res[np.where(key_arr==ikey)] = getattr(instr, ikey) 
            
            
    elif switch==2 and inarr is not None:   
        res = alfobj()    
        for i, ikey in enumerate(usekeys):   
            res.__setattr__(ikey, inarr[i])                
    return res