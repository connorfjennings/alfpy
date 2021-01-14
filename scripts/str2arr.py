from alf_vars import *
import copy, numpy as np

__all__ = ['str2arr']

def str2arr(switch, instr=None, inarr=None):

    """
    routine to dump the information in the parameteter 
    structure (str) into an array (arr), or visa versa, depending
    on the value of the switch
    
    this is actually the location where the parameters to be fit
    in the full model are specified.  If a parameter is left out
    of this list (e.g., [Rb/H]) then it does not affect Chi^2
    """
    key_list = ['velz', 'sigma', 'logage', 'zh', 
                'feh', 'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
                'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
                'teff','imf1','imf2','logfy','sigma2','velz2','logm7g','hotteff',
                'loghot','fy_logage',
                'logemline_h','logemline_oii','logemline_oiii',
                'logemline_sii','logemline_ni','logemline_nii',
                'logtrans','jitter','logsky', 'imf3','imf4','h3','h4']


    if switch == 1 and instr is not None:
        temdict = copy.deepcopy(instr.__dict__)
        arr = np.empty(len(list(temdict.keys())))
    
        for i, ikey in enumerate(key_list):
            # 0-3: the super-simple and Powell-mode parameters
            # 4=13: end of the simple model parameters
            arr[i] = getattr(instr, ikey) 
        return arr
            
            
    elif switch==2 and inarr is not None:
        key_list = ['velz', 'sigma', 'logage', 'zh', 
                    'feh', 'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
                    'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
                    'teff','imf1','imf2','logfy','sigma2','velz2','logm7g','hotteff',
                    'loghot','fy_logage',
                    'logemline_h','logemline_oii','logemline_oiii',
                    'logemline_sii','logemline_ni','logemline_nii',
                    'logtrans','jitter','logsky', 'imf3','imf4','h3','h4']
        param = ALFPARAM()
        #temdict = copy.deepcopy(param.__dict__)
        for i, ikey in enumerate(key_list):   
            param.__setattr__(ikey, inarr[i])
            
        return param