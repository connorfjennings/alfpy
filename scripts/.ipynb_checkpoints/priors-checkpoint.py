# goal: use dynesty

import numpy as np
import scipy.stats 
from str2arr import str2arr

__all__ = ['TopHat']

class TopHat(object):
    """TopHat prior object.
    Based on `prospector`:
        https://github.com/bd-j/prospector/blob/master/prospect/models/priors.py
    """
    def __init__(self, low=0.0, upp=1.0):
        """Constructor.
        Parameters
        ----------
        low : float
            Lower limit of the flat distribution.
        upp : float
            Upper limit of the flat distribution.
        """
        self.distr = scipy.stats.uniform
        self._low = low
        self._upp = upp

    def get_mean(self):
        """Get the mean value of the distribution. Can be used as initial values."""
        return self.distr.mean(loc=self.loc, scale=self.scale)

    def lnp(self, x):
        """Compute the value of the probability desnity function at x and
        return the ln of that.
        Parameters
        ----------
        x : float or numpy array
            Parameter values.
        Return
        ------
        lnp : float or numpy array
            The natural log of the prior probability at x
        """
        return self.distr.logpdf(x, loc=self.loc, scale=self.scale)

    def unit_transform(self, x):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        Parameters
        ----------
        x : float or numpy array.
           Values of the CDF (between 0 and 1).
        Return
        ------
            The parameter value corresponding to the value of the CDF given by `unit_arr`.
        """
        return self.distr.ppf(x, loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x):
        """Go from the parameter value to the unit coordinate using the cdf.
        Parameters
        ----------
        x : float or numpy array.
           Values of the CDF (between 0 and 1).
        Return
        ------
            The corresponding value in unit coordinate.
        """
        return self.distr.cdf(x, loc=self.loc, scale=self.scale)

    def sample(self, nsample):
        """Sample the distribution.
        Parameter
        ---------
        nsample : int
            Number of samples to return.
        Return
        ------
        sample : arr
            `nsample` values that follow the distribution.
        """
        return self.distr.rvs(loc=self.loc, scale=self.scale, size=nsample)

    @property
    def low(self):
        """Lower limit of the distribution."""
        return self._low

    @property
    def upp(self):
        """Upper limit of the distribution."""
        return self._upp

    @property
    def scale(self):
        """The `scale` parameter of the distribution."""
        return self._upp - self._low

    @property
    def loc(self):
        """The `loc` parameter of the distribution."""
        return self._low

    @property
    def range(self):
        """The range of the distribution."""
        return (self._low, self._upp)
    
    
    
# -------------------------------- #
def lnprior(in_arr, usekeys, prhiarr, prloarr, nested=False):
    in_pos = str2arr(2, inarr = in_arr, usekeys=usekeys)
    in_pos_arr = str2arr(1, instr = in_pos, usekeys=usekeys)        
    allprior = []
    for i in range(len(in_arr)):
        a = TopHat(prloarr[i], prhiarr[i])
        allprior.append(a.lnp(in_arr[i]))
        
    lnp = np.nansum(allprior)
    if nested and np.isfinite(lnp):
        return 0.0
    return lnp



# -------------------------------- #
def prior_transform(unit_coords, usekeys,
                    prhiarr, prloarr):    
    theta = np.empty((len(unit_coords)))
    key_arr = np.array(['velz', 'sigma', 'logage', 'zh', 'feh', 
            'ah', 'ch', 'nh','nah','mgh','sih','kh','cah','tih',
            'vh','crh','mnh','coh','nih','cuh','srh','bah','euh',
                'teff','imf1','imf2','logfy','sigma2','velz2',
                'logm7g','hotteff','loghot','fy_logage',
                'logemline_h','logemline_oii','logemline_oiii',
                'logemline_sii','logemline_ni','logemline_nii',
                'logtrans','jitter','logsky', 'imf3','imf4','h3','h4'])
    
    for i, ikey in enumerate(usekeys):
        ind = np.where(key_arr==ikey)
        a = TopHat(prloarr[ind].item(), prhiarr[ind].item())
        theta[i] = a.unit_transform(unit_coords[i])
    return theta