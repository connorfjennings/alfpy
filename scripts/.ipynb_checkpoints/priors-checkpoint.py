import numpy as np
import scipy.stats 
#from alf_constants import *
__all__ = ['TopHat', 'ClippedNormal']


class TopHat(object):
    """
    TopHat prior object.
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
class ClippedNormal(object):
    """A Gaussian prior clipped to some range.
    Based on `prospector`:
        https://github.com/bd-j/prospector/blob/master/prospect/models/priors.py
    """
    def __init__(self, mean=0.0, sigma=1.0, mini=-np.inf, maxi=np.inf, **kwargs):
        """Constructor.
        Parameters
        ----------
        low : float
            Lower limit of the flat distribution.
        upp : float
            Upper limit of the flat distribution.
        """
        self.distr = scipy.stats.truncnorm
        self._mean = mean
        self._sigma = sigma
        self._mini = mini
        self._maxi = maxi

    
    def get_mean(self):
        return self.distr.mean(loc=self.loc, scale=self.scale)

    
    def lnp(self, x, **kwargs):
        return self.distr.logpdf(x, *self.args,loc=self.loc, scale=self.scale)

    
    def unit_transform(self, x, **kwargs):
        return self.distr.ppf(x, *self.args,loc=self.loc, scale=self.scale)

    
    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        Parameters
        ----------
        x : float or numpy array.
           Values of the CDF (between 0 and 1).
        Return
        ------
            The corresponding value in unit coordinate.
        """
        return self.distr.cdf(x, *self.args, loc=self.loc, scale=self.scale)

    def sample(self, nsample, **kwargs):
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
        return self.distr.rvs(*self.args, loc=self.loc, scale=self.scale, size=nsample)


    @property
    def scale(self):
        return self._sigma

    @property
    def loc(self):
        return self._mean

    @property
    def range(self):
        return (self._mini, self._maxi)
    
    @property
    def args(self):
        a = (self._mini - self._mean) / self._sigma
        b = (self._maxi - self._mean) / self._sigma
        return [a, b]

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range




