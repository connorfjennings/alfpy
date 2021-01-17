# 11/14/21 copied from prospector/prospect/models/priors.py /


import numpy as np
import scipy.stats

class Prior(object):
    """
    Encapsulate the priors in an object.  Each prior should have a
    distribution name and optional parameters specifying scale and location
    (e.g. min/max or mean/sigma).  These can be aliased at instantiation using
    the ``parnames`` keyword. When called, the argument should be a variable
    and the object should return the ln-prior-probability of that value.
    .. code-block:: python
        ln_prior_prob = Prior()(value)
    Should be able to sample from the prior, and to get the gradient of the
    prior at any variable value.  Methods should also be avilable to give a
    useful plotting range and, if there are bounds, to return them.
    :param parnames:
        A list of names of the parameters, used to alias the intrinsic
        parameter names.  This way different instances of the same Prior can
        have different parameter names, in case they are being fit for....
    """

    def __init__(self, parnames=[], name='', **kwargs):
        """Constructor.
        :param parnames:
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

    def __repr__(self):
        argstring = ['{}={}'.format(k, v) for k, v in list(self.params.items())]
        return '{}({})'.format(self.__class__, ",".join(argstring))

    def update(self, **kwargs):
        """Update `params` values using alias.
        """
        for k in self.prior_params:
            try:
                self.params[k] = kwargs[self.alias[k]]
            except(KeyError):
                pass
        # FIXME: Should add a check for unexpected kwargs.

    def __len__(self):
        """The length is set by the maximum size of any of the prior_params.
        Note that the prior params must therefore be scalar of same length as
        the maximum size of any of the parameters.  This is not checked.
        """
        return max([np.size(self.params.get(k, 1)) for k in self.prior_params])

    def __call__(self, x, **kwargs):
        """Compute the value of the probability desnity function at x and
        return the ln of that.
        :param x:
            Value of the parameter, scalar or iterable of same length as the
            Prior object.
        :param kwargs: optional
            All extra keyword arguments are sued to update the `prior_params`.
        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        pdf = self.distribution.pdf
        try:
            p = pdf(x, *self.args, loc=self.loc, scale=self.scale)
        except(ValueError):
            # Deal with `x` vectors of shape (nsamples, len(prior))
            # for pdfs that don't broadcast nicely.
            p = [pdf(_x, *self.args, loc=self.loc, scale=self.scale)
                 for _x in x]
            p = np.array(p)

        with np.errstate(invalid='ignore'):
            lnp = np.log(p)
        return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.
        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.rvs(*self.args, size=len(self),
                                     loc=self.loc, scale=self.scale)

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.
        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.ppf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.cdf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def gradient(self, theta):
        raise(NotImplementedError)

    @property
    def loc(self):
        """This should be overridden.
        """
        return 0

    @property
    def scale(self):
        """This should be overridden.
        """
        return 1

    @property
    def args(self):
        return []

    @property
    def range(self):
        raise(NotImplementedError)

    @property
    def bounds(self):
        raise(NotImplementedError)

    def serialize(self):
        raise(NotImplementedError)
        
        
        
class Uniform(Prior):
    """A simple uniform prior, described by two parameters
    :param mini:
        Minimum of the distribution
    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.uniform

    @property
    def scale(self):
        return self.params['maxi'] - self.params['mini']

    @property
    def loc(self):
        return self.params['mini']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range