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