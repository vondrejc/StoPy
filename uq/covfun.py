'''
covariance functions
'''

import numpy as np
from scipy.special import gamma, kv # modified Bessel function

class Covariance():
    def __call__(self, d):
        return self.eval(d)

class Matern(Covariance):
    # matern covariance functions

    def __init__(self, **kwargs):
        pars = {'nu': 0.5, 'rho': 1., 'sigma': 1.}
        pars.update(kwargs)
        self.__dict__.update(pars)

    def eval(self, d):
        nu = self.nu
        sigma = self.sigma
        rho = self.rho
        if False:#np.isclose(nu, 3./2):
            fac = 3**0.5*np.abs(d)/rho
            return sigma**2 * (1+fac) * np.exp(-fac)
        else:
            fac = (2*nu)**0.5 * np.abs(d)/rho
            return sigma**2* 2**(1-nu)/gamma(nu) * (fac)**nu * kv(nu, fac)

class Expcov(Matern):
    # exponential covariance functions

    def eval(self, d):
        val = self.sigma**2 * np.exp(-np.abs(d)/self.rho)
        return val

    def ft(self, k):
        # valid for dim=1
        a = 1./self.rho
        s = self.Y #scale
        return self.sigma**2 * 2*a/s / (a**2+4*np.pi**2*(k/s)**2)

class Sexpcov(Matern):
    # squared exponential covariance functions

    def eval(self, d):
        val = self.sigma**2*np.exp(-np.abs(d)**2/2./self.rho**2)
        return val

    def ft(self, k):
        # valid for dim=1
        alp = 1./2/self.rho**2
        s = self.Y #scale
        return self.sigma**2 * (np.pi/alp)**0.5/s * np.exp(-(np.pi*k/s)**2/alp)
