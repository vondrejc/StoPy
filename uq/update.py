from __future__ import division
import numpy as np
import scipy.linalg as splin
from copy import copy
from uq.dist import Dists
from uq.gpc import GPC


class PCE_CE():
    # polynomial chaos expansion of conditional expectation

    def __init__(self, Yfun, phi, quad, **kwargs):
        self.Yfun=Yfun
        self.phi=phi.copy_zeros()
        self.quad=quad
        self.__dict__.update(kwargs)
        self.prepare()

    def prepare(self):
        ip, iw=self.quad.get_integration() # quadrature points and weights
        Yxi=np.atleast_2d(self.Yfun(ip))
        phibasY=self.phi.eval_all_basis(Yxi)
        self.wphibasY=phibasY*iw
        A=np.einsum('ik,jk->ij', phibasY, self.wphibasY) # linear system
        self.lu_factor=splin.lu_factor(A)
        self.ip=ip

    def __call__(self, Xfun):
        self.Xfun=Xfun
        Xxi=np.atleast_2d(Xfun(self.ip))
        B=np.einsum('...k,jk->...j', Xxi, self.wphibasY) # right-hand sides
        rshape=Xxi.shape[:-1]
        phicoef=np.empty(rshape+(self.phi.ndofs,))
        for ind in np.ndindex(*rshape): # solving linear system
            phicoef[ind]=splin.lu_solve(self.lu_factor, B[ind])

        self.phi.set_coef(np.swapaxes(phicoef, axis1=-1, axis2=0)) # setting coefficients to mappings
        return self.phi.copy()

    def fun(self, x=None, Xfun=None, quad=None):
        if Xfun is None:
            Xfun=self.Xfun

        if x is None:
            phi=self.phi.copy()
        else:
            phi=self.phi.copy_zeros()
            phi.set_coef(x.reshape(phi.coef.shape))

        if quad is None:
            quad=self.quad
        ip, iw=quad.get_integration()
        Yxi=phi(np.atleast_2d(self.Yfun(ip)))
        Xxi=np.atleast_2d(Xfun(ip))
        val=np.sum(iw*(Xxi-Yxi)**2)
        return val


class ConditionedExpectation():
    # Conditioned expectation
    def __init__(self, Y, E, yhat, quad, theta, **kwargs):
        self.Y=Y
        self.E=E
        self.yhat=yhat
        self.quad=quad
        self.theta=theta
        if 'Ediff' in kwargs:
            self.Ediff=kwargs['Ediff']
        else:
            self.Ediff=1

    def __call__(self, X, threshold=1e-14, debug=True):
        if isinstance(self.E, Dists): # old method (working for \xi - mean and cov)
            method_str='Dist'
            Y=self.Y
            yhat=self.yhat
            Ediff=self.Ediff
            assert(self.E.dim==1)
            ip, iw = self.quad.get_integration()
            lhs=0.
            rhs=np.zeros_like(X(ip[:,0],yhat))
            nthreshold=0
            for ii in range(iw.size):
                ipi = ip[:,ii]
                Yipi=Y(ipi)
                likelihood = self.E.pdf(Yipi)[0]
                eta = None
                if likelihood > threshold:
                    nthreshold += 1
                    rhs += iw[ii]*X(ipi, eta)*likelihood
                    lhs += iw[ii]*likelihood
            val = rhs/lhs
        elif isinstance(self.E, GPC): # new - more general for GPC error
            method_str='GPC'
            Y=self.Y
            yhat=self.yhat
            Epol = self.E.get_poly()
            E0pol = copy(Epol)
            E0pol.coef[0] = 0.
            Ediff=Epol.deriv()
            assert(self.E.dim==1)
            ip, iw = self.quad.get_integration()
            lhs=0.
            rhs=np.zeros_like(X(ip[:,0],yhat))
            nthreshold=0
            for ii in range(iw.size):
                ipi = ip[:,ii]
                Epom = E0pol - (Y(ipi).squeeze()-yhat)
                r = Epom.roots()
                eta = np.unique(np.extract(np.abs(r.imag)<1e-14, r).real)
                likelihood = self.theta.pdf(eta)[0]/Ediff(eta)[0]
                if likelihood > threshold:
                    nthreshold += 1
                    rhs += iw[ii]*X(ipi, eta)*likelihood
                    lhs += iw[ii]*likelihood
            val = rhs/lhs

        if debug:
            print('method: {4}; treshold={0}, nonzeros={1:.5%} = {2}/{3}'.format(threshold, nthreshold/iw.size, nthreshold, iw.size, method_str))
            print('rhs={0}; lhs={1}'.format(rhs, lhs))
        return val

    def mean(self, q0, threshold=1e-14):
        return self(lambda xi, eta: q0(xi), threshold)

    def covariance(self, q0, mean=0., threshold=1e-14):
        def fun(xi, eta):
            qxim = q0(xi)-mean
            return np.outer(qxim, qxim)
        covar = self(fun, threshold=threshold)
        return covar
