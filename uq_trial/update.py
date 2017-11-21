from __future__ import division
from numpy.linalg import norm
import scipy.optimize

import numpy as np
import scipy.linalg as splin
from uq.gpc import LinOper, GPC
from uq.operators import matrix_fun
from scipy.integrate import quad as sp_quad
import warnings


def mmse_update(Yfun, Xfun, phi, quad, debug=False, solver='lu'): # delete ???
    warnings.warn("Will be deleted")

    ip, iw=quad.get_integration() # quadrature points and weights
    Yxi=np.atleast_2d(Yfun(ip))
    Xxi=np.atleast_2d(Xfun(ip))

    phibasY=phi.eval_all_basis(Yxi)
    wphibasY=phibasY*iw

#     if isinstance(phi, Simple):
#         Adiag=np.einsum('ikl,ikl->i', phibasY, wphibasY) # linear system
#         B=np.einsum('ik,jlk->ji', Xxi, wphibasY) # right-hand sides
#         phicoef=B/np.atleast_2d(Adiag).T # solving linear system
    if isinstance(phi, GPC):
        A=np.einsum('ik,jk->ij', phibasY, wphibasY) # linear system
        B=np.einsum('ik,jk->ji', Xxi, wphibasY) # right-hand sides
        if solver in ['lu']:
            phicoef=np.linalg.solve(A, B) # solving linear system
        elif solver in ['pinv']:
            iA=splin.pinv(A)
            phicoef=iA.dot(B)
        elif solver in ['pinv2']:
            iA=splin.pinv2(A)
            phicoef=iA.dot(B)
        elif solver in ['pinvh']:
            iA=splin.pinvh(A)
            phicoef=iA.dot(B)
    else:
        raise NotImplementedError()

    if debug:
        print('condition number = {}'.format(np.linalg.cond(A)))
    phi.set_coef(phicoef) # setting coefficients to mappings
    return phi

def spectral_Kalman(xf, y_truth, Yfun, u, err, correct_var=True):
    y=Yfun(u)
    z=y.add_independent(err)
    xf=xf.add_independent(err.copy_zeros())
    Cxz=xf.covariance(z)
    iCz=np.linalg.inv(z.covariance())
    K=Cxz.dot(iCz)
    Kfun=LinOper(name='KalmanGain', val=K)
    xa=xf+Kfun(-(z-y_truth))
    xa.name='xa'

    assert(norm(xa.covariance()-(xf.covariance()-K.dot(Cxz.T)))<1e-14)
    if correct_var:
        xa0=xf-Kfun(z)
        assert(norm(xa0.variance()-xa.variance())<1e-14)
        Cxa=xa.variance()
        sqrtCxa=matrix_fun(Cxa, lambda x: x**0.5, nulzero=False)
        print(norm(Cxa-sqrtCxa.dot(sqrtCxa)))
        assert(norm(Cxa-sqrtCxa.dot(sqrtCxa))<1e-14)
    else:
        return xa


class MMSEnonlinear():
    """
    general class for nonlinear MMSE update
    """

    def eval(self, y):
        raise NotImplementedError()

    def fun(self, x):
        raise NotImplementedError()

    def jac(self, x):
        raise NotImplementedError()


class MMSE_update_cov(MMSEnonlinear): # delete ???

    def __init__(self, Yfun, Xfun, phi, quad, method='exp', **kwargs):
        warnings.warn("Will be deleted")
        self.Yfun=Yfun
        self.Xfun=Xfun
        self.phi=phi
        self.quad=quad
        self.ip, self.iw=quad.get_integration() # quadrature points and weights
        self.phi_coef_shape=phi.coef.shape
        self.iter_fun=0
        self.iter_jac=0
        if method in ['exp', 0]: # exponential Ansatz
            self.fun=self._fun_exp
            self.jac=self._jac_exp
            self.eval=self._eval_exp
        elif method in ['R', 1]: # Riemanian metric
            self.fun=self._fun_R
            self.jac=self._jac_R
        elif method in ['S', 2]: # Quadratic Ansatz
            self.fun=self._fun_S
            self.jac=self._jac_S
            self.eval=self._eval_S
        else:
            raise NotImplementedError('method ({})'.format(method))

    def _eval_S(self, y):
        return self.phi(y)**2

    def _fun_S(self, x):
        self.iter_fun+=1
        self.phi.coef=x.reshape(self.phi_coef_shape)

        Yxi=self.Yfun(self.ip)
        Xxi=self.Xfun(self.ip)
        val=np.sum(self.iw*(Xxi**2-self.phi(Yxi)**2)**2)
        return val

    def _jac_S(self, x, xdel=None):
        self.iter_jac+=1
        self.phi.coef=x.reshape(self.phi_coef_shape)

        Xxi=self.Xfun(self.ip)
        Yxi=self.Yfun(self.ip)
        phi_Yxi=self.phi(Yxi)
        vec=self.iw*(Xxi**2-phi_Yxi**2)*phi_Yxi
        dphi=self.phi.copy_zeros()
        if xdel is None:
            val=np.empty(self.phi.ndofs)
            for ii in range(self.phi.ndofs):
                xdel=np.zeros(self.phi.ndofs)
                xdel[ii]=1.
                dphi.coef=xdel.reshape(self.phi_coef_shape)
                val[ii]=-4*np.sum(vec*dphi(Yxi))

        elif isinstance(xdel, np.ndarray):
            dphi.coef=xdel.reshape(self.phi_coef_shape)
            val=-4*np.sum(vec*dphi(Yxi))

        else:
            raise ValueError('xdel type ({0})'.format(type(xdel)))
        return val

    def _fun_R(self, x):
        self.iter_fun+=1
        self.phi.coef=x.reshape(self.phi_coef_shape)

        Yxi=self.Yfun(self.ip)
        Xxi=self.Xfun(self.ip)

        val=np.log(self.phi(Yxi)/Xxi)**2
        return val

    def _jac_R(self, x, xdel=None):
        self.iter_jac+=1
        self.phi.coef=x.reshape(self.phi_coef_shape)

        Yxi=self.Yfun(self.ip)
        ratio=self.phi(Yxi)/self.Xfun(self.ip)
        dphi=self.phi.copy_zeros()
        if xdel is None:
            val=np.empty(self.phi.ndofs)
            for ii in range(self.phi.ndofs):
                xdel=np.zeros(self.phi.ndofs)
                xdel[ii]=1.
                dphi.coef=xdel.reshape(self.phi_coef_shape)
                val[ii]=-2*np.log(ratio)/ratio*dphi(Yxi)

        elif isinstance(xdel, np.ndarray):
            dphi.coef=xdel.reshape(self.phi_coef_shape)
            val=-2*np.log(ratio)/ratio*dphi(Yxi)
        else:
            raise ValueError('xdel type ({0})'.format(type(xdel)))
        return val

    def _eval_exp(self, y):
        return np.exp(self.phi(y))

    def _fun_exp(self, x):
        self.iter_fun+=1
        self.phi.coef=x.reshape(self.phi_coef_shape)

        Yxi=self.Yfun(self.ip)
        Xxi=self.Xfun(self.ip)

        val=np.sum(self.iw*(Xxi-np.exp(self.phi(Yxi)))**2)
        return val

    def _jac_exp(self, x, xdel=None):
        self.iter_jac+=1
        self.phi.coef=x.reshape(self.phi_coef_shape)

        Yxi=self.Yfun(self.ip)
        Xxi=self.Xfun(self.ip)
        exp_phi_Yxi=np.exp(self.phi(Yxi))
        vec=self.iw*(Xxi-exp_phi_Yxi)*exp_phi_Yxi
        dphi=self.phi.copy_zeros()
        if xdel is None:
            val=np.empty(self.phi.ndofs)
            for ii in range(self.phi.ndofs):
                xdel=np.zeros(self.phi.ndofs)
                xdel[ii]=1.
                dphi.coef=xdel.reshape(self.phi_coef_shape)
                val[ii]=-2*np.sum(vec*dphi(Yxi))

        elif isinstance(xdel, np.ndarray):
            dphi.coef=xdel.reshape(self.phi_coef_shape)
            val=-2*np.sum(vec*dphi(Yxi))
        else:
            raise ValueError('xdel type ({0})'.format(type(xdel)))
        return val

    def update(self, **kwargs):
        solver=kwargs.get('solver', 'BFGS')
        x0=kwargs.get('x0', np.zeros(self.phi.ndofs))
        res=scipy.optimize.minimize(fun=self.fun, x0=x0, jac=self.jac,
                                    method=solver)
        self.phi.coef=res.x.reshape(self.phi_coef_shape)
        if kwargs.get('debug', False):
            print('iter_fun={0}, iter_jac={1}'.format(self.iter_fun, self.iter_jac))
        return self.phi

def get_phi_pdf(ys, prior, err, S, tol=None):
    pa_mean=np.empty(ys.shape[0])
    pa_var=np.empty(ys.shape[0])
    if tol is None:
        tol=1e-8

    for ii, y in enumerate(ys):
        mean, mean_err=sp_quad(lambda q:err.pdf(y-S(q))*prior.pdf(q),-np.inf, np.inf,
                               epsabs=tol, epsrel=tol)
        pam, pam_err=sp_quad(lambda q:q*err.pdf(y-S(q))*prior.pdf(q)/mean,-np.inf, np.inf,
                             epsabs=tol, epsrel=tol)
        pa_mean[ii]=pam
        pav, pav_err=sp_quad(lambda q:(q-pam)**2*err.pdf(y-S(q))*prior.pdf(q)/mean,-np.inf, np.inf,
                             epsabs=tol, epsrel=tol)
        pa_var[ii]=pav
        print('errors = {0}; {1}; {2}'.format(mean_err, pam_err, pav_err))

    return pa_mean, pa_var


if __name__=='__main__':
    exec('../lorentz.py')
