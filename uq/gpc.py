from copy import deepcopy
from numpy.linalg import norm
import warnings

from general.base import Struct
from general.base import Timer
import numpy as np
from uq.polynomials import poly, find_poly_intervals, polyval, GPCpoly
from uq.quadrature import Quad_Gauss, Quad_rectangular
from uq.samples import Samples


# import uq.cgpc as cgpc
# import uq.cpoly as cpoly
astr='abcdefgh'
istr='ijklmnopqrst'
ustr='uvwxyz'

class GPC(Struct, GPCpoly):

    def __init__(self, pchars=None, name='', I=None, p=0, p_max=None, rshape=(1,),
                 coef=None):
        self.name=name
        self.pchars=pchars
        self.dim=len(pchars)
        if I is None:
            self.I=self.get_indices(p, p_max)
        else:
            self.I=np.array(I, dtype=np.int)

        self.p=self.I.max(axis=0)
        self.p_max=self.I.sum(axis=1).max()
        self.ndofs=self.I.shape[0]
        self.set_coef(coef=coef, rshape=rshape)
        self._assert()

        self._norm=[]
        self._eval=[]
        for char in self.pchars:
            self._norm.append(poly[char].norm)
            self._eval.append(poly[char].eval)

    def _assert(self):
        assert(self.dim==len(self.pchars))
        assert(self.dim==self.I.shape[1])
        assert(self.ndofs==self.I.shape[0])
        assert(self.ndofs==self.coef.shape[0])
        assert(self.rshape==self.coef.shape[1:])
        assert(self.dim==self.p.size)
        assert(np.all(np.equal(self.I[0], np.zeros(self.dim, dtype=np.int))))

    def set_coef(self, coef=None, rshape=None):
        if coef is not None: # coef is numpy.ndarray
            assert(coef.shape[0]==self.ndofs)
            self.coef=np.atleast_2d(coef)
        else:
            self.coef=np.zeros((self.ndofs,)+rshape)
        self.rshape=self.coef.shape[1:]
        self.rdim=len(self.rshape)

    @staticmethod
    def get_indices(p, p_max=None):
        p=np.atleast_1d(np.array(p))
        if p_max is None:
            process_order=lambda I: I
        elif p_max>=p.max():
            def process_order(I):
                order=np.sum(I, axis=1)
                ind=np.where(order<=p_max)
                return I[ind[0]]
        else:
            raise ValueError('p ({0}) >= p_max({1}).'.format(p, p_max))

        I=np.atleast_2d(np.arange(p[0]+1, dtype=np.int)).T
        for ii in range(1, p.size):
            I_new=np.repeat(np.atleast_2d(np.arange(p[ii]+1, dtype=np.int)).T, I.shape[0], axis=0)
            I=np.tile(I, (p[ii]+1, 1))
            I=process_order(np.hstack([I, I_new]))
        return I

    def sparsify(self, threshold=1e-12):
        nrm=norm(self.coef, axis=tuple(range(1, self.rdim+1)))
        nrm[0]=2*threshold
        ind=np.flatnonzero(nrm>threshold)
        print('{0} of {1} zeroed.'.format(self.ndofs-ind.size, self.ndofs))
        return GPC(pchars=self.pchars, name=self.name, I=self.I[ind], coef=self.coef[ind])

    def sort(self, kind='mergesort'):
        order=np.sum(self.I, axis=1)
        ind=np.argsort(order, kind=kind)
        return GPC(pchars=self.pchars, name=self.name, I=self.I[ind], coef=self.coef[ind])

    def __call__(self, x):
        return self.eval(x)

    def _assert_xi(self, xi):
        xi=np.atleast_2d(xi)
        if not self.dim==xi.shape[0]:
            xi=xi.T
        assert(self.dim==xi.shape[0])
        assert(xi.ndim==2)
        return xi

    def _assert_q(self, q):
        q=np.atleast_2d(q)
        if not self.rshape==q.shape[:-1]:
            q=q.T
        assert(self.rshape==q.shape[:-1])
        return q

    def eval(self, xi):
        # evaluation of gpc at points x
        xi=self._assert_xi(xi)
        val=np.zeros(self.rshape+xi.shape[1:])
        for ii, order in enumerate(self.I):
            val+=np.einsum('...,x->...x', self.coef[ii], self.eval_basis(order, xi))
        return val.squeeze()

    def eval_basis(self, order, xi):
        """
        evaluate a basis function at points xi
        Parameters:
        order : numpy.ndarray of shape = (self.dim,)
            polynomial order
        xi : numpy.array of shape = (self.dim, ...)
            evaluation points
        """
        val=np.ones(xi.shape[1], dtype='float')/self.get_norm(order)
        for ii in range(self.dim):
            val*=self._eval[ii](order[ii], xi[ii])
        return val

    def eval_all_basis(self, xi, fast=0):
        """
        Returns matrix with rows containing evaluation of basis function.
        """
        assert(xi.shape[0]==self.dim)
        if fast==1:
            basis1d=np.empty(self.dim, dtype=object)
            for ii, ps in enumerate(self.p):
                x=xi[ii]
                bas=np.empty([ps+1, xi.shape[1]])
                for p in range(ps+1):
                    bas[p]=self._eval[ii](p, x)/self._norm[ii](p)
                basis1d[ii]=bas

            W=np.ones([self.ndofs, xi.shape[1]])
            for ii, ind in enumerate(self.I):
                for jj, p in enumerate(ind):
                    W[ii]*=basis1d[jj][p]
            return W
        elif fast==2:
            return cpoly.eval_all_basis(xi, self.p, self.I, self._eval, self._norm)
        else:
            W=np.empty([self.ndofs, xi.shape[1]])
            for ii, ind in enumerate(self.I): # TODO: speed up
                W[ii]=self.eval_basis(ind, xi)
            return W

    def get_norm(self, order):
        # weighted square norm of basis functions a(u,u)
        norm=1.
        for ii, syschar in enumerate(self.pchars):
            norm*=poly[syschar].norm(order[ii])
        return norm

    def get_poly(self):
        if self.dim>1:
            raise NotImplementedError("Only for 1d polynomials!")
        if self.pchars in ['u']:
            raise NotImplementedError()
        else:
            poly1d=poly[self.pchars].poly
            pn=poly1d(np.array([self.coef[ii]/self.get_norm(self.I[ii])
                                for ii in range(self.ndofs)]).squeeze())
            return pn

    def process_integration(self, quad, **pars):
        # for general quad it returns integration points
        pars={'pchars': self.pchars,
              'p_int': self.p+1,
              'grid': 'tensor'} # default parameters for integration
        if quad is None:
            ip, iw=Quad_Gauss(**pars).get_integration()
        elif isinstance(quad, dict):
            pars.update(quad)
            ip, iw=Quad_Gauss(**pars).get_integration()
        elif isinstance(quad, tuple):
            ip, iw=quad
            assert(iw.size==ip.shape[1])
        else:
            ip, iw=quad.get_integration()
        return ip, iw

    def project(self, func, quad=None, ip_max=1e5, method=0, **pars):
        # projection of function func on polynomial space
        assert('m' not in self.pchars)
        ip, iw=self.process_integration(quad, **pars)

        if iw.size>ip_max:
            warnings.warn('Split integration in projection (points{0})!'.format(iw.size))

        if method==1: # not working properly
            self.coef=cgpc.project(self.coef, func, ip, iw, self.I, self.pchars)
        elif method==2: # not working properly
            self.coef=cpoly.project(self.coef, func, ip, iw, self.I, self.pchars, self.p)
        else:
            bas=self.eval_basis
            self.coef[:]=0.
            func_ipw=iw*np.atleast_2d(func(ip))
            for ii, ind in enumerate(self.I): # TODO: speed up
                self.coef[ii]+=np.einsum('...i,i', func_ipw, bas(ind, ip))
        return self

    def expand(self, dist, quad=None, fixvar=False):
        # Computes GPC expansion of a distribution.
        assert ('m' not in self.pchars)
        poldist=self.get_dist(self.pchars)
        if self.dim>1:
            raise AssertionError("Unsupported for dim({0})>1!".format(self.dim))

        func=lambda x: dist.ppf(poldist[0].cdf(x))
        self.project(func, quad=quad)

        if fixvar:
            raise NotImplementedError()
        return self

    def combine(self, Y):
        # TODO: need explanation
        return combine(self, Y)

    def dot(self, Y):
        # TODO: need explanation
        return outer(self, Y)

    def outer(self, Y):
        # TODO: need explanation
        return outer(self, Y)

    def element_wise(self, Y):
        # TODO: need explanation
        return element_wise(self, Y)

    def __neg__(self):
        gpc=self.copy()
        gpc.coef*=-1
        return gpc

    def __mul__(self, Y):
        # tensor product of two GPCs
        X=self
        if isinstance(Y, GPC):
            raise NotImplementedError()

        elif isinstance(Y, np.ndarray) and Y.ndim==1:
            raise NotImplementedError()

        elif isinstance(Y, np.ndarray) and Y.ndim==2:
            assert(Y.shape[1:]==X.rshape)
            gpc=X.copy()
            gpc.coef=X.coef.dot(Y)
            return gpc

        else:
            raise NotImplementedError()

    def __add__(self, Y):
        X=self
        if isinstance(Y, GPC):
            assert(X._eq_type(Y))
            gpc=GPC(name=X.name+'+'+Y.name, pchars=X.pchars,
                      I=X.I, rshape=X.rshape, coef=X.coef+Y.coef)
        elif isinstance(Y, np.ndarray):
            Y=np.atleast_1d(Y.squeeze())
            if X.coef.shape==Y.shape:
                gpc=X.copy()
                gpc.coef+=Y
            elif X.rshape==Y.shape:
                gpc=X.copy()
                gpc.coef[0]+=Y
        else:
            raise ValueError('GPC.__add__ do not support operand ({0})'.format(type(Y)))
        return gpc

    def __sub__(self, Y):
        return self.__add__(-Y)

    def add_independent(self, Y):
        # addition of two independent GPCs
        X=self
        assert(X.rshape==Y.rshape)
        assert(np.array_equal(X.I[0], np.zeros(X.dim, dtype=np.int)))
        assert(np.array_equal(Y.I[0], np.zeros(Y.dim, dtype=np.int)))

        gpc=GPC(name=X.name+'+'+Y.name, pchars=X.pchars+Y.pchars,
                I=com_vals(X.I, Y.I), rshape=X.rshape)
        coef=np.empty((X.ndofs+Y.ndofs-1,)+X.rshape)
        coef[:X.ndofs]=X.coef
        coef[0]+=Y.coef[0]
        coef[X.ndofs:]=Y.coef[1:]
        gpc.coef=coef
        return gpc

    def pdf(self, qs, method='direct', smpl_coef=1e2): # EXPERIMENTAL
        """
        Computes the joint PDF of the GPC germ.
        parameters:
        ----------
        qs: numpy.array of shape (self.dim, ...)
            values to evaluate pdf
        """
        if method in [1, 'direct'] and self.dim==1: # direct method in dim=1
            pdf=np.zeros(qs.size) # preallocation
            poldist=self.get_dist(self.pchars)
            pn=self.get_poly()
            dpn=pn.deriv()
            for ii, x in enumerate(qs.T):
                ints=find_poly_intervals(pn-x)
                val=poldist[0].pdf(ints)/polyval(dpn, ints)
                pdf[ii]=np.sum(np.array([-1., 1.])*val)
        elif method in ['mc']:
            assert(self.rdim==1)
            assert(smpl_coef*qs.size<1e8)
            pdf=np.zeros(qs.size) # preallocation
            smpl=self.samples(n=smpl_coef*qs.size, method=method)
            for sm in smpl.val.T:
                ii=(np.abs(qs-sm)).argmin()
                pdf[ii]+=1
            pdf*=1./smpl.n/(qs[1]-qs[0])
        else:
            raise NotImplementedError("method ({0}), dim ({1})".format(method, self.dim))
        return pdf

    def cdf(self, xi, method='sampling', nsmpl=1e6): # EXPERIMENTAL
        """Computes the joint CDF of the GPC germ."""
        xi=self._assert_xi(xi)
        cdf=np.empty(xi.shape[1]) # preallocation
        if method in [1, 'direct'] and self.dim==1: # direct method in dim=1
            poldist=self.get_dist(self.pchars)
            pn=self.get_poly()
            for ii, x in enumerate(xi.T):
                ints=find_poly_intervals(pn-x)
                val=poldist[0].cdf(ints)
                cdf[ii]=np.sum(np.array([-1., 1.])*val)
        elif method in [0, 'sampling']: # sampling method
            ip, iw=Quad_rectangular(self.pchars, n=nsmpl).get_integration()
            vals=self.__call__(ip)
            for ii, x in enumerate(xi.T):
                ind=np.ones(ip.shape[1])
                for jj in range(self.dim):
                    ind*=vals[jj]<x[jj]
                cdf[ii]=float(np.count_nonzero(ind))/iw.size
        else:
            msg="method ({0}) in gpc.pdf".format(method)
            raise NotImplementedError(msg)
        return cdf

    def fourier(self, xi):
        '''Computes the characteristic function.'''
        raise NotImplementedError()

    def stats(self, moments='mv', direct=False):
        """
        moments: str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean, 'v' = variance, 's' = (Fisher's) skew,
            'k' = (Fisher's) kurtosis. (default is 'mv')
        """
        # ToDo: vectorize
        assert('m' not in self.pchars)
        if direct :
            var=0.
            for ii in range(1, self.ndofs):
                var+=(self.coef[ii])**2*self.get_norm(self.I[ii])**2
        else:
            var=self.moment(2)

        kres={'m': self.coef[0],
              'v': var,
              's': self.moment(3)/var**(3./2),
              'k': self.moment(4)/var**2.-3.}

        res=[]
        for key in moments:
            res.append(kres[key])
        return res

    def moment(self, p, centered=True):
        # centralised moments
        assert(p>=1)
        assert('m' not in self.pchars)
        mean=self.coef[0]

        if p==1: # mean
            return mean
        else: # higher moments
            p_int=np.ceil(self.p_max*(1.+p)/2)
            ip, iw=Quad_Gauss(self.pchars, p_int).get_integration()
            if centered:
                func=lambda x: (x-np.atleast_2d(mean).T)**p
            else:
                func=lambda x: x**p

            mom=np.sum(iw*func(self.eval(ip)), axis=1)
            return mom

    def samples(self, n, method='mc'):
        # generate n samples based on GPC
        assert('m' not in self.pchars)
        n=int(n)
        poldist=self.get_dist(self.pchars)
        xi=np.empty([self.dim, n])
        if method in [0, 'mc']: # Monte-Carlo method
            for ii in range(self.dim):
                xi[ii]=poldist[ii].rvs(n)
        elif method in [1, 'grid']: # JV Quasi-Monte-Carlo method
            xi, _=Quad_rectangular(self.pchars, n).get_integration()
        return Samples(self.eval(xi), name='{0}_GPC({1})'.format(self.name, method))

    def mean(self):
        return self.coef[0]

    def variance(self):
        X=self
        val=np.zeros(X.rshape)
        for ii in range(1, X.ndofs):
            val+=X.coef[ii]**2
        return val

    def covariance(self, Y=None):
        X=self
        if Y is None:
            Y=X
        assert(isinstance(Y, GPC))
        assert(X.pchars==Y.pchars)
        assert(np.array_equal(X.p, Y.p))
        assert(X.rdim==1)
        assert(Y.rdim==1)
        assert(np.array_equal(X.I, Y.I))
        assert(np.array_equal(X.I[0], np.zeros(X.dim, dtype=np.int)))

        val=np.zeros(X.rshape+Y.rshape)
        for ii in range(1, X.ndofs):
            val+=np.outer(X.coef[ii], Y.coef[ii])
        return val

    def copy(self, **kwargs):
        req=['name', 'pchars', 'I', 'coef']
        kwpars={k: deepcopy(v) for k, v in self.__dict__.items() if k in req}
        kwpars.update(kwargs)
        return GPC(**kwpars)

    def copy_zeros(self, **kwargs):
        req=['name', 'pchars', 'I', 'rshape']
        kwpars={k: deepcopy(v) for k, v in self.__dict__.items() if k in req}
        kwpars.update(kwargs)
        return GPC(**kwpars)

    def __str__(self):
        ss='\nClass : {0}\n'.format(self.__class__.__name__)
        ss+='    name   = {0}\n'.format(self.name)
        ss+='    dim    = {0}\n'.format(self.dim)
        ss+='    rshape = {0} with rdim = {1}\n'.format(self.rshape, self.rdim)
        ss+='    ndofs  = {0}\n'.format(self.ndofs)
        ss+='    I      = {0} array of {1}\n'.format(self.I.shape, self.I.dtype)
        ss+='    coef   = {0} array of {1}\n'.format(self.coef.shape, self.coef.dtype)
        ss+='    pchars = {0}\n'.format(self.pchars)
        ss+='    p      = {0} with p_max = {1}\n'.format(self.p, self.p_max)
        ss+='    norm   = {0}\n'.format(norm(self.coef))
        if np.array(self.rshape).prod()<20:
            ss+='    mean   = \n{0}\n'.format(self.mean())
        if self.rdim==1:
            ss+='    covar  = \n{0}'.format(self.covariance())
        return ss

    def __repr__(self):
        return self.__str__()

    def __eq__(self, Y):

        if self._eq_type(Y):
            return norm(self.coef-Y.coef)
        else:
            return False

    def _eq_type(self, Y):
        X=self
        if X.rshape==Y.rshape and np.all(X.I==Y.I) and X.pchars==Y.pchars:
            return True
        else:
            print('GPC equality fail info:')
            for key in ['rshape', 'pchars', 'I']:
                for Z in [X, Y]:
                    val=getattr(Z, key)
                    if isinstance(val, np.ndarray):
                        print('    {0}.{1}.shape = {2}'.format(Z.name, key, val.shape))
                    else:
                        print('    {0}.{1} = {2}'.format(Z.name, key, val))

            return False


def combine(X, Y):
    # combination of two gpc (except the deterministic part)
    assert(X.rdim==1)
    assert(Y.rdim==1)
    gpc=GPC(name=X.name+'*'+Y.name, pchars=X.pchars+Y.pchars,
              I=com_vals(X.I, Y.I), rshape=(X.rshape[0]+Y.rshape[0],))

    gpc.coef=com_vals(X.coef, Y.coef)
    return gpc

def dot(X, Y):
    # TODO: need explanation
    assert(isinstance(Y, GPC))
    assert(X.pchars==Y.pchars)
    assert(X.rdim==Y.rdim)
    assert(X.rshape==Y.rshape)
    gpc=GPC(name=X.name+'*'+Y.name, pchars=X.pchars, p=X.p+Y.p, p_max=X.p_max+Y.p_max,
              rshape=(1,))
    func=lambda xi: np.einsum('ip,ip->p', X(xi), Y(xi))
    gpc.project(func=func)
    return gpc

def outer(X, Y):
    # TODO: need explanation
    assert(isinstance(Y, GPC))
    assert(X.pchars==Y.pchars)
    gpc=GPC(name=X.name+'*'+Y.name, pchars=X.pchars, p=X.p+Y.p, p_max=X.p_max+Y.p_max,
              rshape=X.rshape+Y.rshape)
    func=lambda xi: np.einsum('ip,jp->ijp', X(xi), Y(xi))
    gpc.project(func=func)
    return gpc

def element_wise(X, Y):
    # TODO: need explanation
    assert(isinstance(Y, GPC))
    assert(X.pchars==Y.pchars)
    assert(X.rshape==Y.rshape)
    gpc=GPC(name=X.name+'*'+Y.name, pchars=X.pchars, p=X.p+Y.p, p_max=X.p_max+Y.p_max,
              rshape=X.rshape)
    func=lambda xi: X(xi)*Y(xi)
    gpc.project(func=func)
    return gpc

class LinOper():

    def __init__(self, name='', val=None):
        self.name=name
        self.val=np.array(val)

    def __mul__(self, X):
        return self.__call__(X)

    def __call__(self, Y):
        assert(isinstance(Y, GPC))
#         return Y*self.val.T
        gpc=Y.copy_zeros()
        mul_str='{0}{1},a{1}->a{0}'.format(istr[:self.val.ndim-Y.rdim], ustr[:Y.rdim])
        gpc.set_coef(coef=np.einsum(mul_str, self.val, Y.coef))
        return gpc

    def __str__(self):
        ss="Class : {0}\n".format(self.__class__.__name__)
        ss+='    name   = {0}\n'.format(self.name)
        ss+='    shape  = {0}\n'.format(self.val.shape)
        return ss

    def __repr__(self):
        return self.__str__()

class Oper():

    def __init__(self, name='', fun=None):
        self.name=name
        self.fun=fun

    def __mul__(self, X):
        return self.__call__(X)

    def __call__(self, Y):
        assert(isinstance(Y, GPC))
#         return Y*self.val.T
        raise NotImplementedError()

    def __str__(self):
        ss="Class : {0}\n".format(self.__class__.__name__)
        ss+='    name   = {0}\n'.format(self.name)
        ss+='    shape  = {0}\n'.format(self.val.shape)
        return ss

    def __repr__(self):
        return self.__str__()


def com_vals(A, B):
    # combine values of A and B (indices, coefficients, etc.)
    Cmean=np.hstack([A[0], B[0]])
    Cleft=np.hstack([A[1:], np.zeros([A.shape[0]-1, B.shape[1]], dtype=A.dtype)])
    Crigh=np.hstack([np.zeros([B.shape[0]-1, A.shape[1]], dtype=A.dtype), B[1:]])
    return np.vstack([Cmean, Cleft, Crigh])

if __name__=='__main__':
    execfile('../test_gpc2.py')
#     pchars = 'p'
#     gpc = GPC(pchars, p=2)
#
#     ip, iw = Quad_Gauss('u', p=10).get_integration()
#     rv_str = poly[pchars].rv
#     from randfield import rvs
#     rv = rvs[rv_str]
#     print rv
#     print rv.fun.cdf(ip)
#     print val

    print('END')
