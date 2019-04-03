"""
This module use FFTHomPy project from https://github.com/vondrejc/FFTHomPy
"""
from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.cm import coolwarm
from mpl_toolkits.mplot3d import axes3d

import numpy as np
from ffthompy.tensors import Tensor
from ffthompy.trigpol import Grid
from ffthompy.materials import get_weights_lin, get_weights_con
from general import Struct
from uq.covfun import Matern, Expcov, Sexpcov


class KL_Fourier():
    """
    Karhunen-Loeve decomposition using Fourier transform
    """

    def __init__(self, covfun=2, cov_pars={}, N=5*np.ones(2), puc_size=None,
                 transform=lambda x: np.exp(x)):
        self.N=np.array(N, dtype=np.int)
        self.dim=self.N.size
        if puc_size is None:
            self.puc_size=np.ones(self.dim)
        else:
            self.puc_size=puc_size
        self.transform=transform

        # defining Fourier basis functions
        coef=2*np.pi*1j
        scal= lambda xi, x: np.einsum('i,i...', xi, x)
        self.bfun=[lambda xi, x: (np.exp(coef*scal(xi,x))+np.exp(coef*scal(xi,x))).real,
                   lambda xi, x: (np.exp(coef*scal(xi,x))-np.exp(coef*scal(xi,x))).imag]

        # covariances
        if callable(covfun):
            self.covfun=covfun
        elif covfun in [0]:
            def_pars={'nu':3./2}
            def_pars.update(cov_pars)
            self.covfun=Matern(**def_pars)
        elif covfun in [1]:
            self.covfun=Expcov(**cov_pars)
        elif covfun in [2]:
            self.covfun=Sexpcov(**cov_pars)
        else:
            raise ValueError("covfun in KL_Fourier")

    @staticmethod
    def distance2origin(grid_coor):
        # computes the distance to origin for grid coordinates
        d_coor=np.zeros_like(grid_coor[0])
        dim=len(grid_coor)
        for ii in range(dim):
            d_coor+=grid_coor[ii]**2
        return d_coor**0.5

    def calc_modes(self, method=1, n_kl=None, relerr=None, relval=None, debug=False):
        grid_coor=Grid.get_coordinates(self.N, self.puc_size)
        covfun = lambda grid_coor: self.covfun(self.distance2origin(grid_coor))
        ct=Tensor(name='covar', val=covfun(grid_coor), order=0, N=self.N,
                  Y=self.puc_size, Fourier=False, fft_form='c', origin='c').shift()
        ct=ct.fourier()
        assert(np.linalg.norm(ct.val.imag)<1e-12)

        if method in [-1]: # interpolation with trigpol
            ct.val=ct.val.real
        elif method in [0]: # approximation of covariance with constant funcs.
            h = self.puc_size/self.N
            Wraw = get_weights_con(h, self.N, self.puc_size)
            ct.val = np.prod(self.N)*ct.val.real*Wraw
        elif method in [1]: # approximation of covariance with bilinear funcs.
            h = self.puc_size/self.N
            Wraw = get_weights_lin(h, self.N, self.puc_size)
            ct.val = np.prod(self.N)*ct.val.real*Wraw
        print('min(ct.val)={}'.format(ct.val.min()))
        self.ct=ct
        val=np.copy(self.ct.val)
        val[ct.mean_index()]=0 # setting zero mean
        # only independent modes for symmetric
        svalD=np.sort(val.ravel())[::-1] # reordering from biggest to smallest

        # TRUNCATING THE K-L EXPANSION
        eps=5*np.finfo(float).eps # computer precision
        if n_kl is not None:
            flags=val>svalD[n_kl]+eps
        elif relval is not None:
            flags=val>svalD[0]*relval+eps
        elif relerr is not None: # not working
            cumvalrel=np.cumsum(svalD)/svalD.sum()
            threshold=svalD[np.argmax(cumvalrel>(1-relerr))]
            flags=val>=threshold-eps
        else:
            flags=val>0+eps

        assert(flags.sum() % 2 == 0) # even number of basis functions have to be taken

        ind=np.flatnonzero(flags) # indices of reduced basis
        # generating frequencies of reduced basis
        frq=Grid.get_product(Grid.get_ZNl(self.N, fft_form='c'))[:, ind]
        # making flags of basis functions
        bas_type=(frq[0]>0)
        unknown=np.where(frq[0]==0)[0]
        for ii in range(1,self.dim):
            bas_type[unknown]=frq[ii, unknown]>0
            unknown=np.where(frq[ii, unknown]==0)[0]

        print('no. of modes = '+str(ind.size))
        self.modes=Struct(val=self.ct.val.flat[ind], # Fourier coef. of modes
                          frq=frq.T, # frequencies of modes
                          bas_type=np.array(bas_type, dtype=np.int),
                          n_kl=ind.size) # no. of modes
        if debug:
            print(flags)
            print('relerr =', 1-self.modes.val.sum()/svalD.sum())
            print('no_kl =', self.modes.n_kl)

    def mode_fun(self, n, x):
        # evaluates the reduced basis functions with index n over the points x
        frq=self.modes.frq[n]
        Fcoef=self.modes.val.flat[n]
        return Fcoef*self.bfun[self.modes.bas_type[n]](frq, x)

    def kl_cov(self, x, full=False):
        # Constructs the covariance matrix using spectral decomposition
        x=np.array(x)
        Z=np.ones_like(x[0])*self.ct.mean() # preallocation
        if not full:
            for ii in range(self.modes.n_kl):
                Z+=self.mode_fun(ii, x)
        else: # fulls
            from itertools import product
            prodN=np.array(self.ct.N).prod()
            bas_type=np.ones(prodN, dtype=np.int)
            bas_type[:int(prodN/2)]=0
            for ii, frq in enumerate(product(*Grid.get_ZNl(self.N, fft_form='c'))):
                if frq==dim*(0,):
                    continue
                else:
                    pom=np.array(frq)
                    bas_type=pom[np.nonzero(pom)[0][0]]>0
                Fcoef=self.ct.val.flat[ii]
                Z+=Fcoef*self.bfun[int(bas_type)](np.array(frq), x)
        return Z.real

    def plot_mode(self, n):
        coord=Grid.get_coordinates(5*self.N, self.puc_size)
        Zpl=self.mode_fun(n, coord)

        fig=pl.figure(0)
        ax=fig.add_subplot(111, projection='3d')
        ax.plot_surface(coord[0], coord[1], Zpl, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        pl.show()

    def plot_cov(self, full=False, print_coef=3):
        coord=Grid.get_coordinates(print_coef*self.N, self.puc_size)
        Zpl=self.covfun(self.distance2origin(coord))

        dim=self.N.size
        fig=pl.figure(0)
        if dim==1:
            pl.plot(coord[0], Zpl, 'k-', label='cov function')
            pl.plot(coord[0], self.kl_cov(coord, full=True), 'b--', label='app. cov-fun full')
            pl.plot(coord[0], self.kl_cov(coord, full=False), 'r-.', label='app. cov-fun')
            pl.legend(loc='best')
        elif dim==2:
            ax=fig.add_subplot(111, projection='3d')
            ax.plot_surface(coord[0], coord[1], Zpl, rstride=1, cstride=1,
                            cmap=coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_title('Original covariance')

            if full:
                fig=pl.figure(1)
                Zpl2=self.kl_cov(coord, full=True) # reconstruction from KL modes
                ax=fig.add_subplot(111, projection='3d')
                ax.plot_surface(coord[0], coord[1], Zpl2, rstride=1, cstride=1,
                                cmap=coolwarm, linewidth=0, antialiased=False)
                ax.set_xlabel(r'$x_1$')
                ax.set_ylabel(r'$x_2$')
                ax.set_title('Covariance full approx.')

            fig=pl.figure(2)
            Zpl2=self.kl_cov(coord, full=False) # reconstruction from KL modes
            ax=fig.add_subplot(111, projection='3d')
            ax.plot_surface(coord[0], coord[1], Zpl2, rstride=1, cstride=1,
                            cmap=coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_title(r'Covariance reduced approx.')
        pl.show()

if __name__=='__main__':
    dim=2
    kl=KL_Fourier(covfun=1, cov_pars={'rho':0.15}, N=5*3**3*np.ones(dim, dtype=np.int),
                  puc_size=np.ones(dim))
#     kl.calc_modes(relval=1e-1)
    kl.calc_modes(relerr=0.1)
    if dim==2:
    #    kl.calc_modes()
    #    kl.plot_mode(n=15)
#         kl.plot_cov(full=False)
        kl.plot_cov(full=True)
    print('END')
