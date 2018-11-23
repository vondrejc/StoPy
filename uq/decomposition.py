"""
This module use FFTHomPy project from https://github.com/vondrejc/FFTHomPy
"""
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
    def __init__(self, covfun=2, cov_pars={}, N=5*np.ones(2), puc_size=np.ones(2),
                 transform=lambda x: np.exp(x)):
        self.N=np.array(N, dtype=np.int)
        self.dim=self.N.size
        self.puc_size=puc_size # Y
        self.transform=transform

        # covariances
        if covfun in [0]:
            def_pars={'nu':3./2}
            def_pars.update(cov_pars)
            self.covfun=Matern(**def_pars)
        elif covfun in [1]:
            self.covfun=Expcov(**cov_pars)
        elif covfun in [2]:
            self.covfun=Sexpcov(**cov_pars)
        else:
            self.covfun=covfun

    @staticmethod
    def distance2origin(grid_coor):
        d_coor=np.zeros_like(grid_coor[0])
        dim=len(grid_coor)
        for ii in range(dim):
            d_coor+=grid_coor[ii]**2
        return d_coor**0.5

    def calc_modes(self, method=1, n_kl=None, relerr=None, relval=None,
                   debug=False):
        eps=5*np.finfo(float).eps
        grid_coor=Grid.get_coordinates(self.N, self.puc_size)
        covfun = lambda grid_coor: self.covfun(self.distance2origin(grid_coor))
        ct=Tensor(name='covar', val=covfun(grid_coor), order=0, N=self.N,
                  Y=self.puc_size, Fourier=False, fft_form='c', origin='c').shift()
        ct=ct.fourier()
        ct.val = ct.val.real

        if method in [-1]: # interpolation with trigpol
            pass
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
        svalD=np.sort(val.ravel())[::-1] # reordering
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

        bfun=lambda xi, x: np.cos(2*np.pi*np.einsum('i,i...', xi, x))

        ind=np.flatnonzero(flags)
        xi=Grid.get_xil(self.N, self.puc_size, fft_form='c')
        xis=Grid.get_product(xi)
        print('no. of modes = '+str(ind.size))
        self.modes=Struct(val=self.ct.val.real.flat[ind], # coef. of modes
                          xis=xis[:, ind].T, # frequencies of modes
                          n_kl=ind.size, # no. of modes
                          fun=bfun) # basis funs for modes
        if debug:
            print(flags)
            print('relerr =', 1-self.modes.val.sum()/svalD.sum())
            print('no_kl =', self.modes.n_kl)

#     def eval_mode(self):
#         return val

    def get_k(self, n, N):
        k=np.zeros(self.dim)
        k[0]=n % N[0]
        k[1]=n - k[0]*N[0]
        return k

    def mode_fun(self, n, x):
        xi=self.modes.xis[n]
        Fcoef=self.modes.val.flat[n]
        return Fcoef*self.modes.fun(xi, x)

    def kl_cov(self, x, full=False):
        x=np.array(x)
        if not full:
            Z=np.ones_like(x[0])*self.ct.mean()
            for ii in range(self.modes.n_kl):
                Z+=self.mode_fun(ii, x)
        else: # fulls
            xis=Grid.get_xil(self.N, self.puc_size)
            from itertools import product
            Z=np.zeros_like(x[0], dtype=np.complex)
            for ii, xi in enumerate(product(*xis)):
                Fcoef=self.ct.val.flat[ii]
                Z+=Fcoef*np.cos(2*np.pi*np.einsum('i,i...', xi, x))
        return Z.real

    def plot_mode(self, n):
        coord=Grid.get_coordinates(5*self.N, self.puc_size)
        Zpl=self.mode_fun(n, coord)

        fig=pl.figure(0)
        ax=fig.add_subplot(111, projection='3d')
        ax.plot_surface(coord[0], coord[1], Zpl, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        pl.show()

    def plot_cov(self, full=False):
        coord=Grid.get_coordinates(3*self.N, self.puc_size)
        Zpl=self.covfun(self.distance2origin(coord))

        dim=self.N.size
        fig=pl.figure(0)
        if dim==1:
            pl.plot(coord[0], Zpl, label='cov_function')
            pl.plot(coord[0], self.kl_cov(coord, full=True), label='app. cov-fun full')
            pl.plot(coord[0], self.kl_cov(coord, full=False), label='app. cov-fun')
            pl.legend(loc='best')
        elif dim==2:
            ax=fig.add_subplot(111, projection='3d')
            ax.plot_surface(coord[0], coord[1], Zpl, rstride=1, cstride=1,
                            cmap=coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')

            if full:
                fig=pl.figure(1)
                Zpl2=self.kl_cov(coord, full=True) # reconstruction from KL modes
                ax=fig.add_subplot(111, projection='3d')
                ax.plot_surface(coord[0], coord[1], Zpl2, rstride=1, cstride=1,
                                cmap=coolwarm, linewidth=0, antialiased=False)
                ax.set_xlabel('$x_1$')
                ax.set_ylabel('$x_2$')

            fig=pl.figure(2)
            Zpl2=self.kl_cov(coord, full=False) # reconstruction from KL modes
            ax=fig.add_subplot(111, projection='3d')
            ax.plot_surface(coord[0], coord[1], Zpl2, rstride=1, cstride=1,
                            cmap=coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
        pl.show()

if __name__=='__main__':
    dim=2
    kl=KL_Fourier(covfun=2, cov_pars={'rho':0.15}, N=11*np.ones(dim, dtype=np.int),
                  puc_size=np.ones(dim))
#     kl.calc_modes(relval=1e-1)
    kl.calc_modes(relerr=0.1)
    if dim==2:
    #    kl.calc_modes()
    #    kl.plot_mode(n=15)
        kl.plot_cov(full=False)
    #    kl.plot_cov(full=True)
    print('END')
