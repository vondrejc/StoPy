import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as pl
import sys


class Stochastic_solver():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def collocation(self, u, Sfun=None, quad=None, ip=None, iw=None, debug=False):
        if Sfun is None:
            Sfun = self.Sfun
        if quad is not None:
            ip, iw = quad.get_integration()
        elif ip is None and iw is None:
            ip, iw = self.quad.get_integration()

        up = Sfun(ip)

        W = u.eval_all_basis(ip=ip)
        print 'collocation matrix shape = ', W.shape
        if W.shape[0] == W.shape[1]:
            alpha = np.linalg.solve(W, up)
            u.alpha = alpha
        elif W.shape[0] > W.shape[1]:
            alpha, res, rank, s = np.linalg.lstsq(W, up.T)
            u.alpha = alpha
        else:
            raise ValueError('Underdetermined system ({0},{1})!'.format(*W.shape))
        if debug:
            print 'cond of W:', np.linalg.cond(W)
        return u

    def projection(self, u, Sfun=None, quad=None):
        if Sfun is None:
            Sfun = self.Sfun
        if quad is None:
            quad = self.quad

        u.project(Sfun, quad=quad)

        return u
