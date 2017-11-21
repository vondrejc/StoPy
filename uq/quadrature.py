import warnings

from general.base import Struct
import numpy as np
from uq.operators import tensor_product
from uq.polynomials import poly, GPCpoly


class Quad(Struct):
    def __mul__(self, quad):
        return Quad_product([self, quad])

    def __str__(self):
        ss="Class : {0}\n".format(self.__class__.__name__)
        ss+='    name = {0}\n'.format(self.name)
        ss+='    dim = {0}\n'.format(self.dim)
        ss+='    p_int = {0}\n'.format(self.p_int)
        ss+='    grid = {0}\n'.format(self.grid)
        ss+='    no. of points = {0}\n'.format(self.get_npoints())
        return ss

    def __repr__(self):
        return self.__str__()


class Quad_Gauss(Quad):
    def __init__(self, pchars, p_int=0, name='', grid='tensor'):
        self.name=name
        self.pchars=pchars
        self.dim=self.pchars.__len__()
        p_int=np.atleast_1d(np.array(p_int, dtype=np.int))
        if grid=='tensor' and p_int.size==1:
            p_int=p_int[0]*np.ones(self.dim, dtype=np.int)
        elif grid=='smolyak':
            raise NotImplementedError('Smolyak integration.')
        self.p_int=p_int
        self.grid=grid

    @staticmethod
    def cartesian_product(lst):
        dim=len(lst)
        lstg=np.atleast_2d(lst[0])
        for ii in range(1, dim):
            lstg_new=np.tile(lst[ii], lstg.shape[1])
            lstg_old=np.repeat(lstg, lst[ii].size, axis=1)
            lstg=np.vstack([lstg_old, lstg_new])
        return lstg

    def get_integration(self):
        if self.p_int.max()>100:
            warnings.warn('Numpy: integration is not tested for p > 100!')

        if self.grid in ['tensor']:
            ips=[]
            iws=[]
            for ii, method in enumerate(self.pchars):
                ip, iw=poly[method].integ(self.p_int[ii])
                ips.append(ip)
                iws.append(iw)
            ipg=self.cartesian_product(ips)
            iwg=np.prod(self.cartesian_product(iws), axis=0)
        elif self.grid.lower() in ['sparse']:
            pass
        else:
            raise NotImplementedError('integration grid ({})'.format(self.grid))
        return ipg, iwg

    def get_npoints(self):
        # return no. of points
        return np.prod(self.p_int)


class Quad_rectangular(Quad):
    def __init__(self, pchars='', p_int=1e6, name='', grid='tensor'):
        self.name=name
        self.pchars=pchars
        self.grid=grid
        dim=pchars.__len__()
        self.dim=dim
        p_int=np.atleast_1d(np.array(p_int, dtype=np.int))
        if p_int.size<dim:
            self.p_int=np.array(np.round(p_int**(1./dim))*np.ones(dim), dtype=np.int)
        elif p_int.size==dim:
            self.p_int=np.array(p_int, dtype=np.int)
        else:
            msg='not matching pchars ({0}) and n({1})'.format(pchars, p_int)
            raise ValueError(msg)

    def get_integration(self):
        h=1./self.p_int
        if self.grid in ['tensor']:
            poldist=GPCpoly.get_dist(self.pchars)
            Ulin=np.linspace(0+h[0]/2, 1.-h[0]/2, self.p_int[0])
            ip=np.atleast_2d(poldist[0].ppf(Ulin))
            for ii in range(1, self.pchars.__len__()):
                Ulin=np.linspace(0+h[ii]/2, 1.-h[ii]/2, self.p_int[ii])
                ip_new=np.tile(poldist[ii].ppf(Ulin), ip.shape[1])
                ip_old=np.repeat(ip, self.p_int[ii], axis=1)
                ip=np.vstack([ip_old, ip_new])
            iw=np.ones(np.prod(self.p_int))/np.prod(self.p_int)
        else:
            raise NotImplementedError('self.grid ({0})'.format(self.grid))
        return ip, iw

    def get_npoints(self):
        # return no. of points
        return np.prod(self.p_int)

class Quad_MC(Quad):
    def __init__(self, q, p_int, name=''):
        self.name=name
        self.q = q # GPC
        self.p_int = int(p_int)

    def get_integration(self):
        smpl = self.q.samples(n=self.p_int)
        return smpl.val, np.ones(smpl.n, dtype=np.float)/smpl.n

    def get_npoints(self):
        return self.p_int

    def __str__(self):
        ss="Class : {0}\n".format(self.__class__.__name__)
        ss+='    name = {0}\n'.format(self.name)
        ss+='    dim = {0}\n'.format(self.q.dim)
        ss+='    no. of points = {0}\n'.format(self.get_npoints())
        return ss


class Quad_product(Quad):
    def __init__(self, quads):
        self.name=quads[0].name
        for quad in quads[1:]:
            self.name+='*'+quad.name
        self.n=quads.__len__()
        self.quads=quads
        self.dim=0
        grid=''
        self.p_int=np.array([])
        for quad in quads:
            self.dim+=quad.dim
            self.p_int=np.hstack((self.p_int, quad.p_int))
            grid+='*'+quad.grid
        self.grid=grid[1:]

    def get_integration(self):
        ips, iws=self.quads[0].get_integration()
        for quad in self.quads[1:]:
            ip, iw=quad.get_integration()
            ips=tensor_product(ips, ip)
            iws=np.prod(tensor_product(iws, iw), axis=0)
        return ips, iws

    def get_npoints(self):
        # return no. of points
        val=1
        for quad in self.quads:
            val*=quad.get_npoints()
        return val

# class Quad_NewtonCotes():
#     """
#     Experimental Newton-Cotes inegration in 1D.
#     """
#     def __init__(self, box=None, pint=[1], ns=None, intervals=None, grid='tensor'):
#         dim=1
#         if len(pint)==1:
#             pint=dim*pint
#
#         if intervals is None and box is None :
#             box=np.tile(np.array([0, 1]), (dim, 1))
#
#         def get_points_weights(intr, order):
#             if order==1:
#                 points=intr
#                 weights=
#             elif order==2:
#                 points=np.zeros((intr.size-1)*order+1)
#                 points[0:points.size:order]=intr
#             else:
#                 raise NotImplementedError('order ({})'.format(order))
#             return points, weights
#
#         intervals=[]
#         weights=[]
#         points=[]
#         for d in range(dim):
#             points, weights=get_points_weights(intr=np.linspace(box[d, 0], box[d, 1], ns[d]),
#                                                order=pint[d])
#
#     def get_integration(self):
#         pass
#
#     def get_npoints(self):
#         pass


if __name__=='__main__':
#    execfile('unittests.py')
#    for p_int in [3]:
#        print '-- %d --------' % p_int
#        ip, iw = Quad_Gauss('pt', p_int).get_integration()
#     for n in [10]:
#         ip, iw=Quad_rectangular('pp', n=1e2).get_integration()
#     quad=Quad_NewtonCotes(pint=[2], ns=[3])
    print('END')

#     print('points\n', ip)
#     print(ip.shape)
#     print('weights\n', iw)
#     print(iw.shape)
