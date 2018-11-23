import itertools
from numpy.linalg import norm
import scipy
import unittest

import numpy as np
import scipy.stats as stats
from uq.dist import Dists
from uq.gpc import GPC
from uq.quadrature import Quad_Gauss
from uq_trial.quad_sparse import generator


class Test(unittest.TestCase):

    def test_gpc(self):
        print('Testing uq.gpc...')
        # testing projection
        for gpc in [GPC('ut', p=2*[4], p_max=4),
                    GPC('pu', p=2*[2]),
                    GPC('hu', p=2*[2])]:
            self.assertTrue(isinstance(gpc.__str__(), str))
            func=lambda x: 100.*x[0]**2*x[1]**2
            gpc.project(func)
            x=np.random.random([2, 5])
            val=norm(func(x)-gpc(x))
            self.assertAlmostEqual(0, val, msg='GPC', delta=1e-12)
            self.assertTrue(isinstance(gpc.__str__(), str))
        for gpc in [GPC('ptu', p=3*[4], p_max=7),
                    GPC('tuh', p=3*[4], p_max=7)]:
            func=lambda x: 100.*x[0]**2*x[1]**3*x[2]**2
            gpc.project(func)
            x=np.random.random([3, 5])
            val=norm(func(x)-gpc(x))
            self.assertAlmostEqual(0, val, msg='GPC', delta=1e-12)
            self.assertTrue(isinstance(gpc.__str__(), str))
        # testing approximation of normal distribution with Hermite polynomials
        dist=stats.norm(loc=1., scale=2.)
        gpc=GPC(pchars='h', p=1)
        gpc.expand(dist, quad={'p_int': 10})
        dist_st=np.array(dist.stats())
        gpc_st=gpc.stats()
        for ii in range(2):
            self.assertAlmostEqual(dist_st[ii], gpc_st[ii], delta=1e-12)
        self.assertTrue(isinstance(gpc.__str__(), str))
        print('...OK')

    def test_sparse_integration(self):
        print('Testing uq.quadrature (sparse integration)...')
        # testing projection
        for gpc in [GPC('ptu', p=3*[4], p_max=7),
                    GPC('tuh', p=3*[4], p_max=7)]:
            func=lambda x: 100.*x[0]**2*x[1]**3*x[2]**2
            quad=Quad_Gauss(pchars=gpc.pchars, p_int=7, name='Gauss', grid='tensor')
            gpc.project(func, quad=quad)
            x=np.random.random([3, 5])
            val=norm(func(x)-gpc(x))
            self.assertAlmostEqual(0, val, msg='GPC', delta=1e-12)

        for f in [lambda x: 1.,
                  lambda x: x[0]**2*x[1]**2*x[2]**2+4*x[1]**2*x[2]**2+x[0]**2]:

            def integral(ip, iw):
                summ=0
                for i in range(len(ip)):
                    summ+=f(ip[i])*iw[i]
                return summ

            dim=4
            quad=Quad_Gauss(pchars=dim*'p', p_int=dim*[3], name='test_fig', grid='tensor')
            ipf, iwf=quad.get_integration()
            ipf=ipf.T

            ips, iws=generator(3, dim, "Legendre")
            self.assertAlmostEqual(integral(ipf, iwf), integral(ips, iws), msg='quad_sparse',
                                   delta=1e-10)
        print('...OK')

    def test_quad(self):
        print('Testing uq.quadrature....')
        for quad in [Quad_Gauss('pth', p_int=[2, 3, 4]),
                     Quad_Gauss('pth', p_int=3)]:
            self.assertTrue(isinstance(quad.__str__(), str))
            ip, iw=quad.get_integration()
            self.assertEqual(len(quad.pchars), ip.shape[0], msg=quad.__str__())
            self.assertEqual(iw.size, ip.shape[1], msg=quad.__str__())

        for pchar, p_int in itertools.product(['p', 't', 'u', 'h'], range(1, 100)):
            quad=Quad_Gauss(pchars=pchar, p_int=p_int, name='test_integration', grid='tensor')
            ip, iw=quad.get_integration()
            self.assertAlmostEqual(1., np.sum(iw), msg='Quad_Gauss', delta=1e-15)
        print('...OK')

    def test_dist(self):
        print('Testing uq.dists, and uq.samples...')
        loc=np.array([1., 2.])
        var=np.array([4., .1])
        dist1=Dists(name='n1', keys=['norm'], args=[()],
                      kargs=[{'loc': loc[0], 'scale': var[0]**0.5}])
        dist2=Dists(name='n2', keys=['norm'], args=[()],
                      kargs=[{'loc': loc[1], 'scale': var[1]**0.5}])
        dist=dist1*dist2
        stats=dist.stats()
        self.assertAlmostEqual(0, norm(stats[:, 0]-loc), msg='Dists')
        self.assertAlmostEqual(0, norm(stats[:, 1]-var), msg='Dists')
        self.assertAlmostEqual(0, norm(stats[:, 2]-np.zeros(2)), msg='Dists')
        self.assertAlmostEqual(0, norm(stats[:, 3]-np.zeros(2)), msg='Dists')

        gpc1=GPC('h', p=2, name='gpc1').expand(dist1, quad={'p_int': 5})
        gpc2=GPC('h', p=2, name='gpc2').expand(dist2, quad={'p_int': 5})
        gpc=gpc1.combine(gpc2)
        gpcB=dist.to_gpc(name='gpcB', pchars=gpc.pchars, p=gpc.p)

        self.assertAlmostEqual(0, gpc==gpcB, delta=1e-13)
        self.assertAlmostEqual(0, norm(gpc.mean()-loc), msg='GPC')
        self.assertAlmostEqual(0, norm(gpc.moment(1)-loc), msg='GPC')
        self.assertAlmostEqual(0, norm(gpc.covariance()-np.diag(var)), msg='GPC')
        smpl_dist=dist.samples(n=1e3)
        smpl_gpc=gpc.samples(n=1e3)
        self.assertTrue(isinstance(smpl_dist.__str__(), str))
        self.assertTrue(isinstance(smpl_gpc.__str__(), str))
        self.assertTrue(isinstance(dist.__str__(), str))
        print('...OK')

#    def test_samples(self):
#        print 'Testing uq.samples'
#        print '...OK'

if __name__=="__main__":
    unittest.main()
