import warnings

from general.base import Struct
import matplotlib.pylab as pl
import numpy as np
import scipy.stats as st
from uq.gpc import GPC
from uq.samples import Samples


rv_list={
    'arcsine':      Struct(name='arcsine',
                           fun=st.arcsine,
                           supp=np.array([0, 1], dtype=np.float)),
    'beta':         Struct(name='beta',
                           fun=st.beta,
                           supp=np.array([0, 1], dtype=np.float)),
    'norm':         Struct(name='norm', # gaussian
                           fun=st.norm,
                           supp=np.array([-np.inf, np.inf], dtype=np.float)),
    'semicircular': Struct(name='semicircular',
                           fun=st.semicircular,
                           supp=np.array([-1, 1], dtype=np.float)),
    'uniform':      Struct(name='uniform',
                           fun=st.uniform,
                           supp=np.array([0, 1], dtype=np.float)),
           }


class Dists(Struct):
    """
    Joint distribution
    """
    def __init__(self, name='', keys=[], args=[], kargs=[]):
        self.name=name
        self.dim=keys.__len__()
        self.keys=keys
        self.args=args
        self.kargs=kargs
        self.dists=[]
        self.supp=[]

        for ii, (key, arg, karg) in enumerate(zip(keys, args, kargs)):
            if key in ['norm']:
                if 'mean' in karg:
                    karg['loc']=karg['mean']
                    karg.pop('mean')
                if 'var' in karg:
                    karg['scale']=karg['var']**0.5
                    karg.pop('var')

                self.dists.append(rv_list[key].fun(*arg, **karg))
                self.supp.append(self.map_supp(rv_list[key].supp, **karg))
            else:
                self.dists.append(rv_list[key].fun(*arg, **karg))
                self.supp.append(self.map_supp(rv_list[key].supp, **karg))
        vals=self.stats(moments='mv')
        self.mean=vals[:, 0]
        self.var=vals[:, 1]

    @staticmethod
    def map_supp(supp, scale=1., loc=0.):
            return supp*scale+loc

    @staticmethod
    def get_loc_scale(dist_name, supp):
        raise NotImplementedError()

    def __getitem__(self, ind):
        if isinstance(ind, int):
            ind=[ind]
        try:
            return Dists(keys=[self.keys[ii] for ii in ind],
                         args=[self.args[ii] for ii in ind],
                         kargs=[self.kargs[ii] for ii in ind])
        except:
            raise ValueError('Wrong index (%s)!'% str(ind))

    def __mul__(self, y):
        x=self
        return Dists(name=x.name+'*'+y.name, keys=x.keys+y.keys,
                     args=x.args+y.args, kargs=x.kargs+y.kargs)

    def samples(self, n):
        n=int(n)
        val=np.empty([self.dim, n])
        for ii, dist in enumerate(self.dists):
            val[ii]=dist.rvs(n)
        return Samples(val, name='Dists:{0}'.format(self.name))

    def cdf(self, x):
        x=np.atleast_2d(np.array(x))
        val=np.ones(x.shape[1])
        for ii in range(self.dim):
            val*=self.dists[ii].cdf(x[ii])
        return val

    def pdf(self, x):
        if np.array(x).ndim==1:
            x=np.atleast_2d(np.array(x).T)
        val=np.ones(x.shape[1])
        for ii in range(self.dim):
            val*=self.dists[ii].pdf(x[ii])
        return val

    def ppf(self, x):
        assert(self.dim==1)
        return self.dists[0].ppf(x)

    def stats(self, moments='mvsk'):
        val=np.empty([self.dim, len(moments)])
        for ii in range(self.dim):
            val[ii]=np.array(self.dists[ii].stats(moments=moments))
        return val

    def moment(self, mm=0):
        val=np.empty(self.dim)
        for ii in range(self.dim):
            val[ii]=self.dists[ii].moment(mm)
        return val

    def percentile(self, q):
        val=np.empty(self.dim)
        for ii in range(self.dim):
            val[ii]=self.dists[ii].ppf(q)
        return val

    def to_gpc(self, name=None, pchars=None, p=None):
        assert(len(pchars)==self.dim)
        assert(np.array(p).size==self.dim)
        gpc=GPC(pchars=pchars[0], p=p[0]).expand(self.dists[0])
        for ii in range(1, self.dim):
            gpc_next=GPC(pchars=pchars[ii], p=p[ii]).expand(self.dists[ii])
            gpc=gpc.combine(gpc_next)

        if name is None:
            name=self.name
        gpc.name=name
        return gpc

    def plot(self, val='cdf', figtype=0):

        def plot_range(ii):
            supp=self.supp[ii]
            smpl=self[ii].get_samples(1e2)
            if supp[0]==-np.inf:
                supp[0]=smpl.min
            if supp[1]==np.inf:
                supp[1]=smpl.max
            dst=supp[1]-supp[0]
            return (supp[0]-dst*0.1, supp[1]+dst*0.1)

        if self.dim==1:
            pl.figure()
            x=np.linspace(*plot_range(0), num=1e3)
            pl.plot(x, self.dists[0].pdf(x), label="pdf")
            pl.plot(x, self.dists[0].cdf(x), label="cdf")
            pl.legend(loc='best')
            pl.show()

        elif self.dim==2:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.cm import coolwarm, RdBu
            from matplotlib.ticker import LinearLocator, FormatStrFormatter

            n=1e2
            X=np.linspace(*plot_range(0), num=n)
            Y=np.linspace(*plot_range(1), num=n)
            X, Y=np.meshgrid(X, Y)
            x=np.vstack([X.flatten(), Y.flatten()])
            fun=getattr(self, val)
            Z=np.reshape(fun(x), X.shape)

            fig=pl.figure()
            if figtype in [1]:
                ax=fig.gca(projection='3d')
                surf=ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                       cmap=coolwarm,
                                       linewidth=0, antialiased=False)
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                fig.colorbar(surf, shrink=0.5, aspect=5)
            elif figtype in [0]:
                ax=fig.gca(projection='3d')
                cmap=RdBu
                ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3, cmap=cmap)
                cset=ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cmap)
                cset=ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cmap)
                cset=ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cmap)
            elif figtype in [2]:
                CS=pl.contourf(X, Y, Z, interpolation='none')
                pl.colorbar(CS)
            pl.xlabel(r'$\xi_0$')
            pl.ylabel(r'$\xi_1$')
            pl.show()
        else:
            warnings.warn("Plotting is impossible for dim(%d)>2."%self.dim)

    def __str__(self, full=False, detailed=False):
        ss="Class : {0}\n".format(self.__class__.__name__)
        ss+='    name  = {0}\n'.format(self.name)
        ss+='    dim   = {0}\n'.format(self.dim)
        ss+='    keys  = {0}\n'.format(self.keys)
        ss+='    args  = {0}\n'.format(self.args)
        ss+='    kargs = {0}\n'.format(self.kargs)
        ss+='    supp  = {0}\n'.format(self.supp)
        ss+='    mean  = {0}\n'.format(self.mean)
        ss+='    var   = {0}\n'.format(self.var)
        ss+='    moment(1)= {0} (non-centered)\n'.format(self.moment(1))
        ss+='    moment(2)= {0} (non-centered)\n'.format(self.moment(2))
        return ss


if __name__=='__main__':
    args=(1.1, 1.1)
#     rv = st.beta(*(1.2, 2.), loc=.5, scale=4.5)
#     rv = st.beta(*(1.2, 2.), **{'loc': .5, 'scale': 4.5})
#     rv = st.uniform(loc=1., scale=2.) # p: Lagrange
#     rv = st.arcsine(loc=1., scale=2.) # t: Chebyshev 1st
#     rv = st.semicircular(loc=1., scale=1.) # u: Chebyshev 2st
#     rv = st.beta(a=1.5, b=1.5, loc=0., scale=2.)
#     rv = st.norm(loc=1., scale=1.) # h: Hermite

    rvs=Dists(keys=['beta', 'beta'],
                args=[(1.2, 2.), (0.7, .6)],
                kargs=[{'loc': .5, 'scale': 4.5}, {'loc': 1., 'scale': 6.}])

#     rvs = Dists(keys=['beta', 'norm'],
#                 args=[(1.2, 2.), ()],
#                 kargs=[{'loc': .5, 'scale': 4.5}, {'loc': 1., 'scale': 6.}])
    print(rvs)
#     rvs[0].plot()
#     rvs[1].plot()
    rvs.plot(val='cdf', figtype=0)
