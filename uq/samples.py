from general.base import Struct
import matplotlib.pyplot as pl
import numpy as np
import scipy.stats as st


class Samples(Struct):
    def __init__(self, val, name='samples'):
        self.name=name
        self.val=np.atleast_2d(val)
        (self.dim, self.n)=self.val.shape
        self.calc_stats()

    def __getitem__(self, ind):
        return self.val[ind]

    def get_subsamples(self, ind):
        return Samples(self.val[ind])

    def calc_stats(self):
        self.mean=np.mean(self.val, axis=1)
        self.kvar_diag=np.zeros_like(self.mean)
        self.var_diag=np.zeros_like(self.mean)
        self.min=np.zeros_like(self.mean)
        self.max=np.zeros_like(self.mean)
        for ii in range(self.dim):
            self.kvar_diag[ii]=np.sum((self.val[ii]-self.mean[ii])**2)/(self.n-1)
            self.var_diag[ii]=np.mean((self.val[ii]-self.mean[ii])**2)
            self.min[ii]=self.val[ii].min()
            self.max[ii]=self.val[ii].max()

    def get_stats(self, moment=1):
        if moment in [1, 'mean']:
            return self.mean
        elif moment in [2, 'var']:
            return self.var

    def percentile(self, q):
        return np.percentile(self.val, q, axis=1)

    def cov(self, **kwargs):
        return np.cov(self.val, rowvar=True, **kwargs)

    def map(self, fun, fun_name='F'):
        return Samples(val=fun(self.val), name='{0}({1})'.format(fun_name, self.name))

    def pdf(self, qs, bw_method=None):
        kernel = st.gaussian_kde(self.val, bw_method=bw_method)
        return kernel(qs)

    def plot(self):
        if self.dim>2:
            raise NotImplementedError()
        elif self.dim==1:
            pl.figure()
            pl.hist(self.val[0], bins=100, normed=True)
#             pl.plot(self.val[0], np.zeros(self.n), 'x', markersize=1)
            pl.xlabel('Variable with index 0')
            pl.ylabel('Density')
            pl.show()
        elif self.dim==2:
            pl.figure()
            pl.plot(self.val[0], self.val[1], 'o', markersize=1.)
            pl.xlabel('Variable with index 0')
            pl.ylabel('Variable with index 1')
            pl.show()

    def __str__(self, full=False):
        ss="\nClass : {}\n".format(self.__class__.__name__)
        ss+='    name = {}\n'.format(self.name)
        ss+='    dim  = {}\n'.format(self.dim)
        ss+='    n    = {0:e}\n'.format(self.n)
        ss+='    mean = {}\n'.format(self.mean)
        ss+='    var  = {}\n'.format(self.var_diag)
        ss+='    kvar = {}\n'.format(self.kvar_diag)
        ss+='    min  = {}\n'.format(self.min)
        ss+='    max  = {}\n'.format(self.max)
        return ss

class MapSample():
    def __init__(self, fun, name='F'):
        self.fun=fun
        self.name=name

    def __call__(self, smpl):
        assert(isinstance(smpl, Samples))
        return self._call(self.fun, smpl.vals,
                          name='{0}({1})'.format(self.name, smpl.name))

    @staticmethod
    def _call(fun, vals, name=''):
        return Samples(val=fun(vals), name=name)


if __name__=='__main__':
    import scipy.stats as st
    n=1e4
    val=np.vstack([(np.random.randn(n)+1.)/0.5,
                     np.random.random(n),
                     np.random.random(n)])

    smpl=Samples(val)
#     smpl.calc_stats()
    print(smpl)
    smpl[[0, 1]].plot()
#     print(smpl[0])
#     print(smpl[1])

    Fval=np.fft.fft(smpl.val, axis=1)/smpl.n
    print('mean')
    print(smpl.mean)
    print(Fval[0].shape)
    print(Fval[:, 0])
    print('var')
    print(smpl.var)
#     var = np.real(4*Fval[:, 1]*np.conj(Fval[:, 1]))
#     print(var)
#     print(smpl.var/var)
    print(Fval[:, 1])
    print('end calc_stats')

#     smpl[[0, 2]].plot()
    import sys
    sys.exit()
    print('--mean--')
    mean=smpl[0].mean
    print(smpl[0].mean)
    print(st.kstat(smpl.val[0], n=1))
    print(st.moment(smpl.val[0], moment=1, axis=None))
#     print(smpl[0])
    print('--variance--')
    print(st.kstat(smpl.val[0], n=2))
    print(np.sum((smpl.val[0]-mean)**2)/(smpl.n-1))
    print(st.moment(smpl.val[0], moment=2, axis=None))
    print(np.mean((smpl.val[0]-mean)**2))
    mean=np.mean(smpl.val[1])
    print(st.kstat(smpl.val[1], n=2))
    print(np.sum((smpl.val[1]-mean)**2)/(smpl.n-1))
    print(st.moment(smpl.val[1], moment=2, axis=None))
    print(np.mean((smpl.val[1]-mean)**2))

#     print(np.mean((smpl[0])**2))
    print('END')
