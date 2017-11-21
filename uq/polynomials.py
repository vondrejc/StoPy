import numpy as np
import numpy.polynomial as nppol
import scipy.special as spec
import scipy.stats as stats


class Legendre():

    def __init__(self):
        self.rv_name='uniform'
        self.syschar='p'
        self.rv_fun=stats.uniform(loc=-1., scale=2.)
        self.supp=np.array([-1, 1], dtype=np.float64)
        self.eval=spec.eval_legendre
        self.poly=nppol.Legendre
        self.norm=lambda p: (1./(2*p+1))**0.5
        self.weight=nppol.legendre.legweight

    def integ(self, n):
        ip, iw=nppol.legendre.leggauss(n)
        return ip, iw/2


class Chebyshev_1nd():
    def __init__(self):
        self.name='Chebyshev 1nd'
        self.syschar='t'
        self.rv_name='arcsine'
        self.rv_fun=stats.arcsine(loc=-1., scale=2.)
        self.supp=np.array([-1, 1], dtype=np.float64)
        self.eval=spec.eval_chebyt
        self.poly=nppol.Chebyshev
        self.norm2=lambda p: 1./2+1./2*(p==0)
        self.norm=lambda p: (1./2+1./2*(p==0))**0.5
        # weight=lambda x: nppol.chebyshev.chebweight(x)/np.pi,
        self.weight=lambda x: 1./np.pi/(1-x**2)**0.5

    def integ(self, n):
        ip, iw=nppol.chebyshev.chebgauss(n)
        return ip, iw/np.pi


class Chebyshev_2nd():
    def __init__(self):
        self.name='Chebyshev 2nd'
        self.syschar='u'
        self.rv_name='semicircular'
        self.rv_fun=stats.beta(a=1.5, b=1.5, loc=-1., scale=2.)
        # rv_fun=stats.semicircular(loc=0., scale=1.),
        self.supp=np.array([-1, 1], dtype=np.float64)
        self.eval=spec.eval_chebyu
        self.norm=lambda p: 1.
        self.weight=lambda x: 2./np.pi*(1-x**2)**0.5

    def integ(self, n):
        kk=np.arange(n)
        ii=kk+1
        ip=np.cos(np.pi*ii/(n+1))
        iw=np.pi/(n+1)*np.sin(np.pi*ii/(n+1))**2*2./np.pi
        return ip, iw


class Hermite():
    def __init__(self):
        self.name='Hermite'
        self.syschar='h'
        self.rv_name='norm' # gaussian distribution
        self.rv_fun=stats.norm(loc=0., scale=1.)
        self.supp=np.array([-np.inf, np.inf])
        self.eval=spec.eval_hermitenorm
#        self.polybas=spec.hermitenorm
        self.poly=nppol.HermiteE
        self.norm=lambda p: (np.math.factorial(p))**0.5
        # weight=lambda x: nppol.hermite_e.hermeweight(x)/(2*np.pi)**0.5,
        self.weight=lambda x: np.exp(-x**2/2)/(2*np.pi)**0.5

    def integ(self, n):
        ip, iw=np.polynomial.hermite_e.hermegauss(n)
        return ip, iw/(2*np.pi)**0.5

class Monomial():
    def __init__(self):
        self.name='Monomial'
        self.syschar='m'
        self.rv_name=None # no distribution
        self.rv_fun=None
        self.supp=np.array([-np.inf, np.inf])
        self.eval=lambda ord, xi: xi**ord
        # self.poly=nppol.HermiteE
        self.norm=lambda p: 1 # redundant
        # weight=lambda x: nppol.hermite_e.hermeweight(x)/(2*np.pi)**0.5,
        # self.weight=lambda x: np.exp(-x**2/2)/(2*np.pi)**0.5


poly={'p': Legendre(),
      't': Chebyshev_1nd(),
      'u': Chebyshev_2nd(),
      'h': Hermite(),
      'm': Monomial(),
      }


class GPCpoly():
    """
    Polynomials for GPC.
    """
    @staticmethod
    def get_dist(syschars):
        # get distributions corresponding to orthogonal polynomials
        dist=[]
        for syschar in syschars:
            dist.append(poly[syschar].rv_fun)
        return dist

    @staticmethod
    def get_supp(syschars):
        # get distributions corresponding to orthogonal polynomials
        dist=[]
        for syschar in syschars:
            dist.append(poly[syschar].supp)
        return dist

def find_poly_intervals(p):
    """
    Find the intervals of 1D-polynomial (numpy.polynomial) where the polynomial is negative.
    """
    assert(np.abs(p.coef[-1]) > 1e-14)
    r=p.roots()
    # remove imaginary roots, multiple roots, and sort
    r=np.unique(np.extract(np.abs(r.imag)<1e-14, r).real)

    ints = []
    for ii in range(r.size-1):
        rmean = 0.5*(r[ii]+r[ii+1])
        if p(rmean)<0:
            ints.append([r[ii],r[ii+1]])

    sign_pinf = np.sign(p.coef[-1])
    if p.coef[-1] < 0: # polynomial sign at plus infinity
        ints.append([r[-1], np.inf])
    if (-1)**p.degree()*sign_pinf<0: # polynomial sign at minus infinity
        ints.append([-np.inf, r[0]])

    return np.array(ints)

# def polyval(pn, xi):
#     """
#     Evaluate a polynomial pn at points xi.
#     """
#     val=np.zeros_like(xi.flatten())
#     sign_pinf=np.sign(pn.coef[-1]) # sign at infinity
#     sign_minf=(-1)**pn.degree()*sign_pinf # sign at minus infinity
#     indp=np.argwhere(xi.flatten()==np.inf).squeeze()
#     val[indp]=sign_pinf*np.inf # sign at infinity
#     indm=np.argwhere(xi.flatten()==-np.inf).squeeze()
#     val[indm]=sign_pinf*np.inf # sign at infinity
#     indother=np.setdiff1d(np.arange(xi.size), np.hstack([indp, indm]))
#     val[indother]=pn(xi.flatten()[indother])
#     return val.reshape(xi.shape)

if __name__=='__main__':
#     execfile('../test_gpc.py')
    pol=nppol.Polynomial([0,-1,1,2,3,4,1])
    ints=find_poly_intervals(pol)
    print(ints)
    print('end')
