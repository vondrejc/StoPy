"""
examples of Conditioned expectation (CdE) in 1D
"""

from general import Struct
import numpy as np
from uq.dist import Dists
from uq.gpc import GPC
from uq.quadrature import Quad_Gauss, Quad_rectangular
from uq.update import PCE_CE, ConditionedExpectation
from uq_trial.update import get_phi_pdf
import os
import sys


problem=1

if __name__=='__main__':
    debug=0
else:
    debug=1
    sys.stdout = open(os.devnull, 'w')

# debug=1
print('debug={}'.format(debug))


p=Struct(CdE=Struct(pint=1e4),
         bayes_pdf=Struct(tol=1e-8),
         fig=Struct(Npl=int(1e2),
                    y_nvals=30,
                    ),)

if debug:
    p+=Struct(CdE=Struct(pint=35),
              bayes_pdf=Struct(tol=1e-3),
              fig=Struct(Npl=int(10),
                         y_nvals=3,
                         ),)

if problem in [0]:
    # non-linear observation operator
    p+=Struct(name='Q3',
              Y_Q=lambda q: ((q+0.5)**3+q/2.),
              q_true=np.array([1.]), # The true parameter
              Q=Struct(mean=0., var=1.0),
              E=Struct(var=.4),
              fig=Struct(xran=(-3, 2),
                         yran=(-10, 12),)
              )

elif problem in [1]:
    # double-well observation operator
    p+=Struct(name='Q2',
              Y_Q=lambda q: (q)**2+0.5*q,
              q_true=np.array([1.0]), # The true parameter
              Q=Struct(mean=0., var=1.0),
              E=Struct(var=.1),
              fig=Struct(xran=(-2, 3),
                         yran=(-2, 8),
                         ),
              )
else:
    raise ValueError()

################################################################################
y_true=p.Y_Q(p.q_true) # The true measurement value at q_true
Y=lambda xi: p.Y_Q(Q(xi)) # measurement RV

## prior parameters ###########
f_Q=Dists(keys=['norm'], args=[()], kargs=[{'mean': p.Q.mean, 'var': p.Q.var}])
Q=GPC('h', p=[2], p_max=2, rshape=(p.q_true.size,))
Q.expand(f_Q)

## Error ######################
E_dist=Dists(keys=['norm'], args=[()], kargs=[{'loc': 0., 'var': p.E.var}])
E=GPC('h', p=[1], p_max=1, rshape=(1,))
E.expand(E_dist.dists[0], quad={'p_int':2}) # error with zero mean
theta=Dists(keys=['norm'], args=[()], kargs=[{'loc': 0., 'var': 1.}])

print('\n== MEAN and COVARIANCE by Bayesian updating with pdfs ==================')
mean_pdf, var_pdf=get_phi_pdf(np.array([y_true]), Q, E, p.Y_Q, tol=p.bayes_pdf.tol) # reference value
print('mean={0}; var={1}'.format(mean_pdf, var_pdf))

print('\n== MEAN by Conditioned expectation ==================')
quad=Quad_rectangular(pchars=Q.pchars, p_int=p.CdE.pint, name='')
CdE=ConditionedExpectation(Y=Y, E=E, yhat=y_true, quad=quad, theta=theta)
mean=CdE.mean(Q)
print('mean = {0}; Quad_rectangular'.format(mean))

print('\n== COVARIANCE by Conditioned expectation ==================')
covar=CdE.covariance(Q, mean=mean)
print('covar = {0}'.format(covar))

print('\n== MEAN by PCE conditional expectation ==')
Y_E=lambda xi: Y(xi[0])+E(xi[1])
X=lambda xi: Q(xi[0])

phis=[]
for pord in [1,5,15]: # iteration over polynomial order of PCE
    phi=GPC(pchars='h', name='phi', p=[pord], rshape=(1,))
    quad=Quad_Gauss(Q.pchars+E.pchars, p_int=np.array([3*Q.p[0], E.p[0]])*phi.p[0]+1)
    pce_CE=PCE_CE(Y_E, phi, quad)
    phi=pce_CE(X)
    print('mean(pce ord={1})={0}'.format(phi(y_true), pord))
    phis.append(phi)

"""################################################################################
## PLOTTING ################################################################################
################################################################################"""

print('\n== plotting =================')
if not os.path.exists('figures/'):
    os.makedirs('figures/')

print('\n== UPDATE covariance - CdE ==================')
y_nvals=np.linspace(*p.fig.yran, num=p.fig.y_nvals)
mean_CdE=np.zeros_like(y_nvals)
covar_CdE=np.zeros_like(y_nvals)
quad=Quad_rectangular(pchars=Q.pchars, p_int=p.CdE.pint, name='')

for ii, y in enumerate(y_nvals):
    CdE=ConditionedExpectation(Y=Y, E=E, yhat=y, quad=quad, theta=theta)
    mean_CdE[ii]=CdE.mean(Q)
    covar_CdE[ii]=CdE.covariance(Q, mean=mean_CdE[ii])

import matplotlib as mpl
import matplotlib.pyplot as pl
import figures_par

parf=figures_par.set_pars(mpl)

qpl=np.linspace(*p.fig.xran, num=p.fig.Npl)

## MEAN - PCE-CE: ###############################################################################
label_mean=r'$\mathrm{Mean}_{Q|Z}(y)=\Phi_{Q|Z}(y)$'
label_covar=r'$\mathrm{Covar}_{Q|Z}(y)=\Phi_{\bar{Q}\otimes\bar{Q}|Z}(y)$'
xlabel=r'Parameter $q$ / PDF of error (dotted lines)'
ylabel=r'Measurement $y$ / PDF of prior (dotted lines)'

fig=pl.figure(figsize=parf['figsize'], dpi=parf['dpi'])
pl.plot(qpl, p.Y_Q(qpl), 'k-', label='observation operator $Y_Q(q)$')
ypl=np.linspace(*p.fig.yran, num=1e3)
mean_pdf, var_pdf=get_phi_pdf(y_nvals, Q, E, p.Y_Q, tol=p.bayes_pdf.tol)
pl.plot(mean_pdf, y_nvals, 'b+-', label=label_mean)

markevery=50
pl.plot(phis[0](ypl), ypl, 'go--', label='$\Phi_{Q|Z}^1(y)$', markevery=markevery)
# pl.plot(phis[1](ypl), ypl, 'rD--', markersize=5, label='$\Phi_{Q|Z}^{5}(y)$', markevery=markevery)
pl.plot(phis[2](ypl), ypl, 'ms--', markersize=5, label='$\Phi_{Q|Z}^{15}(y)$', markevery=markevery)

pl.plot(2*E.pdf(ypl), ypl, 'c-.', label='PDF of error')
pl.plot(qpl, 5*Q.pdf(qpl), 'b:', label='PDF of prior')

pl.legend(loc='best')
pl.xlabel(xlabel)
pl.ylabel(ylabel)
pl.xlim(*p.fig.xran)
pl.ylim(*p.fig.yran)
pl.grid()
filen='figures/fig1d_{0}_mean_pol.pdf'.format(p.name)
if __name__=='__main__':
    pl.savefig(filen, dpi=parf['dpi'],
               pad_inches=parf['pad_inches'], bbox_inches='tight')

## MEAN by CdE: ###############################################################################
fig=pl.figure(figsize=parf['figsize'], dpi=parf['dpi'])
pl.plot(qpl, p.Y_Q(qpl), 'k-', label='observation operator $Y_Q(q)$')

pl.plot(mean_pdf, y_nvals, 'b+-', label=label_mean)
pl.plot(mean_CdE, y_nvals, 'rx--', label=r'$\widetilde{\Phi}_{Q|Z}(y)$')
pl.plot(2*E.pdf(ypl), ypl, 'c-.', label='PDF of error')
pl.plot(qpl, 5*Q.pdf(qpl), 'b:', label='PDF of prior')

pl.xlabel(xlabel)
pl.ylabel(ylabel)
pl.xlim(*p.fig.xran)
pl.ylim(*p.fig.yran)
pl.legend(loc=2)
pl.grid()
filen='figures/fig1d_{0}_mean.pdf'.format(p.name)
if __name__=='__main__':
    pl.savefig(filen, dpi=parf['dpi'],
               pad_inches=parf['pad_inches'], bbox_inches='tight')

## Covariance by CdE #########################################################
xis=np.copy(qpl)
fig=pl.figure(figsize=parf['figsize'], dpi=parf['dpi'])
pl.plot(var_pdf, y_nvals, 'b+-', label=label_covar)
pl.plot(covar_CdE, y_nvals, 'rx--', label=r'$\widetilde{\Phi}_{\bar{Q}\otimes\bar{Q}|Z}(y)$')

pl.ylim(*p.fig.yran)
pl.grid()
pl.legend(loc='best')
pl.ylabel(r'Measurement $y$')
pl.xlabel(r'Covariance')
filen='figures/fig1d_{0}_cov.pdf'.format(p.name)
if __name__=='__main__':
    pl.savefig(filen, dpi=parf['dpi'],
               pad_inches=parf['pad_inches'], bbox_inches='tight')

print('END')
sys.stdout = sys.__stdout__
