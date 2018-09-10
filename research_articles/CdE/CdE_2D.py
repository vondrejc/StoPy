"""
examples of Conditioned expectation (CdE) in 2D
"""

from general import Struct
import numpy as np
import sympy
from uq.dist import Dists
from uq.gpc import GPC
from uq.quadrature import Quad_Gauss, Quad_rectangular, Quad_MC
from uq.update import PCE_CE, ConditionedExpectation
import os
import sys


if __name__=='__main__':
    debug=0
else:
    debug=1
    sys.stdout = open(os.devnull, 'w')

debug=1
# DEFINITION OF PARAMETERS ##############################
if debug:
    p=Struct(CdE=Struct(pint=1e2,
                        ngauss=15,
                        nMC=1e2),
             fig=Struct(Npl=10)
             )
else:
    p=Struct(CdE=Struct(pint=1e6,
                        ngauss=95,
                        nMC=1e6),
             fig=Struct(Npl=int(2e2)))

## Set up the model ##########################
p+=Struct(Y_Q=lambda q: q[0]*(q[0]+1.5)*(q[0]-1.5)+q[1]*(q[1]+1.5)*(q[1]-1.5)-1.*q[0]*q[1],
          q_true=-np.array([1.0, 1.0]), # The true parameter
          E=Struct(var=0.4),
          )

y_true=p.Y_Q(p.q_true) # The true measurement value at q_true
Q=GPC(2*'h', p=2*[1], p_max=1, rshape=(p.q_true.size,))
Q.project(lambda xi: xi, p_int=2)
f_Q=Dists(keys=2*['norm'], args=[(), ()], kargs=[{'loc': 0., 'scale': 1.},
                                                  {'loc': 0., 'scale': 1.}])

## Error ######################
E_dist=Dists(keys=['norm'], args=[()], kargs=[{'loc': 0., 'var': p.E.var}]) # variance 0.1
E=GPC('h', p=[1], p_max=1, rshape=(1,))
E.expand(E_dist.dists[0], quad={'p_int':3})
theta=Dists(keys=['norm'], args=[()], kargs=[{'loc': 0., 'var': 1.}])

print('== MEAN by PCE conditional expectation =======================')
Y=lambda xi: p.Y_Q(Q(xi))
Y_E=lambda xi: Y(xi[:2])+E(xi[2])
X=lambda xi: Q(xi[:2])

phis=[]
for pord in [1,5,10,15,20]:
    phi=GPC(pchars='h', name='hermite', p=[pord], rshape=(2,))
    quad=Quad_Gauss(Q.pchars+E.pchars, p_int=np.hstack([3*Q.p, E.p[0]])*phi.p[0]+1)
    pce_ce=PCE_CE(Y_E, phi, quad)
    phi=pce_ce(X)
    print('mean(pce ord={1})={0}'.format(phi(y_true), pord))
    phis.append(phi)

print('\n== MEAN by Conditioned expectation ==================')
quad=Quad_rectangular(pchars=Q.pchars, p_int=p.CdE.pint, name='')
CdE=ConditionedExpectation(Y=Y, E=E, yhat=y_true, quad=quad, theta=theta)
mean=CdE.mean(Q)
print('mean = {0}; quadrature rectangular\n'.format(mean))

for quad_alternative in [Quad_Gauss(pchars=Q.pchars, p_int=15, name='Gauss'),
                         Quad_MC(Q, p_int=1e2, name='MonteCarlo')]:
    CdE=ConditionedExpectation(Y=Y, E=E, yhat=y_true, quad=quad_alternative, theta=theta)
    mean_alternative=CdE.mean(Q)
    print('mean = {0}; quadrature {1}\n'.format(mean_alternative, quad_alternative.name))

print('\n== COVARIANCE by CdE =======================================')
CdE=ConditionedExpectation(Y=Y, E=E, yhat=y_true, quad=quad, theta=theta)
covar=CdE.covariance(Q, mean=mean)
print('covar = {0}'.format(covar))
print(sympy.latex(sympy.Matrix(covar)))
eig, vec = np.linalg.eigh(covar)

print('eig(covar)={}'.format(eig))
print('eigvec(covar)={}'.format(vec))

#########################
## PLOTTING #############
if not os.path.exists('figures/'):
    os.makedirs('figures/')

import matplotlib as mpl
import matplotlib.pyplot as pl
import figures_par
parf=figures_par.set_pars(mpl)

print('\nplotting...')
xran=(-2, 2.5)
yran=(-2, 2.5)
xpl=np.linspace(xran[0], xran[1], p.fig.Npl)
ypl=np.linspace(yran[0], yran[1], p.fig.Npl)
xlabel='Parameter $q_1$'
ylabel='Parameter $q_2$'

coord=np.array(np.meshgrid(xpl, ypl, sparse=False, indexing='xy'))
Z=p.Y_Q(coord.reshape([2, p.fig.Npl**2])).reshape([p.fig.Npl, p.fig.Npl])

## 1: response surface #############
fig=pl.figure(figsize=parf['figsize'], dpi=parf['dpi'])
CS=pl.contour(coord[0], coord[1], Z-y_true, levels=[-7,-5,-3,-1.,0,1.,3,5,7,10,15])
zc = CS.collections[4]
pl.setp(zc, linewidth=3)
zcN=CS.collections[:4]
pl.setp(zcN, linestyle='--')

pl.clabel(CS, inline=1, fontsize=10, fmt='%1.1f', inline_spacing=50)
pl.xlabel(xlabel)
pl.ylabel(ylabel)
pl.axis('square')
pl.xlim(*xran)
pl.ylim(*yran)

fname='figures/fig2d_prior.pdf'
pl.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')

## 2: expectation by pdf #############
likelihood=lambda xi: E.pdf(y_true-Y(xi))
Z=likelihood(coord.reshape([2, coord[0].size])).reshape(coord[0].shape)

pl.figure(figsize=parf['figsize'], dpi=parf['dpi'])
CS=pl.contour(coord[0], coord[1], Z, levels=[0.05, 0.3, 0.55, 0.65])
pl.clabel(CS, inline=1, fontsize=10, fmt='%1.2f', inline_spacing=50)
pl.xlim(*xran)
pl.ylim(*yran)
pl.xlabel(xlabel)
pl.ylabel(ylabel)
pl.axis('square')
apost=lambda xi: E.pdf(y_true-Y(xi))*f_Q.pdf(xi)
Z=apost(coord.reshape([2, coord[0].size])).reshape(coord[0].shape)
fname='figures/fig2d_likelihood.pdf'
pl.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')

## 3: posterior distribution #########
pl.figure(figsize=parf['figsize'], dpi=parf['dpi'])
CS=pl.contour(coord[0], coord[1], Z, levels=[0.005, 0.02, 0.04, 0.06, 0.08])
manual_loc=None
pl.clabel(CS, inline=1, fontsize=10, inline_spacing=50, fmt='%1.3f', manual_locations=manual_loc)

pl.plot(phis[0](y_true)[0], phis[0](y_true)[1], '+r', label='$\Phi^1_{Q|Z}(\hat{y})$ (Kalman filter)')
pl.plot(phis[2](y_true)[0], phis[2](y_true)[1], 'xr', label='$\Phi^{10}_{Q|Z}(\hat{y})$')
pl.plot(mean[0], mean[1], 'ob', label='$\widetilde{\Phi}_{Q|Z}(\hat{y})$')

# def get_ellipse_coor(A):
#     eig, vecs=np.linalg.eigh(A)
# 
#     phis=np.linspace(0, 2*np.pi)
#     x = lambda phi: eig[0]**0.5*np.cos(phi)
#     y = lambda phi: eig[1]**0.5*np.sin(phi)
#     coors = []
#     for phi in phis:
#         coor = vecs.dot(np.array([x(phi), y(phi)]))
#         coors.append(coor)
#     return np.array(coors).T
#
# coor = get_ellipse_coor(covar)
pl.axis('square')
pl.xlim(*xran)
pl.ylim(*yran)
pl.xlabel(xlabel)
pl.ylabel(ylabel)
pl.legend(loc='best')
fname='figures/fig2d_apost.pdf'
pl.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')

print('END')
sys.stdout = sys.__stdout__
