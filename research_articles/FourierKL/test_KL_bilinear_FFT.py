#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#mport matplotlib.pyplot as pl
import matplotlib.pylab as plt

import numpy as np
np.set_printoptions(precision=6)
np.set_printoptions(linewidth=999999)

from uq.decomposition import KL_Fourier
import os

dim=2

## get the reference eigen values from a FFT with large n
n=3**7
N=n*np.ones(dim, dtype=np.int)
kl=KL_Fourier(covfun=1, cov_pars={'rho':0.15}, N=N, puc_size=np.ones(dim))

kl.calc_modes(method=-1,relerr=0.01)

print(kl.ct.val[(n-1)/2-1:(n-1)/2+2, (n-1)/2-1:(n-1)/2+2] )

k_ref=1
mean_index=(n-1)/2
eigVal_ref= kl.ct.val[mean_index-k_ref: mean_index+k_ref+1,
                      mean_index-k_ref: mean_index+k_ref+1]
print(eigVal_ref)
##############################

m_list=np.array(range(1,19,2))*3
err_trig=np.zeros((m_list.shape[0],))
err_bili=np.zeros((m_list.shape[0],))

for i in range(m_list.shape[0]):

    N=m_list[i]*np.ones(dim, dtype=np.int)
    mean_index=(m_list[i]-1)/2

    kl=KL_Fourier(covfun=1, cov_pars={'rho':0.15}, N=N, puc_size=np.ones(dim))

    kl.calc_modes(method=-1,relerr=0.01)

    err_trig[i]= np.linalg.norm(eigVal_ref - kl.ct.val[mean_index-k_ref: mean_index+k_ref+1, mean_index-k_ref: mean_index+k_ref+1])

    kl.calc_modes(method=1,relerr=0.01)

    err_bili[i]= np.linalg.norm(eigVal_ref - kl.ct.val[mean_index-k_ref: mean_index+k_ref+1, mean_index-k_ref: mean_index+k_ref+1])

print(err_trig)
print(err_bili)

fig, ax2 = plt.subplots()
fig.set_size_inches(5 , 3.5 , forward=True)

ax2.plot(m_list, np.log(err_trig),   linewidth=1 , marker='o' , markersize=2, label="trig")
ax2.plot(m_list, np.log(err_bili),   linewidth=1 , marker='o' , markersize=2, label="bili")
plt.title('error of eigen values by FFT and by bilinear approx.')
plt.ylabel('log(2-norm error)')
plt.xlabel('m')
plt.legend(loc='upper right')
picname = 'error_in_eigenValues' +'.png'

plt.savefig(picname)
os.system('eog'+' '+picname +' '+ '&')
