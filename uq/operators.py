import numpy as np
from numpy.linalg import norm, eigh

def tensor_product(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    ab = np.vstack([np.tile(a, b.shape[1]),
                    np.repeat(b, a.shape[1], axis=1)])
    return ab

def matrix_fun(A, fun, pseudotol=1e-14, nulzero=False):
    assert(isinstance(A, np.ndarray))

    if norm(A-A.T) < 1e-14: # symmetric matrix
        eigs, eigvecs =eigh(A, UPLO='L')
        assert(eigs.min()>1e-12)
#        ind = 0
#        eig = eigs[ind]
#        vec = eigvecs[:, ind]
#        print 'zero', norm(A.dot(vec) - eig*vec)
#        print 'zero', norm(A-eigvecs.dot(np.diag(eigs).dot(eigvecs.T)))
#        print 'zero', norm(A-np.einsum('i,ji,ki->jk', eigs, eigvecs, eigvecs))
        if nulzero:
            fun_eigs = np.zeros_like(eigs)
            inds = np.nonzero(eigs>pseudotol)
            print('{0} eigs below tol ({1})'.format(eigs.size-len(inds), pseudotol))
            fun_eigs[inds] = fun(eigs[inds])
        else:
            fun_eigs = fun(eigs)
        return np.einsum('i,ji,ki->jk', fun_eigs, eigvecs, eigvecs)
    else:
        raise NotImplementedError()
