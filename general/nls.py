import numpy as np
from general.base import Timer, Struct

def solve_Newton(F, derF, X, tol=1e-6, maxiter=1e2):
    res_log = []
    dif_log = []
    res = 1.
    iters = 0
    while res > tol and iters < maxiter:
        iters += 1
        X_prev = X.copy()
        dX = np.linalg.solve(derF(X), F(X))
        X = X - dX
        dif = np.linalg.norm(X-X_prev)
        res = np.linalg.norm(F(X))
        dif_log.append(dif)
        res_log.append(res)

    info = Struct(tol=tol, maxiter=maxiter, iters=iters, res=res_log)
    return X, info

def fixed_point_iteration(F, X, tol=1e-6, maxiter=1e2):
    res_log = []
    dif_log = []
    res = 1.
    iters = 0
    while res > tol and iters < maxiter:
        iters += 1
        X_prev = X
        X = F(X)
        dif = np.linalg.norm(X-X_prev)
        res = np.linalg.norm(F(X)-X)
        dif_log.append(dif)
        res_log.append(res)
    info = {'tol': tol,
            'maxiter': maxiter,
            'iters': iters,
            'res': res_log}
    return X, info

# class NewtonRaphson():
# 
#     def __init__(self):
