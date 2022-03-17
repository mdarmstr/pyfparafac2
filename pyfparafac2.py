# Flexible Coupling (Non-negative) PARAFAC2
# As described by Cohen and Bro

import numpy as np
import scipy as sp
import random as rnd
from fnnls import fnnls

def pyfparafac2parse(Xk): #This is working; can either open an .npz or an .npy file
    if Xk.endswith(".npz"):
        Xk = np.load(Xk)
        Xk = Xk.f.arr_0
        sz = np.shape(Xk)
    elif Xk.endswith(".npy"):
        Xk = np.load(Xk)
        sz = np.shape(Xk)
        return Xk, sz #may have some redundancies, let's worry about that later :)

def pyfparafac2als(Xk, R):

    sz = np.shape(Xk)
    Pk = np.zeros((sz[0], R, sz[2]))
    mk = np.zeros((1, sz[2]))
    Bk = np.random.rand(sz[0], R, sz[2])
    Dk = np.repeat(np.eye(R), sz[2], axis=2)
    A = np.random.rand(sz(1), R)
    Xh = np.zeros(1, sz[2])
    BsT = Dk
    BkDk = np.zeros((sz[0], R, sz[2]))

    for kk in range(sz[2]):

        U, S, V = sp.sparse.linalg.svds(Xk[:, :, kk], R)
        Pk[:, :, kk] = U.dot(np.transpose(V))
        Xh[kk] = Bk[:, :, kk].dot(Dk[:, :, kk]).dot(np.transpose(A))
        mk[kk] = np.linalg.norm(np.ravel(Xk[:, :, kk]) - np.ravel(Xh[:, :, kk]))**2/R**2
        ssr1 = np.linalg.norm(np.ravel(Xk) - np.ravel(Xh))**2

    ssr2 = 0
    iterNo = 1
    maxIter = 1000
    eps = 1e-8
    YNorm = np.linalg.norm(np.ravel(Xk))**2
    ssr1 = ssr1/YNorm

    while (ssr1-ssr2)/ssr2 > eps and (ssr1 - ssr2) > eps and iterNo < maxIter:

    ssr1 = ssr2

    #Pk estimation
    for kk in range(sz[2]):
        if iterNo > 1:
            U, S, V = sp.sparse.linalg.svds(Xk[:, :, kk])
            Pk[:,:,kk] = U.dot(np.transpose(V))
            #Bs estimation
            BsT[:, :, kk] = mk[kk]*np.transpose(Pk[:, :, kk]).dot(Bk[:, :, kk])
            BkDk[:, :, kk] = Bk[:, :, kk].dot(Dk[:, :, kk])





def pyfparafac2(Xk, R):
    Xk = pyfparafac2parse(Xk)
    Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk,R)
    return Bk, A, Dk, Bs, ssr, pvar


if __name__ == '__main__':
    Xk = pyfparafac2parse("roi_2.npy")
    print(Xk)

