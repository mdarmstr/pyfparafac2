# Flexible Coupling (Non-negative) PARAFAC2
# As described by Cohen and Bro

import numpy as np
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
    Pk = np.zeros((sz[0], R, sz[3]))
    for kk in range(sz[2]):
        U, S, V = np.linalg.svd(Xk[:, :, kk], R)
        Pk[:, :, kk] = U.dot(np.transpose(V))



def pyfparafac2(Xk, R):
    Xk = pyfparafac2parse(Xk)
    Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk,R)
    return Bk, A, Dk, Bs, ssr, pvar


if __name__ == '__main__':
    Xk = pyfparafac2parse("roi_2.npy")
    print(Xk)

