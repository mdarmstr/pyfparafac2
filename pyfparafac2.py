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
    return Xk, sz

def pyfparafac2als(Xk,R):

def pyfparafac2(Xk,R):
    Xk, sz = pyfparafac2parse(Xk)




if __name__ == '__main__':
    Xk = pyfparafac2parse("roi_2.npy")
    print(Xk)

