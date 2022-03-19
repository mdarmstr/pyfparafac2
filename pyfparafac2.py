# Flexible Coupling (Non-negative) PARAFAC2
# As described by Cohen and Bro
# [c] Michael Sorochan Armstrong, 2022

import numpy as np
import scipy.sparse.linalg
from sparsesvd import sparsesvd
from fnnls import fnnls


def pyfparafac2parse(Xk):  # This is working; can either open an .npz or an .npy file
    if Xk.endswith(".npz"):
        Xk = np.load(Xk)
        Xk = Xk.f.arr_0
    elif Xk.endswith(".npy"):
        Xk = np.load(Xk)
    return Xk  # may have some redundancies, let's worry about that later :)


def pyfparafac2als(Xk, R):
    # Initialisation
    sz = np.shape(Xk)
    Pk = Bhk = np.zeros((sz[0], R, sz[2]))
    mk = np.zeros((1, sz[2]))
    Bk = np.random.rand(sz[0], R, sz[2])
    Dk = np.eye(R)
    Dk = np.repeat(Dk[:, :, np.newaxis], sz[2], axis=2)
    A = np.random.rand(sz[1], R)
    Xh = np.zeros(sz)
    BsT = Dk
    BkDk = np.zeros((sz[0], R, sz[2]))
    Xkij = np.zeros((sz[0] * sz[2], sz[1]))
    SSR = np.zeros((1, 1))

    for kk in range(sz[2]):
        U, S, V = sparsesvd(Xk[:, :, kk], R)
        #svd = TruncatedSVD(n_components = R)
        Pk[:, :, kk] = U.dot(np.transpose(V))
        Xh[:, :, kk] = Bk[:, :, kk].dot(Dk[:, :, kk]).dot(np.transpose(A))
        mk[kk] = np.linalg.norm(np.ravel(Xk[:, :, kk]) - np.ravel(Xh[:, :, kk])) ** 2 / R ** 2 # The norm of Bk should be equal to the number of factors
        ssr1 = np.linalg.norm(np.ravel(Xk) - np.ravel(Xh)) ** 2

    ssr2 = 0
    iterNo = 1
    maxIter = 1000
    eps = 1e-8
    YNorm = np.linalg.norm(np.ravel(Xk)) ** 2
    ssr1 = ssr1 / YNorm

    #Loop
    while (ssr1 - ssr2) / ssr2 > eps and (ssr1 - ssr2) > eps and iterNo < maxIter:

        ssr1 = ssr2

        # Pk estimation
        for kk in range(sz[2]):
            if iterNo > 1:
                U, S, V = scipy.sparse.linalg.svds(Xk[:, :, kk], k=R)
                Pk[:, :, kk] = U.dot(np.transpose(V))
                # Bs estimation
                BsT[:, :, kk] = mk[kk] * np.transpose(Pk[:, :, kk]).dot(Bk[:, :, kk])
                BkDk[:, :, kk] = Bk[:, :, kk].dot(Dk[:, :, kk])
            else:
                BsT[:, :, kk] = mk[kk] * np.transpose(Pk[:, :, kk]).dot(Bk[:, :, kk])
                BkDk[:, :, kk] = Bk[:, :, kk].dot(Dk[:, :, kk])

        Bs = 1 / (np.sum(mk) * np.sum(BsT, axis=2))
        Bs /= np.linalg.norm(Bs, axis=0)

        BkDkIK = np.vstack(BkDk)

        if iterNo == 1:
            Xkij = np.vstack(Xk)

        A = fnnls(np.transpose(BkDkIK).dot(BkDkIK), np.transpose(BkDkIK).dot(Xkij))

        A /= np.linalg.norm(A, axis=0)

        A = np.nan_to_num(A)

        for kk in range(sz[2]):
            for ii in range(sz[0]):
                Bk[ii, :, kk] = np.linalg.pinv(
                    Dk[:, :, kk].dot(np.transpose(A).dot(A)).dot(Dk[:, :, kk]) + mk[kk] * np.eye(R)).dot(
                    np.transpose((Xk[ii, :, kk]).dot(A).dot(Dk[:, :, kk]) + mk[kk] * Pk[ii, :, kk].dot(Bs)))
            Bk[:, :, kk] /= np.linalg.norm(Bk, axis=0)
            Bk[:, :, kk] = np.nan_to_num(Bk[:, :, kk])
        Dk[:, :, kk] = np.diag(np.diag(np.linalg.pinv(np.transpose(Bk[:, :, kk]).dot(Bk[:, :, kk])).dot(
            (Bk[:, :, kk]).dot(Xk[:, :, kk]).dot(Xk[:, :, kk]).dot(np.linalg.pinv(A)))))

        if iterNo == 1:
            for kk in range(sz[2]):
                U, S, V = scipy.sparse.linalg.svds(Xk[:, :, kk] - np.mean(Xk[:, :, kk], axis=0))
                SNR = S[1] ** 2 / S[2] ** 2
                mk[kk] = 10 ** (-SNR / 10) * np.linalg.norm(
                    Xk[:, :, kk] - Bk[:, :, kk].dot(Dk[:, :, kk]).dot(np.transpose(A))) ** 2 / np.linalg.norm(
                    Bk[:, :, kk] - Pk[:, :, kk].dot(Bs)) ** 2
        elif iterNo < 10:
            for kk in range(sz[2]):
                mk[kk] = np.minimum(mk[kk] * 1.02, 1e12)

        for kk in range(sz[2]):
            Xh[:, :, kk] = Bk[:, :, kk].dot(Dk[:, :, kk]).dot(np.transpose(A))
            Bhk[:, :, kk] = mk[kk]*np.linalg.norm(Bk[:, :, kk] - Pk[:, :, kk].dot(Bs))

        ssr2 = np.sum(np.linalg.norm(np.ravel(Xk) - np.ravel(Xh))**2, Bhk**2)/YNorm
        SSR[iterNo] = ssr2

        if iterNo == 1:
            print("Iteration\t\t", "Absolute Error\t\t", "Relative Error\t\t", "SSR\t\t", "mk\n")
            print(iterNo, "\t\t", ssr2-ssr1, "\t\t", (ssr2-ssr1)/ssr2, "\t\t", SSR[iterNo], "\t\t", np.mean(mk), "\n")
        else:
            print(iterNo, "\t\t", ssr2-ssr1, "\t\t", (ssr2-ssr1)/ssr2, "\t\t", SSR[iterNo], "\t\t", np.mean(mk), "\n")

        iterNo += 1

    pvar = 100*(1 - np.linalg.norm(np.ravel(Xk) - np.ravel(Xh)))
    return Bk, A, Dk, Bs, SSR, pvar


def pyfparafac2(Xk, R):
    Xk = pyfparafac2parse(Xk)
    Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R)
    return Bk, A, Dk, Bs, ssr, pvar


if __name__ == '__main__':
    Xk = pyfparafac2("roi_2.npy", 3)
    print(Xk)
