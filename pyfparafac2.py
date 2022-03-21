# Flexible Coupling (Non-negative) PARAFAC2
# As described by Cohen and Bro
# [c] Michael Sorochan Armstrong, 2022

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.sparse.linalg import svds

from fnnls import fnnls

import matplotlib.pyplot as plt

def pyfparafac2parse(Xk):  # This is working; can either open an .npz or an .npy file
    if Xk.endswith(".npz"):
        Xk = np.load(Xk)
        Xk = Xk.f.arr_0
    elif Xk.endswith(".npy"):
        Xk = np.load(Xk)
        Xk = Xk.astype("float64")
    return Xk  # may have some redundancies, let's worry about that later :)


def pyfparafac2als(Xk, R):
    # Initialisation
    sz = np.shape(Xk)
    Pk = np.zeros((sz[0], R, sz[2]))
    Bhk = np.zeros((sz[0], R, sz[2]))
    mk = np.zeros((1, sz[2]))
    Bk = np.random.rand(sz[0], R, sz[2])
    Dk = np.eye(R)
    Bs = np.eye(R)
    Dk = np.repeat(Dk[:, :, np.newaxis], sz[2], axis=2)
    A = np.random.rand(sz[1], R)
    Xh = np.zeros(sz)
    BkDk = np.zeros((sz[0], R, sz[2]))
    Xkij = np.zeros((sz[0] * sz[2], sz[1]))
    SSR = []

    for kk in range(sz[2]):
        U, S, V = np.linalg.svd(Bk[:, :, kk] @ Bs)
        # svd = TruncatedSVD(n_components = R)
        Pk[:, :, kk] = U[:, :R].dot(np.transpose(V))
        Xh[:, :, kk] = Bk[:, :, kk].dot(Dk[:, :, kk]).dot(np.transpose(A))
        mk[0, kk] = np.linalg.norm(np.ravel(Xk[:, :, kk]) - np.ravel(Xh[:, :, kk])) ** 2 / (np.linalg.norm(Bk[:, :, kk]) ** 2) # The norm of Bk should be equal to the number of factors
        Bhk[:, :, kk] = Bk[:, :, kk] - Pk[:, :, kk] @ Bs
    ssr1 = np.linalg.norm(np.ravel(Xk) - np.ravel(Xh)) ** 2 + np.linalg.norm(np.ravel(Bk) - np.ravel(Bhk)) ** 2

    ssr2 = 1e-6
    iterNo = 1
    maxIter = 1000
    eps = 1e-8
    YNorm = np.linalg.norm(np.ravel(Xk)) ** 2
    ssr1 /= YNorm

    #Loop
    while abs(ssr1 - ssr2) / ssr2 > eps and abs(ssr1 - ssr2) > eps and iterNo < maxIter:

        ssr1 = ssr2

        # Pk estimation
        for kk in range(sz[2]):
            if iterNo > 1:
                U, S, V = np.linalg.svd(Bk[:, :, kk].dot(Bs))
                Pk[:, :, kk] = U[:, :R].dot(np.transpose(V))
                # Bs estimation
                Bs += Pk[:, :, kk].T @ Bk[:, :, kk]
                BkDk[:, :, kk] = Bk[:, :, kk] @ (Dk[:, :, kk])
            else:
                Bs += np.transpose(Pk[:, :, kk]).dot(Bk[:, :, kk])
                BkDk[:, :, kk] = Bk[:, :, kk].dot(Dk[:, :, kk])

        # Bs = 1 / (np.sum(mk) * np.sum(Bs, axis=2))
        Bs /= np.sum(mk) ** (-1) * np.linalg.norm(Bs, axis=0)

        # BkDkIK = np.vstack(BkDk)
        BkDkIK = BkDk.transpose(0, 2, 1).reshape((-1, R), order="F")


        if iterNo == 1:
            Xkij = Xk.transpose(0, 2, 1).reshape((-1, sz[1]), order="F")

        for jj in range(sz[1]):
            d, res = fnnls(BkDkIK.T @ BkDkIK, BkDkIK.T @ Xkij[:, jj])
            A[jj, :] = d

        A /= np.linalg.norm(A, axis=0)

        A = np.nan_to_num(A)

        for kk in range(sz[2]):
            for ii in range(sz[0]):
                # Bk[ii, :, kk] = np.linalg.pinv(
                #   Dk[:, :, kk].dot(np.transpose(A).dot(A)).dot(Dk[:, :, kk]) + mk[0, kk] * np.eye(R)).dot(
                #    np.transpose((Xk[ii, :, kk]).dot(A).dot(Dk[:, :, kk]) + mk[0, kk] * Pk[ii, :, kk].dot(Bs)))
                Bk[ii, :, kk] = np.linalg.pinv(Dk[:, :, kk] @ (A.T @ A) @ Dk[:, :, kk] + mk[0, kk] * np.eye(R)) @ \
                                np.transpose(Xk[ii, :, kk] @ A @ Dk[:, :, kk] + mk[0, kk]*Pk[ii, :, kk] @ Bs)
            Bk[:, :, kk] /= np.linalg.norm(Bk[:, :, kk], axis=0)
            Bk[:, :, kk] = np.nan_to_num(Bk[:, :, kk])
        # Dk[:, :, kk] = np.diag(np.diag(np.linalg.pinv(np.transpose(Bk[:, :, kk]).dot(Bk[:, :, kk])).dot(np.transpose(Bk[:, :, kk])).dot(Xk[:, :, kk]).dot(np.linalg.pinv(np.transpose(A)))))
            #Dk[:, :, kk] = np.diag(np.diag(np.linalg.pinv(Bk[:, :, kk]).dot(Xk[:, :, kk]).dot(np.linalg.pinv(np.transpose(A)))))
            Dk[:, :, kk] = np.diag(np.diag(np.linalg.pinv(Bk[:, :, kk]) @ Xk[:, :, kk] @ np.linalg.pinv(A.T)))
        if iterNo == 1:
            for kk in range(sz[2]):
                U, S, V = svds(Xk[:, :, kk] - np.mean(Xk[:, :, kk], axis=0), k=2)
                S.sort()
                SNR = S[1] ** 2 / S[0] ** 2
                mk[0, kk] = 10 ** (-SNR / 10) * np.linalg.norm(Xk[:, :, kk] - Bk[:, :, kk] @ (Dk[:, :, kk]) @ (np.transpose(A))) ** 2 / np.linalg.norm(Bk[:, :, kk] - Pk[:, :, kk] @ (Bs)) ** 2
        elif iterNo < 10:
            for kk in range(sz[2]):
                mk[0, kk] = np.minimum(mk[0, kk] * 1.02, 1e12)

        for kk in range(sz[2]):
            Xh[:, :, kk] = Bk[:, :, kk] @ (Dk[:, :, kk]) @ A.T
            Bhk[:, :, kk] = np.linalg.norm(Bk[:, :, kk] - Pk[:, :, kk] @ Bs)

        # ssr2 = np.sum(np.linalg.norm(np.ravel(Xk) - np.ravel(Xh))**2, np.ravel(Bhk)**2)/YNorm
        ssr2 = ((np.linalg.norm(np.ravel(Xk) - np.ravel(Xh))**2) + np.linalg.norm(np.ravel(Bhk))**2)/YNorm
        SSR.append(ssr2)

        if iterNo == 1:
            print("Iteration\t\t", "Absolute Error\t\t", "Relative Error\t\t", "SSR\t\t", "mk\n")
            print(iterNo, "\t\t", abs(ssr2-ssr1), "\t\t", abs(ssr2-ssr1)/ssr2, "\t\t", SSR[iterNo-1], "\t\t", np.mean(mk), "\n")
        else:
            print(iterNo, "\t\t", abs(ssr2-ssr1), "\t\t", abs(ssr2-ssr1)/ssr2, "\t\t", SSR[iterNo-1], "\t\t", np.mean(mk), "\n")

        if iterNo == 1:
            plt.plot(np.arange(sz[0]), Bk[:, :, 20] @ Dk[:, :, 20])
            plt.draw()
            plt.pause(0.001)
        else:
            plt.clf()
            plt.plot(np.arange(sz[0]), Bk[:, :, 20] @ Dk[:, :, 20])
            plt.draw()
            plt.pause(0.001)

        iterNo += 1


    pvar = 100*(1 - np.linalg.norm(np.ravel(Xk) - np.ravel(Xh)))

    return Bk, A, Dk, Bs, SSR, pvar


def pyfparafac2(Xk, R):
    Xk = pyfparafac2parse(Xk)
    Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, 3)
    return Bk, A, Dk, Bs, ssr, pvar


if __name__ == '__main__':
    Bk, A, Dk, Bs, ssr, pvar = pyfparafac2("roi_2.npy", 3)
    print(np.log(ssr))
