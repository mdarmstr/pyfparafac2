import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.sparse.linalg import svds
from fnnls import fnnls
import matplotlib.pyplot as plt


# Flexible Coupling (Non-negative) PARAFAC2
# As described by Cohen and Bro
# [c] Michael Sorochan Armstrong, 2022


def pyfparafac2parse(Xk):  # This is working; can either open an .npz or an .npy file
    if Xk.endswith(".npz"):
        Xk = np.load(Xk)
        Xk = Xk.f.arr_0
    elif Xk.endswith(".npy"):
        Xk = np.load(Xk)
        Xk = Xk.astype("float64")
    return Xk  # may have some redundancies, let's worry about that later :)


def pyfparafac2als(Xk, R, eps, maxIter, displ, animate, Bk, A, Dk, Bs):
    # Initialisation
    sz = np.shape(Xk)
    Pk = np.zeros((sz[0], R, sz[2]))
    Bhk = np.zeros((sz[0], R, sz[2]))
    mk = np.zeros(sz[2])
    # Bk = np.random.rand(sz[0], R, sz[2])
    # Dk = np.eye(R)
    # Bs = np.eye(R)
    # Dk = np.repeat(Dk[:, :, np.newaxis], sz[2], axis=2)
    # A = np.random.rand(sz[1], R)
    Xh = np.zeros(sz)
    BkDk = np.zeros((sz[0], R, sz[2]))
    Xkij = np.zeros((sz[0] * sz[2], sz[1]))
    BkDkIK = np.zeros((sz[0] * sz[2], R))
    Bst = np.zeros((R, R, sz[2]))
    SSR = []
    res_mdl = []
    res_cpl = []

    for kk in range(sz[2]):
        U, S, V = np.linalg.svd(Bk[:, :, kk] @ Bs)
        Pk[:, :, kk] = U[:, :R] @ V.T
        mk[kk] = np.linalg.norm(Xk[:, :, kk] - Xh[:, :, kk]) ** 2 / (np.linalg.norm(Bk[:, :, kk]) ** 2) # The norm of Bk should be equal to the number of factors
        res_mdl.append(np.linalg.norm(Xk[:, :, kk] - Bk[:, :, kk] @ Dk[:, :, kk] @ A.T) ** 2)
        res_cpl.append(mk[kk] * np.linalg.norm(Bk[:, :, kk] - Pk[:, :, kk] @ Bs) ** 2)

    #ssr1 = np.linalg.norm(np.ravel(Xk) - np.ravel(Xh)) ** 2 + np.linalg.norm(np.ravel(Bk) - np.ravel(Bhk)) ** 2

    ssr2 = 1e-6
    iterNo = 1
    YNorm = np.linalg.norm(np.ravel(Xk)) ** 2
    ssr1 = sum(res_mdl + res_cpl) / YNorm

    # Loop
    while abs(ssr1 - ssr2) / ssr2 > eps and abs(ssr1 - ssr2) > eps and iterNo < maxIter:

        ssr1 = ssr2

        # Pk estimation
        for kk in range(sz[2]):
            if iterNo > 1:
                U, S, V = np.linalg.svd(Bk[:, :, kk] @ Bs)
                Pk[:, :, kk] = U[:, :R] @ V.T
                Bst[:, :, kk] = mk[kk] * Pk[:, :, kk].T @ Bk[:, :, kk]
                BkDk[:, :, kk] = Bk[:, :, kk] @ Dk[:, :, kk]
            else:
                Bst[:, :, kk] = mk[kk] * Pk[:, :, kk].T @ Bk[:, :, kk]
                BkDk[:, :, kk] = Bk[:, :, kk] @ Dk[:, :, kk]

        Bs = (1 / np.sum(mk)) * np.sum(Bst, axis=2)
        # Bs /= np.sum(mk)

        # for rr in range(R):
        #    Bs[:, rr] /= np.linalg.norm(Bs[:,rr])
        Bs = Bs / np.linalg.norm(Bs, axis=0)

        BkDkIK = BkDk.transpose(0, 2, 1).reshape((-1, R), order="F")

        if iterNo == 1:
            Xkij = Xk.transpose(0, 2, 1).reshape((-1, sz[1]), order="F")
            #Xkij = np.vstack(Xk)

        for jj in range(sz[1]):
            d, res = fnnls(BkDkIK.T @ BkDkIK, BkDkIK.T @ Xkij[:, jj])
            A[jj, :] = d

        #A = np.transpose(np.linalg.pinv(BkDkIK.T @ BkDkIK) @ BkDkIK.T @ Xkij)
        #A[A == 0] = 1e-20
        A = A / np.linalg.norm(A, axis=0)

        #A = np.nan_to_num(A)
        A[np.isnan(A)] = 0
        for kk in range(sz[2]):
            for ii in range(sz[0]):
                Bk[ii, :, kk] = np.linalg.pinv(Dk[:, :, kk] @ (A.T @ A) @ Dk[:, :, kk] + mk[kk] * np.eye(R)) @ (Xk[ii, :, kk] @ A @ Dk[:, :, kk] + mk[kk] * Pk[ii, :, kk] @ Bs).T
            #Bk[Bk == 0] = 1e-20
            Bk[:, :, kk] /= np.linalg.norm(Bk[:, :, kk], axis=0)
            #Bk[:, :, kk] = np.nan_to_num(Bk[:, :, kk])
            # Dk[:, :, kk] = np.diag(np.diag(np.linalg.pinv(Bk[:, :, kk]) @ Xk[:, :, kk] @ np.linalg.pinv(A.T)))
            Dk[:, :, kk] = np.diag(np.diag(np.linalg.pinv(Bk[:, :, kk].T @ Bk[:, :, kk]) @ Bk[:, :, kk].T @ Xk[:, :, kk] @ np.linalg.pinv(A).T)) #@ A @ np.linalg.pinv(A.T @ A)))

        Bk[np.isnan(Bk)] = 0

        if iterNo == 1:
            for kk in range(sz[2]):
                Xt = Xk[:, :, kk] - np.mean(Xk[:, :, kk], axis=0) / np.std(Xk[:, :, kk], axis=0)
                Xt[np.isnan(Xt)] = 0
                U1, S1, V1 = svds(Xt, k=2)
                S1.sort()
                SNR = S1[1] ** 2 / S1[0] ** 2
                mk[kk] = 10 ** (-SNR / 10) * (np.linalg.norm(Xk[:, :, kk] - Bk[:, :, kk] @ Dk[:, :, kk] @ A.T) ** 2 / np.linalg.norm(Bk[:, :, kk] - Pk[:, :, kk] @ Bs) ** 2)
        elif iterNo < 10:
            for kk in range(sz[2]):
                mk[kk] = np.minimum(mk[kk] * 1.05, 1e12)

        for kk in range(sz[2]):
            res_mdl[kk] = (np.linalg.norm(Xk[:, :, kk] - Bk[:, :, kk] @ Dk[:, :, kk] @ A.T, 'fro') ** 2)
            res_cpl[kk] = (mk[kk] * np.linalg.norm(Bk[:, :, kk] - Pk[:, :, kk] @ Bs, 'fro') ** 2)

        ssr2 = sum(res_mdl + res_cpl) / YNorm

        #for kk in range(sz[2]):
        #    Xh[:, :, kk] = Bk[:, :, kk] @ Dk[:, :, kk] @ A.T
        #    Bhk[:, :, kk] = np.linalg.norm(Bk[:, :, kk] - Pk[:, :, kk] @ Bs)

        #ssr2 = ((np.linalg.norm(np.ravel(Xk) - np.ravel(Xh))**2) + np.linalg.norm(np.ravel(Bhk))**2)/YNorm
        SSR.append(ssr2)

        if iterNo == 1:
            print("Iteration\t\t", "Absolute Error\t\t", "Relative Error\t\t", "SSR\t\t", "mk\n")
            print(iterNo, "\t\t", "%.4e" % abs(ssr2-ssr1), "\t\t", "%.4e" % (abs(ssr2-ssr1)/ssr2), "\t\t", "%.4e" % SSR[iterNo-1], "\t\t", "%.4e" % np.mean(mk), "\n")
        else:
            print(iterNo, "\t\t", "%.4e" % abs(ssr2-ssr1), "\t\t", "%.4e" % (abs(ssr2-ssr1)/ssr2), "\t\t", "%.4e" % SSR[iterNo-1], "\t\t", "%.4e" % np.mean(mk), "\n")

        if iterNo == 1:
            plt.clf()
            plt.plot(np.arange(sz[0]), Bk[:, :, 19] @ Dk[:, :, 19])
            #plt.plot(np.arange(sz[1]), A)
            plt.draw()
            plt.pause(0.001)
        else:
            plt.clf()
            #plt.plot(np.arange(sz[0]), Bk[:, :, 19]) #@ Dk[:, :, 19])
            plt.plot(np.arange(sz[1]), A)
            plt.draw()
            plt.pause(0.001)

        iterNo += 1

    pvar = 100*(1 - np.linalg.norm(np.ravel(Xk) - np.ravel(Xh)))

    plt.clf()
    plt.plot(np.arange(len(SSR)-1), np.diff(np.log(SSR)))
    plt.show()

    return Bk, A, Dk, Bs, SSR, pvar


def pyfparafac2(Xk, R, eps=1e-8, maxIter=1000, displ=True, animate=False, *args):

    Bsi = []
    Dki = []
    Ai = []
    Bki = []

    Xk = pyfparafac2parse(Xk)

    if len(args) == 4:
        Bsi = args[3]
        Dki = args[2]
        Ai = args[1]
        Bki = args[0]

        print('Utilising input initialisations, and readout controls')

        Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R, eps, maxIter, displ, animate, Bki, Ai, Dki, Bsi)

    elif len(args) == 3:
        Bsi = np.eye(R)
        Dki = args[2]
        Ai = args[1]
        Bki = args[0]

        print('Bs initialised as identity matrix. Utilising remaining input initialisations and readout controls')

        Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R, eps, maxIter, displ, animate, Bki, Ai, Dki, Bsi)

    elif len(args) == 2:
        Bsi = np.eye(R)
        Dki = np.eye(R)
        Dki = np.repeat(Dki[:, :, np.newaxis], np.size(Xk, 2), axis=2)
        Ai = args[1]
        Bki = args[0]

        print('Bs, and Dk initialised as identity matrices. Utilising remaining input initialisations and readout \n '
              'controls')

        Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R, eps, maxIter, displ, animate, Bki, Ai, Dki, Bsi)

    elif len(args) == 1:
        Bsi = np.eye(R)
        Dki = np.eye(R)
        Dki = np.repeat(Dki[:, :, np.newaxis], np.size(Xk, 2), axis=2)
        Ai = np.zeros((np.size(Xk, 1), R, 10))
        Bki = args[0]
        ssrInit = []

        print('Bs, and Dk initialised as identity matrices. A is randomly initialised. Utilising remaining input \n '
              'initialisations and readout controls. Best of 10 random initialisations.')

        for re in range(9):
            Ai[:, :, re] = np.random.rand(np.size(Xk, 1), R)
            Ai /= np.linalg.norm(Ai, axis=0)
            print('Testing initialisation ', re, 'out of ', 10)
            Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R, 1e-20, 20, displ, animate, Bki, Ai[:, :, re], Dki, Bsi)
            ssrInit.append(ssr[-1])
            del ssr

        initIndx = np.argmin(ssrInit)

        Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R, eps, maxIter, displ, animate, Bki, Ai[:, :, initIndx], Dki, Bsi)

    else:
        Bsi = np.eye(R)
        Dki = np.eye(R)
        Dki = np.repeat(Dki[:, :, np.newaxis], np.size(Xk, 2), axis=2)
        Ai = np.zeros((np.size(Xk, 1), R, 10))
        Bki = np.zeros((np.size(Xk, 0), R, np.size(Xk, 2), 10))
        ssrInit = []

        print('Bs, and Dk initialised as identity matrices. A and Bk are randomly initialised. Utilising remaining input \n '
              'initialisations and readout controls. Best of 10 random initialisations.')

        for re in range(9):
            Ai[:, :, re] = np.random.rand(np.size(Xk, 1), R)
            Ai /= np.linalg.norm(Ai, axis=0)
            Bki[:, :, :, re] = np.random.rand(np.size(Xk, 0), R, np.size(Xk, 2))

            for kk in range(np.size(Xk, 2)):
                Bki[:, :, kk, re] /= np.linalg.norm(Bki[:, :, kk, re], axis=0)

            print('Testing initialisation ', re, 'out of ', 10)
            Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R, 1e-20, 20, displ, animate, Bki[:, :, :, re], Ai[:, :, re], Dki, Bsi)
            ssrInit.append(ssr[-1])
            del ssr

        initIndx = np.argmin(ssrInit)

        Bk, A, Dk, Bs, ssr, pvar = pyfparafac2als(Xk, R, eps, maxIter, displ, animate, Bki[:, :, :, initIndx], Ai[:, :, initIndx], Dki, Bsi)

    return Bk, A, Dk, Bs, ssr, pvar


if __name__ == '__main__':
    Bk, A, Dk, Bs, ssr, pvar = pyfparafac2("roi_2.npy", 2)
    print(np.log(ssr))
