import numpy as np
from scipy.sparse.linalg import svds
from fnnls import fnnls
import matplotlib.pyplot as plt
import tensorly as tl

def krb(A,B): #KRB product
    C = np.einsum('ij,kj -> ikj', A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])
    return C

class nnparafac2parse:
    def __init__(self, dataLoc):
        self.dataLoc = dataLoc
        
    def loadData(self, dataLoc):
        if self.dataLoc.endswith(".npz"):
            dataFile = np.load(dataLoc)
            self.Xk = dataFile.f.arr_0
        elif self.dataLoc.endswith(".npy"):
            dataFile = np.load(dataLoc)
            self.Xk = dataFile.astype("float64")
        
    def returnData(self):
        return self.Xk
    
class initnnparafac2:
    def __init__(self, tnsr, noComponents, Bki = None, Dki = None, Ai = None, Bsi = None):
        if Bki == None:
            Bki = np.random.rand(tnsr.shape[0], noComponents)
            self.__Bki = Bki / np.linalg.norm(Bki, axis=0)
        else:
            self.__Bki = Bki
            
        if Dki == None:
            self.__Dki = np.repeat(np.eye(noComponents), tnsr.shape[2], axis=2)
        else:
            self.__Dki = Dki
            
        if Ai == None:
            Ai = np.random.rand(tnsr.shape[1], noComponents)
            self.__Ai = Ai / np.linalg.norm(Ai, axis=0)
        else:
            self.__Ai = Ai
        
        if Bsi == None:
            self.__Bsi = np.eye(noComponents)
        else:
            self.__Bsi = Bsi
    
    def getBki(self):
        return self.__Bki
    
    def getDki(self):
        return self.__Dki
    
    def getAi(self):
        return self.__Ai
    
    def getBsi(self):
        return self.__Bsi
    
class nnparafac2:
    def __init__(self, tnsr, noComponents, absThres=1e-6, relThres=1e-6, maxIter=1000, 
                 initial=None, isNormal=np.array([1,0,1],dtype=bool), isNonNeg=np.array([1,0,0],dtype=bool)):
        
        self.__tnsr = tnsr
        self.__noComponents = noComponents
        self.__absThres = absThres
        self.__relThres = relThres
        self.__maxIter = 1000
        self.__isNormal = isNormal
        self.__isNonNeg = isNonNeg
        self.__iterNo = 1
        
        if initial == None:
            initial = initnnparafac2(self.__tnsr, self.__noComponents)
            
        self.__Bk = initial.__Bki
        self.__Dk = initial.__Dki
        self.__A  = initial.__Ai
        self.__Bs = initial.__Bsi
        
        for kk in range(self.__tnsr.shape[2]):
            U, S, V = np.linalg.svd(self.__Bk[:, :, kk] @ self.__Bs)
            self.__Pk[:, :, kk] = U[:, :noComponents] @ V.T
            self.__resMdl[:, :, kk] = self.__tnsr - self.__Bk @ self.__Dk[:, :, kk] @ self.__A.T
            
            self.__mk[kk] = np.linalg.norm(self.__tnsr[:, :, kk] - self.__resMdl[:, :, kk]) ** 2 / ...
            np.linalg.norm(self.__Bk[:, :, kk]) ** 2
            
            self.__Bst[:, :, kk] = self.__mk[kk] * self.__Pk[:, :, kk].T @ self.__Bk[:, :, kk]
            self.__Bs = np.sum(self.__Bst, axis=2)
            self.__Bs /= np.linalg.norm(self.__Bs, axis=0)
            
            self.__resCpl[:, :, kk] = self.__Bk - self.__Pk @ self.__Bs
            
        self.__tnsr0 = tl.unfold(self.__tnsr, 0) #For the BkDk step.
        self.__BkDk = krb(tl.unfold(tnsr.__Bk,0), tl.unfold(tnsr.__Dk,0))
        
        # You forgot to optimise for Pk here.
        
        def optBs(self): #vectorize this
            for kk in range(self.__tnsr.shape[2]):
                self.__Bst[:, :, kk] = self.__mk[kk] * self.__Pk[:, :, kk].T @ self.__Bk[:, :, kk]
        
        def optA(self):
            for jj in range(self.__tnsr.shape[1]):
                d, res = fnnls(self.__BkDk.T @ self.__BkDk, self.__tnsr0[:, jj])
                self.__A[jj, :] = d
            
            self.__A /= np.linalg.norm(self.__A, axis=0)
            self.__A[np.isnan(self.__A)] = 0
            
        def optBk(self): #vectorize this
            for kk in range(self.__tnsr.shape[2]):
                for ii in range(self.__tnsr.shape[0]):
                    self.__Bk[ii, :, kk] = np.linalg.pinv(self.__Dk[:, :, kk] @ (self.__A.T @ self.__A) @ self.__Dk[:, :, kk] + self.__mk[kk] * np.eye(noComponents)) @ (self.__tnsr[ii, :, kk] @ self.__A @ self.__Dk[:, :, kk] + self.__mk[kk] * self.__Pk[:, :, kk] @ self.__Bs).T
            self.__Bk[:, :, kk] /= np.linalg.norm(self.__Bk[:, :, kk], axis=0)
        self.__Bk[np.isnan(self.__Bk)] = 0
        
        def optDk(self): #vectorize this
            for kk in range(self.__tnsr.shape[2]):
                self.__Dk[:, :, kk] = np.diag(np.diag(np.linalg.pinv(self.__Bk[:, :, kk].T @ self.__Bk[:, :, kk]) @ self.__Bk[:, :, kk].T @ self.__tnsr[:, :, kk] @ np.linalg.pinv(self.__A).T))
        
        def incMk(self):
            if self.__iterNo == 1:
                for kk in range(self.__tnsr.shape[2]):
                    Xt = self.__tnsr[:, :, kk] - np.mean(self.__tnsr[:, :, kk], axis=0) / np.std(self.__tnsr[:, :, kk], axis=0)
                    Xt[np.isnan(Xt)] = 0
                    U, S, V = svds(Xt, 2)
                    S.sort()
                    SNR = S[1] ** 2 / S[0] ** 2
                    self.__mk[kk] = 10 ** (-SNR / 10) * np.linalg.norm(self.__resMdl[:, :, kk]) ** 2 / np.linalg.norm(self.__Bk[:, :, kk]) ** 2
            
            elif self.__iterNo > 1 and self.__iterNo < 10:
                for kk in range(self.__tnsr.shape[2]):
                    self.__mk[kk] = np.minimum(self.__mk[kk] * 1.03, 1e12)
        
        def incIter(self):
            self.__iterNo += 1
        
        def fit(self):
            
            epsRel = self.__relThres
            epsAbs = self.__absThres
            
            yMdl = np.linalg.norm(np.ravel(self.__tnsr)) ** 2
            
            yCpl = np.linalg.norm(np.ravel(self.__Bk)) ** 2
            
            # ssr1 = 1
            ssr2 = 0.5 * np.linalg.norm(self.__resMdl, 'fro') ** 2 / yMdl  + 0.5 * np.linalg.norm(self.__resCpl, 'fro') ** 2 / yCpl
            
            while abs(ssr1 - ssr2) / ssr2 > relThres and abs(ssr1 - ssr2) > absThres:
                
                ssr1 = ssr2
                
                self.opt
        
            
            
        
        
        
        
        
        