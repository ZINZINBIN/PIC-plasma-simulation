import numpy as np
from typing import Union
from numba import jit

@jit(nopython=True)
def Gaussian_Elimination_TriDiagonal(A_origin : np.ndarray, B_origin : np.ndarray):
    A = np.copy(A_origin)
    B = np.copy(B_origin).reshape(-1)
    
    m,n = A.shape[0], A.shape[1]
            
    X = np.zeros_like(B)
    
    # Forward Elimination
    for idx_j in range(1,n):
        A[idx_j, idx_j] = A[idx_j, idx_j] - A[idx_j, idx_j - 1] * A[idx_j-1, idx_j] / A[idx_j-1, idx_j-1]
        B[idx_j] = B[idx_j] - A[idx_j, idx_j - 1] * B[idx_j - 1] / A[idx_j-1, idx_j-1]
        A[idx_j, idx_j-1] = 0
        
    # Backward Substitution 
    X[-1] = B[-1] / A[-1,-1]
    for idx_i in [n - i - 1 for i in range(1,n)]:
        X[idx_i] = 1 / A[idx_i, idx_i] * (B[idx_i] - A[idx_i, idx_i +1] * X[idx_i + 1])

    return X

@jit(nopython=True)
def Gaussian_Elimination_Improved(A: np.ndarray, B: np.ndarray, gamma:float = 5.0):
    A_new = np.copy(A)
    A_new[0,0] -= gamma
    A_new[-1,-1] -= A[0,-1] * A[-1,0] / gamma
    A_new[-1,0] = 0
    A_new[0,-1] = 0

    u = np.zeros((A.shape[1],1))
    u[0,0] = gamma
    u[-1,0] = A[-1,0]

    v = np.zeros((A.shape[1],1))
    v[0,0] = 1
    v[-1,0] = A[0,-1] / gamma

    x1 = Gaussian_Elimination_TriDiagonal(A_new, B)
    q = Gaussian_Elimination_TriDiagonal(A_new, u)

    x = x1 - q * np.dot(v.reshape(1,-1),x1.reshape(-1,1)) / (1 + np.dot(v.reshape(1,-1),q.reshape(-1,1)))
    return x

def SOR(A:np.ndarray, B:np.ndarray, x_origin:np.ndarray, w = 1.0, n_epoch:int = 1, eps:float = 1e-8):
    
    A_diag = A.diagonal()
    D = np.diag(A_diag)
    p = A - D
    L, U = np.zeros_like(p), np.zeros_like(p)

    rows = D.shape[0]
    cols = D.shape[1]

    for row in range(0, rows):
        for col in range(0, cols):
            if row > col:
                L[row, col] = p[row, col]
            elif row < col:
                U[row, col] = p[row, col]
            else:
                pass
    
    # SOR alogorithm
    xi = np.copy(x_origin)
    xf = np.copy(x_origin)
    
    def _update(xl:np.ndarray, xr:np.ndarray):
        for idx in range(0, xr.shape[0]):
            xr[idx] = (1-w) * xl[idx] + w / (A_diag[idx] + 1e-16 * np.sign(A_diag[idx])) * (B[idx] - L[idx, :]@xr - U[idx, :]@xl)
        return xl,xr
    
    def _compute_relative_error(x:np.ndarray):
        err = np.sum(np.abs(A@x-B)) / len(B)
        return err
    
    for _ in range(n_epoch):
        
        _, xf = _update(xi,xf)
        
        err = _compute_relative_error(xf)
        
        if err < eps:
            break

        xi = np.copy(xf)
            
    return xf