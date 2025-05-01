import numpy as np
from typing import Union, Optional
from numba import jit

def Jacobi(A:np.ndarray, B:np.ndarray, x0:Optional[np.ndarray] = None, w:float=2/3, n_epoch:int = 16, eps:float = 1e-8):
    N = A.shape[0]
    x = np.zeros(N, dtype = float) if x0 is None else x0.copy().reshape(-1) 
    
    D = A.diagonal()

    for k in range(n_epoch):
        
        res = B - A@x
        x_new = x + w * res / D
        
        if np.linalg.norm(res) / N < eps:
            break
        
        x = x_new

    return x_new

@jit(nopython=True)
def Gaussian_Elimination_TriDiagonal(A_origin:np.ndarray, B_origin:np.ndarray):
    A = A_origin.copy()
    B = B_origin.copy().reshape(-1)
    
    n = A.shape[0]
            
    X = np.zeros_like(B)
    
    # Forward Elimination
    for idx_j in range(1,n):
        A[idx_j, idx_j] = A[idx_j, idx_j] - A[idx_j, idx_j - 1] * A[idx_j-1, idx_j] / A[idx_j-1, idx_j-1]
        B[idx_j] = B[idx_j] - A[idx_j, idx_j - 1] * B[idx_j - 1] / A[idx_j-1, idx_j-1]
        A[idx_j, idx_j-1] = 0
        
    # Backward Substitution 
    X[-1] = B[-1] / A[-1,-1]
    for idx_i in range(n-2,-1,-1):
        X[idx_i] = (B[idx_i] - A[idx_i, idx_i +1] * X[idx_i + 1]) / A[idx_i, idx_i]

    return X

@jit(nopython=True)
def Gaussian_Elimination_Improved(A: np.ndarray, B: np.ndarray, gamma:float = 5.0):
    
    N = A.shape[0]

    if A.shape[0] != A.shape[1] or B.shape[0] != N:
        raise ValueError("Matrix A must be square and B must have compatible dimensions.")

    A_new = A.copy()
    A_new[0,0] -= gamma
    A_new[-1,-1] -= A[0,-1] * A[-1,0] / gamma
    A_new[-1,0] = 0.0
    A_new[0,-1] = 0.0

    u = np.zeros(N, dtype = float)
    u[0] = gamma
    u[-1] = A[-1,0]

    v = np.zeros(N, dtype = float)
    v[0] = 1
    v[-1] = A[0,-1] / gamma

    x = Gaussian_Elimination_TriDiagonal(A_new, B)
    q = Gaussian_Elimination_TriDiagonal(A_new, u)
    
    x -= q * np.dot(v,x) / (1 + np.dot(v,q))
    return x

@jit(nopython=True)
def SOR(A:np.ndarray, B:np.ndarray, x_origin:np.ndarray, w = 1.0, n_epoch:int = 1, eps:float = 1e-8):

    N = len(A)
    A_diag = np.array([A[i,i] for i in range(N)])
    
    D = np.zeros((N, N))
    
    for i in range(N):
        D[i, i] = A_diag[i]
        
    p = A - D
    L, U = np.zeros_like(p), np.zeros_like(p)

    for row in range(0, N):
        for col in range(0, N):
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
