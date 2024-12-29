import numpy as np
from typing import Union
from scipy.linalg import lu
from numba import jit

def compute_hamiltonian(v:np.array, E:np.array):
    H = 0.5 * np.sum(v * v) + 1 / 8 / np.pi * np.sum(E*E)
    return H

def matmul_vec(A : np.ndarray, x : np.array):
    result = np.zeros_like(x)
    n = len(x)
    for idx_i in range(0, n):
        s = 0
        for idx_j in range(0,n):
            s += A[idx_i, idx_j] * x[idx_j]
        result[idx_i] = s
    return result

def norm(vec : np.array):
    n = len(vec)
    result = 0
    for i in range(n):
        result += vec[i] ** 2
    result /= n
    return np.sqrt(result)

def Gaussian_Elimination_TriDiagonal(A_origin : np.ndarray, B_origin : Union[np.array, np.ndarray]):
    A = np.copy(A_origin)
    B = np.copy(B_origin)
    
    m,n = A.shape[0], A.shape[1]
    assert m == n, 'matrix should be n x n shape'
    
    if type(B) == np.ndarray:
        B = B.reshape(-1,)
        
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

def SORsolver(A_origin:np.ndarray, x_origin:np.ndarray, b_origin:np.ndarray, w = 1.0):
    
    A = np.copy(A_origin)
    x = np.copy(x_origin)
    b = np.copy(b_origin)

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
    
    right_term = w * b - (w * U + (w-1) * D) @ x
    
    # SOR alogorithm per 1 epoch
    x_new = np.zeros_like(x)

    for idx in range(0, x_new.shape[0]):
        x_new[idx] = (1-w) * x[idx] + w / (A_diag[idx] + 1e-8 * np.sign(A_diag[idx])) * (b[idx] - L[idx, :]@x_new - U[idx, :]@x)

    return x_new

def QRsolver(A_origin, b_origin):
    ''' solve Grad-Shafranov Equation using cholesky factorization
    - A.T @ A => positive definite matrix
    - solve A.T @ A x = A.T b with LR or QR factorization
    - method
    (1) LR factorization
        - L@R = A.T@A
        - inverse matrix of L and R can be computed by converting sign of components
    (2) QR factorization
        - using Gram-schmit Method, get orthogonal vector set 
        - Q : matrix [v1, v2, v3, .... vn]
    '''
    
    def forward_sub(L,b):
        ''' Forward Subsititution Algorithm for solving Lx = b 
        - L : Lower Triangular Matrix, m*n
        - x : n*1
        - b : m*1
        '''
        
        m = L.shape[0]
        n = L.shape[1]
        x_new = np.zeros((n,1))
        
        for idx in range(0,m):
            x_new[idx] = b[idx] - L[idx,:]@x_new
            x_new[idx] /= L[idx,idx]
        
        return x_new

    def backward_sub(U,b):
        ''' Backward Subsititution Algorithm for solving Lx = b 
        - L : Lower Triangular Matrix, m*n
        - x : n*1
        - b : m*1
        '''
        m = U.shape[0]
        n = U.shape[1]
        x_new = np.zeros((n,1))
        
        for idx in range(m-1,-1,-1):
            x_new[idx] = b[idx] - U[idx, :]@x_new
            x_new[idx] /= U[idx,idx]
        
        return x_new
    
    
    A = np.copy(A_origin)
    b = np.copy(b_origin)
    
    A_ = A.T@A
    b_ = A.T@b
    
    P,L,U = lu(A_, overwrite_a = True, check_finite = True)
    
    # pivoting 
    b_ = P.T@b_
    
    # solve LUx = P@b_
    # using Forward and Back Subsititue Algorithm
    # (1) Forward subsitution : LX = B
    # (2) Backward subsitution : UX = B
    
    x_new = backward_sub(U, forward_sub(L, b_))

    return x_new.reshape(-1,)