import numpy as np
from typing import Union

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