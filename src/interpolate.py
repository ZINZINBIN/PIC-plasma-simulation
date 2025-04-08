import numpy as np
from typing import Optional

# Cell-In cell method
def CIC(x:np.ndarray, n0:float, L:float, N:int, N_mesh:int, dx:float):
    
    x = np.mod(x, L)
    
    indx_l = np.floor(x / dx).astype(int)
    indx_r = indx_l + 1
    
    weight_l = (indx_r * dx - x) / dx
    weight_r = (x - indx_l * dx) / dx
    
    indx_r = np.mod(indx_r, N_mesh)
    
    n = np.bincount(indx_l[:,0], weights = weight_l[:,0], minlength=N_mesh)
    n += np.bincount(indx_r[:,0], weights = weight_r[:,0], minlength=N_mesh)
    n *= n0 * L / N / dx

    return n, indx_l, indx_r, weight_l, weight_r

def TSC(x:np.ndarray, n0:float, L:float, N:int, N_mesh:int, dx:float):
    
    x = np.mod(x, L)

    indx_m = np.floor(x / dx).astype(int)

    dist = (x - indx_m * dx) / dx

    weight_l = 0.5 * (1.5 - dist) ** 2
    weight_m = 0.75 - (dist - 1) ** 2
    weight_r = 0.5 * (dist - 0.5) ** 2

    indx_l = np.mod(indx_m - 1, N_mesh)
    indx_m = np.mod(indx_m, N_mesh)
    indx_r = np.mod(indx_m + 1, N_mesh)

    n = np.bincount(indx_m[:, 0], weights=weight_m[:, 0], minlength=N_mesh)
    n += np.bincount(indx_l[:, 0], weights=weight_l[:, 0], minlength=N_mesh)
    n += np.bincount(indx_r[:, 0], weights=weight_r[:, 0], minlength=N_mesh)

    n *= n0 * L / N / dx
    
    return n, indx_l, indx_m, indx_r, weight_l, weight_m, weight_r