import numpy as np
from typing import Callable

def forward_euler(eta: np.ndarray, grad_func: Callable, dt:float):
    return eta + dt * grad_func(eta)

def explicit_midpoint(eta: np.ndarray, grad_func: Callable, dt: float):
    grad_eta = grad_func(eta)
    grad_eta_half = grad_func(eta + 0.5 * dt * grad_eta)
    eta_f = eta + dt * grad_eta_half
    return eta_f

def explicit_RK4(eta:np.ndarray, grad_func:Callable,dt:float):
    grad_eta_1 = grad_func(eta)
    grad_eta_2 = grad_func(eta + 0.5 * dt * grad_eta_1)
    grad_eta_3 = grad_func(eta + 0.5 * dt * grad_eta_2)
    grad_eta_4 = grad_func(eta + dt * grad_eta_3)
    return eta + 1 / 6 * dt * (grad_eta_1 + grad_eta_2 * 2 + grad_eta_3 * 2 + grad_eta_4)

def leapfrog(eta:np.ndarray, grad_func:Callable,dt:float):
    eta_x, eta_v = np.copy(eta[: len(eta) // 2]), np.copy(eta[len(eta) // 2 :])
    grad_eta = grad_func(eta)
    grad_eta_v = grad_eta[len(eta) // 2 :]

    eta_v_half = eta_v + grad_eta_v * 0.5 * dt

    eta_x += eta_v_half * dt
    eta_n = np.copy(eta)
    eta_n[: len(eta_n) // 2] = eta_x

    grad_eta_ = grad_func(eta_n)
    grad_eta_v_half = grad_eta_[len(eta_n) // 2 :]

    eta_v = eta_v_half + grad_eta_v_half * 0.5 * dt

    eta_n[len(eta_n) // 2 :] = eta_v
    return eta_n

def verlet(eta: np.ndarray, grad_func: Callable, dt: float):    
    eta_x, eta_v = np.copy(eta[: len(eta) // 2]), np.copy(eta[len(eta) // 2 :])
    grad_eta = grad_func(eta)
    grad_eta_x, grad_eta_v = grad_eta[: len(eta) // 2], grad_eta[len(eta) // 2:]

    eta_x += grad_eta_x * dt + 0.5 * grad_eta_v * dt ** 2

    eta[: len(eta) // 2] = eta_x
    grad_eta_ = grad_func(eta)
    grad_eta_v_ = grad_eta_[len(eta)//2:]

    eta_v += 0.5 * (grad_eta_v + grad_eta_v_) * dt
    eta[len(eta) // 2 :] = eta_v
    return eta

def backward_euler(eta: np.ndarray, grad_func: Callable, dt: float, n_epochs:int = 64, eps:float = 1e-8):
    x_init = forward_euler(eta, grad_func, dt)

    def _g(x: np.ndarray, dt: float, xn: np.ndarray):
        return xn + dt * grad_func(0.5 * (x + xn))

    x = x_init

    for _ in range(n_epochs):
        x_prev = np.copy(x)
        x = _g(x, dt, x_init)

        if np.linalg.norm(x - x_prev) < eps:
            break

    return x

def implicit_midpoint(eta: np.ndarray, grad_func: Callable, dt: float, n_epochs:int = 64, eps:float = 1e-8):
    
    x_init = explicit_midpoint(eta, grad_func, dt)
    
    def _g(x:np.ndarray, dt:float, xn:np.ndarray):
        return xn + dt * grad_func(0.5 * (x + xn))
    
    x = x_init

    for _ in range(n_epochs):
        x_prev = np.copy(x)
        x = _g(x, dt, x_init)

        if np.linalg.norm(x-x_prev) < eps:
            break
        
    return x
