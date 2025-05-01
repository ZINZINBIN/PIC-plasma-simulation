import numpy as np
import scipy as sp
from tqdm.auto import tqdm
from typing import Literal, Optional
from src.util import compute_hamiltonian
from src.solve import Gaussian_Elimination_Improved, SOR, Jacobi
from src.integration import explicit_midpoint, leapfrog, verlet, implicit_midpoint
from src.interpolate import CIC, TSC
from src.dist import BasicDistribution

# 1D two-stream instability simulation by solving 1D vlasov equation solver using PIC method
class PIC:
    np.random.seed(42)
    def __init__(
        self,
        N: int = 40000,
        N_mesh: int = 400,
        n0: float = 1.0,
        L: float = 50.0,
        dt: float = 1.0,
        tmin: float = 0.0,
        tmax: float = 50.0,
        gamma: float = 5.0,
        vth: float = 1.0,
        vb: float = 3.0,
        A: float = 0.1,
        n_mode:int = 5,
        method: Literal["midpoint", "leapfrog", "verlet", "implicit"] = "leapfrog",
        solver: Literal["SOR", "Gauss"] = "Gauss",
        interpol: Literal["CIC", "TSC"] = "CIC",
        simcase: Literal["two-stream", "bump-on-tail"] = "two-stream",
        init_dist: Optional[BasicDistribution] = None,
    ):

        # setup
        self.N = N                  # num of particles
        self.N_mesh = N_mesh        # num of mesh cell
        self.n0 = n0                # average electron density
        self.L = L                  # box size
        self.dt = dt                # time difference
        self.tmin = tmin            # minimum time
        self.tmax = tmax            # maximum time
        self.gamma = gamma          # parameters for solving linear equation of variant form of Tri-diagonal matrix

        # particle information
        self.dx = L / N_mesh

        self.simcase = simcase      # simulation case info

        # Two-stream instability
        self.A = A                  # Amplitude of the perturbation
        self.vth = vth              # Thermal velocity of electrons
        self.vb = vb                # Beam velocity 
        self.n_mode = n_mode        # mode number of perturbation

        # Bump-on-tail instability
        self.init_dist = init_dist

        # algorithms to be used
        self.method = method
        self.solver = solver
        self.interpol = interpol

        # Field quantities
        self.phi_mesh = None
        self.E_mesh = None

        # initialize the distribution
        self.initialize(simcase)

        # density computation
        self.update_density()

        # Gradient field and laplacian of potential
        self.grad = np.zeros((N_mesh, N_mesh))
        self.laplacian = np.zeros((N_mesh, N_mesh))

        # Mesh for 1st derivative and 2nd derivative
        self.generate_grad()
        self.generate_laplacian()

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key) is True and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def initialize(self, simcase: Literal["two-stream", "bump-on-tail"] = "two-stream"):
        if simcase == "two-stream":
            self.x = np.random.rand(self.N,1) * self.L                 # initialize the position of particles
            self.v = self.vth * np.random.randn(self.N,1) + self.vb    # initialize the velocity of particles
            self.a = np.zeros((self.N,1))

            # Initial condition
            Nh = int(self.N/2)
            self.Nh = Nh
            self.v[Nh:] *= -1                                                 # anti-symmetric configuration
            self.v *= (1 + self.A * np.sin(2 * self.n_mode * np.pi * self.x / self.L))      # add perturbation

        elif simcase == "bump-on-tail":
            x, v = self.init_dist.get_sample()
            self.x = x.reshape(-1, 1)
            self.v = v.reshape(-1, 1)

            # Initial condition
            self.v *= (1 + self.A * np.sin(2 * np.pi * self.n_mode * self.x / self.L))  # add perturbation

        # check CFL condition for stability
        if self.dt > 2 / np.sqrt(self.N / self.L):
            self.dt = 2 / np.sqrt(self.N / self.L)

    def generate_grad(self):
        dx = self.L / self.N_mesh

        for idx_i in range(0,self.N_mesh):
            if idx_i > 0:
                self.grad[idx_i, idx_i - 1] = -1.0

            if idx_i < self.N_mesh - 1:
                self.grad[idx_i, idx_i + 1] = 1.0

        # periodic condition
        self.grad[0,self.N_mesh - 1] = -1.0
        self.grad[self.N_mesh - 1,0] = 1.0

        self.grad /= 2*dx

    def generate_laplacian(self):
        dx = self.L / self.N_mesh

        for idx_i in range(0,self.N_mesh):
            if idx_i > 0:
                self.laplacian[idx_i, idx_i - 1] = 1.0

            if idx_i < self.N_mesh - 1:
                self.laplacian[idx_i, idx_i + 1] = 1.0

            self.laplacian[idx_i, idx_i] = -2.0

        self.laplacian[0,self.N_mesh-1] = 1.0
        self.laplacian[self.N_mesh-1,0] = 1.0

        self.laplacian /= dx ** 2

    def update_density(self):
        if self.interpol == "CIC":
            n, indx_l, indx_r, weight_l, weight_r = CIC(self.x, self.n0, self.L, self.N, self.N_mesh, self.dx)

        elif self.interpol == "TSC":
            n, indx_l, indx_m, indx_r, weight_l, weight_m, weight_r = TSC(self.x, self.n0, self.L, self.N, self.N_mesh, self.dx)

        self.n = n
        self.indx_l = indx_l
        self.indx_r = indx_r
        self.weight_l = weight_l
        self.weight_r = weight_r

        if self.interpol == "TSC":
            self.indx_m = indx_m
            self.weight_m = weight_m

    def compute_density(self, x:np.ndarray, dx:float, N:int, N_mesh:int, n0:float, L:float, return_all:bool = False):
        if self.interpol == "CIC":
            n, indx_l, indx_r, weight_l, weight_r = CIC(x, n0, L, N, N_mesh, dx)

        elif self.interpol == "TSC":
            n, indx_l, indx_m, indx_r, weight_l, weight_m, weight_r = TSC(x, n0, L, N, N_mesh, dx)

        if return_all:
            if self.interpol == "CIC":
                return n, weight_l, weight_r, indx_l, indx_r

            elif self.interpol == "TSC":
                return n, weight_l, weight_m, weight_r, indx_l, indx_m, indx_r
        else:
            return n

    def update_E_field(self):

        self.phi_mesh = self.linear_solve(self.laplacian, self.n - self.n0, self.phi_mesh, self.gamma)

        self.E_mesh = (-1) * self.grad @ self.phi_mesh

        if self.interpol == "CIC":
            self.E = self.weight_l * self.E_mesh[self.indx_l[:, 0]] + self.weight_r * self.E_mesh[self.indx_r[:, 0]]
            self.phi = self.weight_l * self.phi_mesh[self.indx_l[:, 0]] + self.weight_r * self.phi_mesh[self.indx_r[:, 0]]

        elif self.interpol == "TSC":
            self.E = self.weight_l * self.E_mesh[self.indx_l[:, 0]] + self.weight_m * self.E_mesh[self.indx_m[:,0]] + self.weight_r * self.E_mesh[self.indx_r[:, 0]]
            self.phi = self.weight_l * self.phi_mesh[self.indx_l[:, 0]] + self.weight_m * self.phi_mesh[self.indx_m[:, 0]] + self.weight_r * self.phi_mesh[self.indx_r[:, 0]]

    def compute_E_field(self, x:np.ndarray, dx:float, N:int, N_mesh:int, n0:float, L:float, return_all:bool = False):

        if self.interpol == "CIC":
            n, w_l, w_r, idx_l, idx_r = self.compute_density(x,dx,N,N_mesh,n0,L,True)

        elif self.interpol == "TSC":
            n, w_l, w_m, w_r, idx_l, idx_m, idx_r = self.compute_density(x,dx,N,N_mesh,n0,L,True)

        phi_mesh = self.linear_solve(self.laplacian, n - n0, self.phi_mesh, self.gamma).reshape(-1,1)

        E_mesh = (-1) * np.matmul(self.grad, phi_mesh)

        if self.interpol == "CIC":
            E = w_l * E_mesh[idx_l[:,0]] + w_r * E_mesh[idx_r[:,0]]
            phi = w_l * phi_mesh[idx_l[:, 0]] + w_r * phi_mesh[idx_r[:, 0]]

        elif self.interpol == "TSC":
            E = w_l * E_mesh[idx_l[:,0]] + w_m * E_mesh[idx_m[:,0]] + w_r * E_mesh[idx_r[:,0]]
            phi = w_l * phi_mesh[idx_l[:, 0]] + w_m * phi_mesh[idx_m[:,0]] + w_r * phi_mesh[idx_r[:, 0]]

        if return_all:
            return E, phi, E_mesh, phi_mesh
        else:
            return E

    def update_acc(self):
        self.update_E_field()
        self.a = -self.E

    def compute_grad(self, eta:np.ndarray):

        x,v = eta[:len(eta)//2], eta[len(eta)//2:]
        grad_eta = []
        grad_eta.append(v)

        a = (-1) * self.compute_E_field(x, self.dx, self.N, self.N_mesh, self.n0, self.L, False)

        grad_eta.append(a)
        grad_eta = np.concatenate(grad_eta, axis = 0)
        return grad_eta

    def update_motion(self):
        eta = np.concatenate([self.x.reshape(-1,1), self.v.reshape(-1,1)], axis = 0)

        if self.method == "midpoint":
            eta_f = explicit_midpoint(eta, self.compute_grad, self.dt)

        elif self.method == "leapfrog":
            eta_f = leapfrog(eta, self.compute_grad, self.dt)

        elif self.method == "verlet":
            eta_f = verlet(eta, self.compute_grad, self.dt)

        elif self.method == "implicit":
            eta_f = implicit_midpoint(eta, self.compute_grad, self.dt)

        x = eta_f[:len(eta_f)//2]
        v = eta_f[len(eta_f)//2:]

        x = np.mod(x, self.L)
        self.x = x
        self.v = v

        self.update_density()
        self.update_acc()

    def linear_solve(self, A: np.ndarray, B:np.array, x_ref:Optional[np.ndarray] = None, gamma : float = 5.0):

        if x_ref is None and self.solver == "SOR":
            x = sp.linalg.solve(A, B, assume_a="gen")
            
        else:
            if self.solver == "SOR":
                x = Jacobi(A, B, x_ref, w=2/3, n_epoch=128, eps=1e-10)
                # x = SOR(A, B, x_ref, 2/3, 128, 1e-8)

            elif self.solver == "Gauss":
                x = Gaussian_Elimination_Improved(A, B, gamma)

        return x.reshape(-1,1)

    def solve(self):

        Nt = int(np.ceil((self.tmax - self.tmin) / self.dt))

        # initialize density
        self.update_density()

        # initialize acc
        self.update_acc()

        # snapshot
        pos_list = []
        vel_list = []
        E_list = []
        KE_list = []
        PE_list = []

        E, KE, PE = compute_hamiltonian(self.v, self.E_mesh, self.dx, True)

        pos_list.append(np.copy(self.x))
        vel_list.append(np.copy(self.v))
        E_list.append(E)
        KE_list.append(KE)
        PE_list.append(PE)

        for i in tqdm(range(Nt), 'PIC simulation process'):
            self.update_motion()

            pos_list.append(np.copy(self.x))
            vel_list.append(np.copy(self.v))

            E, KE, PE = compute_hamiltonian(self.v, self.E_mesh, self.dx, True)
            E_list.append(E)
            KE_list.append(KE)
            PE_list.append(PE)

        print("# Simputation process end")

        qs = np.concatenate(pos_list, axis = 1)
        ps = np.concatenate(vel_list, axis = 1)
        snapshot = np.concatenate([qs, ps], axis=0)

        E = np.array(E_list)
        KE = np.array(KE_list)
        PE = np.array(PE_list)

        return snapshot, E, KE, PE
