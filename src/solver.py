import numpy as np
import scipy as sp
from tqdm.auto import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from src.utils import Gaussian_Elimination_TriDiagonal

# 1D two-stream instability simulation by solving 1D vlasov equation solver using PIC method
# linear algebra solver: conjugate-gradient method

class PICsolver:
    def __init__(
        self, 
        N : int, 
        N_mesh : int, 
        n0:float,
        L : float, 
        dt : float, 
        tmin : float, 
        tmax : float, 
        gamma : float,
        vth : float,
        vb : float,
        use_animation:bool=True,
        plot_freq : int = 4,  
        save_dir : str = "./result/simulation.gif"
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
        self.use_animation = use_animation
        self.plot_freq = plot_freq
        self.save_dir = save_dir
        
        # particle information
        self.dx = L / N_mesh
        
        np.random.seed(42)
        self.x = np.random.rand(N,1) * L            # initialize the position of particles
        self.v = vth * np.random.randn(N,1) + vb    # initialize the velocity of particles
        self.a = np.zeros((N,1))
        
        Nh = int(N/2)
        self.Nh = Nh
        self.v[Nh:-1] *= -1                                     # anti-symmetric configuration
        self.v *= (1 + 0.1 * np.sin(2 * np.pi * self.x / L))    # add perturbation
        
        # index paramters for updating each mesh grid
        self.indx_l = np.floor(self.x / self.dx).astype(int)
        self.indx_r = self.indx_l + np.ones_like(self.indx_l)
        
        self.weight_l = (self.indx_r * self.dx - self.x) / self.dx
        self.weight_r = (self.x - self.indx_l * self.dx) / self.dx
        
        # periodic BC
        self.indx_r = np.mod(self.indx_r, N_mesh)
        
        # electron density
        self.n = np.bincount(self.indx_l[:,0], weights=self.weight_l[:,0], minlength=N_mesh)
        self.n += np.bincount(self.indx_r[:,0], weights=self.weight_r[:,0], minlength=N_mesh)
        
        self.n *= self.n0 * self.L / self.N / self.dx
        
        # gradient field and laplacian of potential
        self.grad = np.zeros((N_mesh, N_mesh))
        self.laplacian = np.zeros((N_mesh, N_mesh))
        
        self.generate_grad()
        self.generate_laplacian()
        
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
    
    def linear_solve(self, A : np.ndarray, B : np.array, gamma : float = 5.0):
        
        A_new = np.copy(A)
        A_new[0,0] -= gamma
        A_new[-1,-1] -= A[0,-1] * A[-1,0] / gamma

        u = np.zeros((A.shape[1],1))
        u[0,0] = gamma
        u[-1,0] = A[-1,0]
        
        v = np.zeros((A.shape[1],1))
        v[0,0] = 1
        v[-1,0] = A[0,1] / gamma
        
        if B.ndim == 1:
            B = B.reshape(-1,1)
        
        x1 = Gaussian_Elimination_TriDiagonal(A_new, B)
        q = Gaussian_Elimination_TriDiagonal(A_new,u)
        
        x = x1 - q * np.dot(v.reshape(1,-1),x1.reshape(-1,1)) / (1 + np.dot(v.reshape(1,-1),x1.reshape(-1,1)))
        
        return x.reshape(-1,1)
    
    def solve(self):
        
        Nt = int(np.ceil((self.tmax - self.tmin) / self.dt))
        
        # we have to compute initial acceleration
        self.update_acc()
        
        if self.use_animation:
            pos_list = []
            vel_list = []
            
            pos_list.append(self.x)
            vel_list.append(self.v)
            
        else:
            pos_list = None
            vel_list = None
        
        for i in tqdm(range(Nt), 'PIC simulation process'):
            
            # velocity update
            self.update_velocity()
            
            # position update
            self.update_position()
            
            # density update
            self.update_density()
            
            # acceleration update
            self.update_acc()
            
            # velocity update with 1/2 kick
            self.update_velocity()
            
            if pos_list is not None:
                pos_list.append(self.x)
                vel_list.append(self.v)
                
        plt.figure(figsize = (6,4))
        plt.scatter(self.x[0:self.Nh],self.v[0:self.Nh],s=.5,color='blue', alpha=0.5)
        plt.scatter(self.x[self.Nh:], self.v[self.Nh:], s=.5,color='red',  alpha=0.5)
        plt.xlabel("x pos")
        plt.ylabel("vel")
        plt.xlim([0,self.L])
        plt.ylim([-10, 10])
        plt.tight_layout()
        plt.savefig("./result/PIC.png", dpi=160)
        print("# Computation process end")
            
        if self.use_animation:
            print("# Generating animation file")
            fig, ax = plt.subplots(1,1,figsize = (6,4), facecolor = 'white', dpi=160)
            
            def _plot(idx : int, ax:Axes):
                pos = pos_list[idx]
                vel = vel_list[idx]
                
                ax.cla()
                ax.clear()
                ax.scatter(pos[0:self.Nh],vel[0:self.Nh],s=.5,color='blue', alpha=0.5)
                ax.scatter(pos[self.Nh:], vel[self.Nh:], s=.5,color='red',  alpha=0.5)
                ax.set_xlabel("x pos")
                ax.set_ylabel("vel")
                ax.set_xlim([0,self.L])
                ax.set_ylim([-10, 10])

            replay = lambda idx : _plot(idx, ax)
            idx_max = int((self.tmax - self.tmin) / self.dt)
            indices = [i for i in range(idx_max)]
            ani = animation.FuncAnimation(fig, replay, frames = indices)
            writergif = animation.PillowWriter(fps = self.plot_freq)
            ani.save(self.save_dir, writergif)
        
            print("# Complete")
    
    def update_velocity(self):
        self.v += self.a * self.dt / 2.0
  
    def update_position(self):
        self.x += self.v * self.dt
        self.x = np.mod(self.x, self.L)
    
    def update_acc(self):
        
        # update field for calculating the acceleration
        self.phi_mesh = self.linear_solve(self.laplacian, self.n - self.n0, self.gamma)
        self.E_mesh = (-1) * np.matmul(self.grad, self.phi_mesh)
        E = self.weight_l * self.E_mesh[self.indx_l[:,0]] + self.weight_r * self.E_mesh[self.indx_r[:,0]]

        # update acceleration
        self.a = -E
    
    def update_density(self):
        self.indx_l = np.floor(self.x / self.dx).astype(int)
        self.indx_r = self.indx_l + np.ones_like(self.indx_l)
        
        self.weight_l = (self.indx_r * self.dx - self.x) / self.dx
        self.weight_r = (self.x - self.indx_l * self.dx) / self.dx
        
        self.indx_r = np.mod(self.indx_r, self.N_mesh)
        
        self.n = np.bincount(self.indx_l[:,0], weights=self.weight_l[:,0], minlength=self.N_mesh)
        self.n += np.bincount(self.indx_r[:,0], weights=self.weight_r[:,0], minlength=self.N_mesh)
        
        self.n *= self.n0 * self.L / self.N / self.dx